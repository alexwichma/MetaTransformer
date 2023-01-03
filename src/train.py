"""
Entry point for model training
"""


from __future__ import absolute_import, division, print_function

import logging
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dataset.MetagenomicReadDataset import ProcessingMetagenomicReadDataset
from models.loss import cross_entropy_loss, multi_level_cross_entropy_loss
from models.model_utils import get_label_transforms, read_transforms_for_input_layer, instantiate_model_by_str_name
from trainer import ClassificationTrainer
from utils.OptimizerManager import OptimizerManager
from utils.context_manager import training_configs
import utils.device_handler as DeviceHandler
from utils.metric_utils import multi_lv_precision_factory, multi_lv_recall_factory, precision, recall
from utils.torch_utils import train_collate_fn_padded, weights_init
from utils.utils import fix_random_seeds, load_class_weights, load_vocabulary


logger = logging.getLogger("train/main")


@training_configs
def main(cmd_line_args, cfg):
    print("Starting training script. Refer to the log file for progress information.")    
    fix_random_seeds()
    torch.backends.cudnn.benchmark = cfg.training.use_cudnn_benchmark

    # load vocabulary only when actually needed for the embedding and transformation
    vocab, vocab_size = None, 0
    if not cfg.mdl_common.input_module in ["lsh", "hash", "one_hot", "one_hot_embed", "bpe"]:
        vocab, vocab_size = load_vocabulary(cfg.paths.vocabulary_path)
    else:
        logger.info("Vocabulary is not loaded since {} is used".format(cfg.mdl_common.input_module))
    
    logger.debug("Loading model, embedding and transformations.")
    read_transforms = read_transforms_for_input_layer(cfg.mdl_common.input_module, cfg, vocab)
    label_transform = get_label_transforms(cfg.mdl_common.class_indices)
    net = instantiate_model_by_str_name(cfg.model.name, cfg, vocab_size)

    # only log model summary if model training is started for the first time and not from pretrained model
    if not cfg.common.resume_model:
        logger.debug(net)
        net.apply(weights_init)

    # in case of split gpu, let the model handle the gpu copying itself
    if not cfg.device_settings.split_gpus:
        net = DeviceHandler.model_to_device(net)
    else:
        net.move_to_gpu()

    criterion = None
    if not cfg.multi_level_cls.use:
        criterion = cross_entropy_loss()
    # use multi-level loss
    else:
        path = cfg.paths.class_weight_path if OmegaConf.is_missing(cfg, "paths.class_weight_path") else None
        class_weights = load_class_weights(path, cfg) if path else None
        print(class_weights)
        criterion = multi_level_cross_entropy_loss(class_weights)

    if cfg.training.amp:
        logger.info("Mixed precision training is enabled.")

    opt_mgr = OptimizerManager(net, cfg, lr=0.001)
    
    train_path = cfg.dataset.train_set_path
    val_path = cfg.dataset.validation_set_path
    batch_size = cfg.dataloader.batch_size
    num_workers = cfg.dataloader.num_workers
    
    train_set = ProcessingMetagenomicReadDataset(train_path, read_transforms, label_transforms=label_transform)
    val_set = ProcessingMetagenomicReadDataset(val_path, read_transforms, label_transforms=label_transform)
    train_iter = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, collate_fn=train_collate_fn_padded)
    val_iter = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, collate_fn=train_collate_fn_padded)
    
    metric_fns = None
    if not cfg.multi_level_cls.use:
        metric_fns = [precision, recall]
    else:
        thresh = cfg.mdl_common.classification_threshold
        metric_fns = []
        ranks = [rank.lower() for rank in cfg.multi_level_cls.ranks.split(",")]
        for idx, rank in enumerate(ranks):
            metric_fns.append(multi_lv_precision_factory(idx, rank, thresh))
            metric_fns.append(multi_lv_recall_factory(idx, rank, thresh))
    
    trainer = ClassificationTrainer(net, train_iter, val_iter, criterion, opt_mgr, metric_fns, cfg)
    
    logger.info("Starting training.")
    
    try:
        trainer.train()
    except Exception as e:
        print("An exception occured. Check the log file for details.")
        logger.error("Exception occured", exc_info=True)


if __name__ == "__main__":
    main()
