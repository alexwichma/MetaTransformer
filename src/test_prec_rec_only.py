"""
Testing script that only tracks per-class precision and recall.
"""


from __future__ import absolute_import, division, print_function

import torch
from tqdm import tqdm
import os
import pandas as pd
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dataset.MetagenomicReadDataset import ProcessingMetagenomicReadDataset
from models.model_utils import get_label_transforms, instantiate_model_by_str_name, read_transforms_for_input_layer
from logging import getLogger
import utils.device_handler as DeviceHandler
from utils.metric_utils import MetricTracker, average_n_predictions, precision, recall
from utils.argument_parser import parse_args_test
from utils.torch_utils import train_collate_fn_padded
from utils.utils import fix_random_seeds, load_vocabulary, parse_class_mapping


logger = getLogger('test/main')


def test(net, test_loader, num_classes, avg_n, level_index):

    device = DeviceHandler.get_device()

    metric_fns = [precision, recall]
    # create a metric tracker for each class, since we may need the results later seperately
    metric_trackers = {class_idx: MetricTracker(*[m.__name__ for m in metric_fns], postfix='/test') for class_idx in range(num_classes)}

    with torch.no_grad():

        for data in tqdm(test_loader):
            reads, labels = data

            if device.type == "cuda":
                reads = reads.to(device)
            
            with torch.cuda.amp.autocast(enabled=True):
                out = net(reads)
                # used for multi-level classification
                if level_index != -1:
                    out = out[level_index]

            out = out.float()
            out = out.cpu()
            
            # save predictions to prediction tensor for later analaysis
            out_averaged = average_n_predictions(out, num_classes, avg_n)
            # apply softmax after averaging
            out_averaged = torch.nn.functional.softmax(out_averaged, dim=1)
            labels_reduced = labels[torch.arange(0, labels.shape[0], avg_n)]

            # get the unique class_idxs, and iterate over them so that we can add the metrics to the approriate metric tracker
            # this should not be too bad since in almost all cases only same labels will be in the labels list (since all reads are sorted) 
            # use unique_consequtive since all labels (if there different ones) will appear consecutively
            unique_labels = torch.unique_consecutive(labels_reduced)
            unique_labels_len = unique_labels.shape[0]
            for metric_fn in metric_fns:
                for ul_idx in range(unique_labels_len):
                    unique_label = unique_labels[ul_idx].item()
                    idxs = labels_reduced == unique_label
                    metric_trackers[unique_label].add_metric(metric_fn.__name__, metric_fn(out_averaged[idxs], labels_reduced[idxs]))

    
    return metric_trackers


def main():
    fix_random_seeds()
    cmd_args = parse_args_test()

    DeviceHandler.init_device_handler(False, 1, 0, False)

    device = DeviceHandler.get_device()
   
    hydra_conf = OmegaConf.load(cmd_args.config_state_path)

    level_index = cmd_args.level_index
    
    vocab, vocab_size = None, 0
    if not hydra_conf.mdl_common.input_module in ["lsh", "hash", "one_hot", "one_hot_embed", "bpe"]:
        vocab, vocab_size = load_vocabulary(cmd_args.vocab_path)
    else:
        logger.info("Vocabulary is not loaded since {} is used".format(hydra_conf.mdl_common.input_module))
    
    # either use normal num classes or num_classes from multi_level if multi_level model is used
    num_classes = hydra_conf.mdl_common.num_classes if level_index == -1 else int(hydra_conf.multi_level_cls.num_classes.split(",")[level_index])

    class_mapping = parse_class_mapping(cmd_args.class_mapping_path)

    # load model
    read_transforms = read_transforms_for_input_layer(hydra_conf.mdl_common.input_module, hydra_conf, vocab, train=False)
    label_transform = get_label_transforms(hydra_conf.mdl_common.class_indices)
    net = instantiate_model_by_str_name(hydra_conf.model.name, hydra_conf, vocab_size)
   
    state = torch.load(cmd_args.model_state_path, map_location="cpu")
    net.load_state_dict(state["model_state_dict"])
    net.eval()
    net.to(device)

    dataset = ProcessingMetagenomicReadDataset(cmd_args.reads_path, read_transforms, label_transforms=label_transform)
    dataloader_forward = DataLoader(dataset, batch_size=cmd_args.batch_size, num_workers=64, shuffle=False, collate_fn=train_collate_fn_padded, drop_last=True)

    avg_n = cmd_args.prediction_mode
    
    metric_trackers = test(net, dataloader_forward, num_classes, avg_n, level_index)

    # add results into dataframe, calculate average
    metric_cols = [k for k in metric_trackers[0].get_avgs()]
    cols = ['Organism', *metric_cols]
    result = []
    for class_idx, metric_tracker in metric_trackers.items():
        result_row =  [class_mapping[class_idx]]
        for _, val in metric_tracker.get_avgs().items():
            result_row.append(val)
        result.append(result_row)
    df = pd.DataFrame(result, columns=cols)
    df = df.round(decimals=4)
    out_path = cmd_args.output_path

    df.to_csv(os.path.join(out_path, 'class_metrics.csv'))


if __name__ == "__main__":
    main()
