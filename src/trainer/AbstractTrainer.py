from __future__ import absolute_import, division, print_function

from abc import abstractmethod
from logging import getLogger
from sys import float_info
import os
from typing import Callable, Dict, List, Union
from omegaconf.dictconfig import DictConfig
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import pandas as pd
import operator
from torch import nn
from torch.utils.data import DataLoader

from log.Logger import dict_to_str, log_progress
from utils.OptimizerManager import OptimizerManager
from utils.context_manager import get_checkpoint_dir, get_tensorboard_dir
from utils.metric_utils import MetricTracker
from utils.timer import Timer
from utils.torch_utils import resume_training
from utils.utils import camel_case_to_snake_case, create_if_not_exist
import utils.device_handler as DeviceHandler


MODEL_METADATA_FILE_NAME = 'model_meta.csv'


class AbstractTrainer():
    """ Manages the training process
    
    Abstract helper class that manages the training process. It performs training, regular validation, model checkpointing and tensorboard logging.
    This class needs to be inherited by a class that implements the _train_step and _validate methods
    """

    def __init__(self, 
                 model: nn.Module, 
                 train_iter: DataLoader, 
                 val_iter: DataLoader, 
                 criterion: nn.Module, 
                 opt_mgr: OptimizerManager, 
                 metric_fns: List[Callable], 
                 cfg: DictConfig):
        self.model = model
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.criterion = criterion
        self.opt_mgr = opt_mgr
        self.metric_fns = metric_fns
        self.target_gpu_id = DeviceHandler.get_last_split_gpu_id() if cfg.device_settings.split_gpus else None
        
        # Set shortcuts to commmonly used config fiels
        self.cfg = cfg
        self.threshold = cfg.mdl_common.classification_threshold
        self.ckpt_dir = get_checkpoint_dir()
        self.tb_path = get_tensorboard_dir()
        self.batch_size = cfg.dataloader.batch_size
        self.save_interval = cfg.training.save_interval
        self.amp = cfg.training.amp
        
        self.logger = getLogger('trainer')

        self.training_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_fns], postfix='/train')
        self.validation_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_fns], postfix='/val')
        self.current_batch = 0
        self.timer = Timer()
        self.last_time = None
        self.num_unimproved_evaluations = 0
        self.current_model_best = False
        self.logging_interval = cfg.training.logging_interval
        self.evaluation_interval = cfg.training.evaluation_interval
        self.max_step = cfg.training.max_steps

        self.monitor_metric, self.monitor_mode, self.monitor_best = self.init_monitor_metric()

        # initialize model_meta list, holding information of the currently saved checkpoints
        self.meta_path = os.path.join(self.ckpt_dir, MODEL_METADATA_FILE_NAME)
        self.models_meta = self.init_models_meta()

        if cfg.common.resume_model:
            self._resume_training()
        
        self.save_n = cfg.training.save_top_n_models
        self.early_stop_thresh = cfg.training.early_stop_threshold

        self.tb_logger: SummaryWriter = self.init_tb_writer()
        
    def init_models_meta(self):
        if os.path.exists(self.meta_path):
            csv_df = pd.read_csv(self.meta_path)
            # parse tuples as lists
            vals = [[*t] for t in list(csv_df.itertuples(index=False, name=None))]
            compare_fn = min if self.monitor_mode == "min" else max
            self.monitor_best = compare_fn(vals, key=operator.itemgetter(1))
            return vals
        else:
            return []

    def write_models_meta(self):
        rows = self.models_meta
        df = pd.DataFrame(rows, columns=["file_name", "value_{}_{}".format(self.monitor_mode, self.monitor_metric), "is_current"])
        df.to_csv(self.meta_path, index=False)

    def init_tb_writer(self):
        create_if_not_exist(self.tb_path)
        return SummaryWriter(self.tb_path)

    def init_monitor_metric(self):
        if self.cfg.training.monitor_metric == "off":
            monitor_mode = 'off'
            monitor_metric = None
            monitor_best = 0
            return monitor_metric, monitor_mode, monitor_best
        else:
            monitor_mode, monitor_metric = self.cfg.training.monitor_metric.split()
            assert monitor_mode in ['max', 'min']
            monitor_best = float_info.max if monitor_mode == "min" else float_info.min
            return monitor_metric, monitor_mode, monitor_best

    @abstractmethod
    def _train_step(self, train_dl) -> Union[Dict, int]:
        raise NotImplementedError

    @abstractmethod
    def _validate(self, val_dl) -> Dict:
        raise NotImplementedError

    def train(self) -> None:
        self.logger.info("Starting training at %d batches.", self.current_batch + 1)
        self.timer.start()
        
        for data, target in self.train_iter:
            data, target = DeviceHandler.tensor_to_device(data), DeviceHandler.tensor_to_device(target, id=self.target_gpu_id)
            model_out, loss = self._train_step(data, target)

            self.training_metrics.add_metric('loss', loss.item())
            for metric_fn in self.metric_fns:
                self.training_metrics.add_metric(metric_fn.__name__, metric_fn(model_out, target))
            
            self.current_batch += 1

            if (self.current_batch + 1) % self.logging_interval == 0:
                avgs = self.training_metrics.get_avgs()
                log_progress(self.logger, self.current_batch + 1, avgs)

            if (self.current_batch + 1) % self.evaluation_interval == 0:
                self.timer.update_elapsed_time()
                self._perform_evaluation()

            if (self.current_batch + 1) % self.max_step == 0:
                self.logger.info(f"Maximum number of steps ({self.max_step}) taken. End training.")
                exit(0)

    def _perform_evaluation(self):
        logs = self.training_metrics.get_avgs()
        val_logs = self._validate()
        self.model.train()
        logs.update(val_logs)
        self.logger.info("Elapsed time (in hours): %d", self.timer.get_elapsed_time_in_hours())
        self.logger.info("Metrics before evaluation: %s", dict_to_str(logs))
        self.logger.info("Number of batches processed: %d", self.current_batch + 1)
        self._emit_logs(logs)

        is_best_model = False
        if self.monitor_mode != "off":
            is_best_model = self._evaluate_monitor_metric(logs)
            self.num_unimproved_evaluations = 0 if is_best_model else self.num_unimproved_evaluations + 1
            if self.num_unimproved_evaluations > self.early_stop_thresh:
                msg = "Model did not improve on metric {} for {} evaluations. Early stopping training!" \
                        .format(self.monitor_metric, self.num_unimproved_evaluations)
                self.logger.info(msg)
                exit(0)
        
        # check if save_interval many evaluation intervals passed, save model if so
        if ((self.current_batch + 1) / self.evaluation_interval) % self.save_interval == 0:
                self.timer.update_elapsed_time()
                self._save_checkpoint(logs, is_best_model)
        
        self.training_metrics.reset_metrics()

    def _emit_logs(self, logs) -> None:
        for k, v in logs.items():
            if self.tb_logger:
                self.tb_logger.add_scalar(k, v, self.current_batch + 1)

    def _evaluate_monitor_metric(self, log) -> bool:
        if self.monitor_metric not in log:
            self.monitor_mode = 'off'
            return False
        min_case = self.monitor_mode == 'min' and log[self.monitor_metric] <= self.monitor_best
        max_case = self.monitor_mode == 'max' and log[self.monitor_metric] >= self.monitor_best
        better = min_case or max_case
        self.logger.debug("Is best model: {}, Metric: {}, Metric Value: {}, Monitor Best: {}".format(better, self.monitor_metric, log[self.monitor_metric], self.monitor_best))
        if better:
            self.monitor_best = log[self.monitor_metric]  
        return better

    def _truncate_to_n_models(self):
        if len(self.models_meta) <= self.save_n:
            return
        self.logger.info("More than {} models are currently saved. Removing worst performing model".format(self.save_n))
        sort_descending = self.monitor_mode == 'max'
        self.models_meta.sort(key=operator.itemgetter(1), reverse=sort_descending)
        # Get the worst performing model. Max case: Lowest value, Min case: Highest value => always last position in list
        file_to_truncate, _, _ = self.models_meta[-1]
        file_path = os.path.join(self.ckpt_dir, file_to_truncate)
        os.remove(file_path)
        self.models_meta.pop()

    def _save_checkpoint(self, logs, override_best=False) -> None:
        self.logger.info("Saving model at batch %d", self.current_batch + 1)
        m_name = camel_case_to_snake_case(type(self.model).__name__)
        state = {
            'model_name': m_name,
            'batch_num': self.current_batch,
            'training_time': self.timer.get_elapsed_time(),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt_mgr.state_dict(),
            'monitor_best': self.monitor_best,
            'config': self.cfg
        }

        file_name = "{}_ckpt_bt_{}.pt".format(m_name, self.current_batch + 1)
        filepath = os.path.join(self.ckpt_dir, file_name)
        torch.save(state, filepath)
        self.models_meta.append([file_name, logs[self.monitor_metric], True])
        # remove "current" flag from the other saved model
        for index in range(len(self.models_meta) - 1):
            self.models_meta[index][2] = False

        if override_best:
            self.logger.info("Best model will  be overridden.")
            best_file_path = os.path.join(self.ckpt_dir, "{}_ckpt_best.pt".format(m_name))
            torch.save(state, best_file_path)

        self._truncate_to_n_models()
        self.write_models_meta()

    def _get_most_current_model_path(self):
        for model_meta in self.models_meta:
            if model_meta[2]:
                return os.path.join(self.ckpt_dir, model_meta[0])
        raise Exception("There is nothing to resume from. Check model file in resume directory: {}".format(self.ckpt_dir))
        

    def _resume_training(self) -> None:
        resume_path = self._get_most_current_model_path()
        model, optimizer, meta_infos = resume_training(resume_path, self.model, self.opt_mgr, self.logger)
        self.model = model
        self.optimizer = optimizer
        if not self.cfg.device_settings.split_gpus:
            DeviceHandler.model_to_device(self.model)
        else:
            self.model.move_to_gpu()
        self.current_batch = meta_infos["current_batch"]
        self.timer.set_elapsed_time(meta_infos["training_time"])
        self.monitor_best = meta_infos["monitor_best"]
        self.logger.info("Succesfully resumed model. Continue training.")

