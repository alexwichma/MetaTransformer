from __future__ import absolute_import, division, print_function

from typing import Callable, Dict, Union, List
from omegaconf.dictconfig import DictConfig
from trainer.AbstractTrainer import AbstractTrainer
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.OptimizerManager import OptimizerManager
import utils.device_handler as DeviceHandler


class ClassificationTrainer(AbstractTrainer):
    """
    Implementation of an AbstractTrainer for the task of classification
    """

    def __init__(self, 
                 model: nn.Module, 
                 train_iter: DataLoader, 
                 val_iter: DataLoader, 
                 criterion: nn.Module, 
                 opt_mgr: OptimizerManager, 
                 metric_fns: List[Callable], 
                 config: DictConfig):
        super().__init__(model, 
                         train_iter, 
                         val_iter, 
                         criterion, 
                         opt_mgr, 
                         metric_fns,
                         config)
        self.softmax = nn.Softmax(dim=1)
    
    def _apply_softmax(self, model_out):
        if isinstance(model_out, torch.Tensor):
            return self.softmax(model_out)
        elif isinstance(model_out, list):
            return [self.softmax(t) for t in model_out]
        else:
            raise Exception("Invalid model output received")


    def _train_step(self, data, target) -> Union[Dict, int]:
            
        self.opt_mgr.zero_grad()
        print("amp:"+str(self.amp))
        with torch.cuda.amp.autocast(enabled=self.amp):
                model_out = self.model(data)
                loss = self.criterion(model_out, target)

        self.opt_mgr.step(loss)

        return self._apply_softmax(model_out), loss
    

    def _validate(self, val_steps=2500) -> Dict:
        
        self.model.eval()
        self.validation_metrics.reset_metrics()

        with torch.no_grad():

            current_step = 0
            
            for data, target in self.val_iter:
                
                data, target = DeviceHandler.tensor_to_device(data), DeviceHandler.tensor_to_device(target, self.target_gpu_id)

                with torch.cuda.amp.autocast(enabled=self.amp):
                    model_out = self.model(data)
                    loss = self.criterion(model_out, target)
                
                softmaxed_model_out = self._apply_softmax(model_out)
                
                self.validation_metrics.add_metric('loss', loss.item())
                for metric_fn in self.metric_fns:
                    self.validation_metrics.add_metric(metric_fn.__name__, metric_fn(softmaxed_model_out, target))
                
                # do early stopping in case more than val_steps batches are available
                if (current_step + 1) == val_steps:
                    return self.validation_metrics.get_avgs()
                
                current_step += 1

        return self.validation_metrics.get_avgs()
