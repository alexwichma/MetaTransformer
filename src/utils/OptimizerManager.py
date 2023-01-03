from __future__ import absolute_import, division, print_function

import logging
import torch
from hydra.utils import instantiate

class OptimizerManager:
    """ Holds state of multiple optimizers

    Helper class used to maintain sparse and dense optimizers at the same time. Provides convenience functions for step, zero_grad, loading and saving
    of the optimizers.
    """

    def __init__(self, net, cfg, lr=0.001) -> None:
        self.logger = logging.getLogger(OptimizerManager.__name__)
        self.logger.info("Initializing optimizers")
        self.use_sparse = cfg.mdl_common.sparse_embedding
        self.use_wu_lr = cfg.training.warmup_lr_scheduler_enabled
        self.use_tr_lr = cfg.training.lr_scheduler_enabled
        self.amp = cfg.training.amp

        #if self.use_sparse and self.amp:
        #    raise Exception("Sparse embedding can not be used with autmatic mixed precision (at least for the moment)")

        self.optimizers_container = []
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        embs_names = ["embedding", "shared_embedding", "hash_weights"]

        if self.use_sparse:
            self.logger.info("Sparse embedding is used. Multiple optimizers (dense and sparse adam) will be created")
            dense_params = [param for param_name, param in net.named_parameters() if not any(emb_name in param_name for emb_name in embs_names)]
            dense_optimizer = torch.optim.Adam(dense_params, lr)
            dense_container = {}
            dense_container["opt"] = dense_optimizer
            sparse_params = [param for param_name, param in net.named_parameters() if any(emb_name in param_name for emb_name in embs_names)]
            sparse_optimizer = torch.optim.SparseAdam(sparse_params, lr)
            sparse_container = {}
            sparse_container["opt"] = sparse_optimizer
            self.optimizers_container.append(dense_container)
            self.optimizers_container.append(sparse_container)
        else:
            optim = torch.optim.Adam(net.parameters(), lr)
            container = {}
            container["opt"] = optim
            self.optimizers_container.append(container)

        if self.use_wu_lr:
            warmup_period = cfg.training.warmup_lr_scheduler.warmup_period
            self.logger.info(f"Using warmup scheduler to warm up over {warmup_period} steps")
            for optim_container in self.optimizers_container:
                warmup_scheduler = instantiate(cfg.training.warmup_lr_scheduler, optim)
                optim_container["wu_sched"] = warmup_scheduler
        
        if self.use_tr_lr:
            step_size = cfg.training.lr_scheduler.step_size
            gamma = cfg.training.lr_scheduler.gamma
            self.logger.info(f"Using learning rate scheduler with step size {step_size} and gamma {gamma}")
            for optim_container in self.optimizers_container:
                lr_scheduler = instantiate(cfg.training.lr_scheduler, optim)
                optim_container["lr_sched"] = lr_scheduler

    
    def state_dict(self):
        state = {
            "grad_scaler": self.grad_scaler.state_dict(),
            "optimizers": []
        }

        # retrieve all the state dicts of the optimizers
        for optim_container in self.optimizers_container:
            optimizer_state = {}
            optimizer_state["opt"] = optim_container["opt"].state_dict()
            for sched_key in ["wu_sched", "lr_sched"]:
                if sched_key in optim_container:
                    optimizer_state[sched_key] = optim_container[sched_key].state_dict()
            state["optimizers"].append(optimizer_state)

        return state

    def load_state_dict(self, state):
        self.logger.info("Loading optimizer state dict")
        # fallback for previous optimizer state dict format
        # if no grad scaler is present, just initialize a state dict from the current one
        if isinstance(state, dict) and not "optimizers" in state:
            self.logger.info("Loading info: Old state dict found, converting to new format")
            state = {"grad_scaler": self.grad_scaler.state_dict(), "optimizers": [{"opt": state}]}

        assert len(state["optimizers"]) == len(self.optimizers_container)

        self.grad_scaler.load_state_dict(state["grad_scaler"])
        
        for optim_container, optim_state in zip(self.optimizers_container, state["optimizers"]):
            optim_container["opt"].load_state_dict(optim_state["opt"])
            for sched_key in ["wu_sched", "lr_sched"]:
                if sched_key in optim_state:
                    optim_container[sched_key].load_state_dict(optim_state[sched_key])

    def step(self, scalar_input):
        self.grad_scaler.scale(scalar_input).backward()
        
        for optim_container in self.optimizers_container:
            self.grad_scaler.step(optim_container["opt"])
            
            if self.use_tr_lr:
                optim_container["lr_sched"].step()
            
            if self.use_wu_lr:
                optim_container["wu_sched"].dampen()
        
        self.grad_scaler.update()

    
    def zero_grad(self):
        for optim_container in self.optimizers_container:
            optim_container["opt"].zero_grad()
