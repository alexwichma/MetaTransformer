"""
Manages experiment directory creation, config loading and logging initialization during training
"""


from __future__ import absolute_import, division, print_function

import os
import pathlib
from typing import Callable
from omegaconf import OmegaConf
import datetime

import log.Logger as Logger
import utils.device_handler as DeviceHandler

ORIGINAL_DIR = os.getcwd()
EXPERIMENT_DIR = None


def set_experiment_dir(path):
    global EXPERIMENT_DIR
    if EXPERIMENT_DIR != None:
        raise Exception("New current working directory should only be set once per script")
    EXPERIMENT_DIR = path


def check_experiment_dir():
    if EXPERIMENT_DIR == None:
        raise Exception("Experiment dir should be set before starting training")


def get_experiment_dir():
    return EXPERIMENT_DIR


def get_original_dir():
    return ORIGINAL_DIR


def create_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def create_experiment_dirs():
    check_experiment_dir()
    create_dir(get_checkpoint_dir())
    create_dir(get_tensorboard_dir())


def get_config_path():
    return os.path.join(get_experiment_dir(), "config.yaml")


def get_checkpoint_dir():
    return os.path.join(get_experiment_dir(), "checkpoints")


def get_tensorboard_dir():
    return os.path.join(get_experiment_dir(), "tensorboard")


def _init_device_handler(cfg):
    use_cpu = cfg.device_settings.use_cpu
    gpu_count = cfg.device_settings.gpu_count
    gpu_ids = cfg.device_settings.gpu_ids
    split_gpus = cfg.device_settings.split_gpus
    DeviceHandler.init_device_handler(use_cpu, gpu_count, gpu_ids, split_gpus)


def _generate_job_id() -> str:
    if "SLURM_JOB_ID" in os.environ:
        return os.environ["SLURM_JOB_ID"]
    else:
        return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")


# decorator for functions to load configurations for training
def training_configs(main_fn: Callable) -> None:
 
    def wrap_fn():
        timestamp = "_" + _generate_job_id()
        cmd_line_args = OmegaConf.from_cli()
        # resume dir specified, set experiment dir accordingly and load already merged config file
        if cmd_line_args.resume_dir is not None:
            set_experiment_dir(cmd_line_args.resume_dir)
            base_cfg = OmegaConf.load(get_config_path())
            base_cfg.common.resume_model = True
        else:
            if cmd_line_args.experiment_name is None:
                raise Exception("Experiment name needs to be provided")
            if cmd_line_args.experiment_base_dir is None:
                raise Exception("Experiment dir needs to be provided")
            if cmd_line_args.cfg_path is None:
                raise Exception("Base config path needs to be provided")
            if cmd_line_args.data_path_root is None:
                raise Exception("Data path root needs to be provided.")
            
            # set and create all the directories
            set_experiment_dir(os.path.join(cmd_line_args.experiment_base_dir, cmd_line_args.experiment_name + timestamp))
            create_dir(get_experiment_dir())
            create_experiment_dirs()
            
            # load all configurations and merge them into single config
            base_dir = os.path.dirname(cmd_line_args.cfg_path)
            base_cfg = OmegaConf.load(cmd_line_args.cfg_path)
            base_cfg.common.experiment_name = cmd_line_args.experiment_name
            base_cfg.paths.data_path_root = cmd_line_args.data_path_root
            model_cfg_path = os.path.join(base_dir, "model", base_cfg.cfgs.model + ".yaml")
            model_cfg = OmegaConf.load(model_cfg_path)
            base_cfg.model = model_cfg
            dataset_cfg_path = os.path.join(base_dir, "dataset", base_cfg.cfgs.dataset + ".yaml")
            dataset_cfg = OmegaConf.load(dataset_cfg_path)
            base_cfg.dataset = dataset_cfg
            opt_cfg_path = os.path.join(base_dir, "optimizer", base_cfg.cfgs.optimizer + ".yaml")
            opt_cfg = OmegaConf.load(opt_cfg_path)
            base_cfg.optimizer = opt_cfg
            # write merged config to experiment directory
            OmegaConf.save(base_cfg, f=get_config_path())
            # invoke main function

        # init file logger
        Logger.init_logging(os.path.join(get_experiment_dir(), "train.log"))
        _init_device_handler(base_cfg)
        
        main_fn(cmd_line_args, base_cfg)
    
    return wrap_fn


# short test of the decorator
@training_configs
def test_fn(cmd_line_args, cfg):
    print(cmd_line_args)
    print(cfg)
    print(cfg.paths.vocabulary_path)


if __name__ == "__main__":
    test_fn()