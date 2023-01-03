"""
Manages devices (cpu / cuda) and provides utility functions around them for training/testing scripts
"""


from __future__ import absolute_import, division, print_function

import logging
import torch

from utils.torch_utils import optimizer_to


logger = logging.getLogger(__name__)

USE_CPU = False
DEVICE_IDS = None
SPLIT_GPUS = False


def init_device_handler(use_cpu, gpu_count, gpu_ids, split_gpus):
    global USE_CPU, DEVICE_IDS, SPLIT_GPUS
    USE_CPU = use_cpu
    SPLIT_GPUS = split_gpus

    if gpu_ids is not None:
        gpu_ids = _parse_ids(gpu_ids)

    if SPLIT_GPUS and gpu_ids is not None and len(gpu_ids) != 2:
        raise Exception("Split gpus only supported with exactly two gpus for now")
    if SPLIT_GPUS and gpu_ids is None and gpu_count != 2:
        raise Exception("Split gpus only supported with exactly two gpus for now")
        
    num_gpus_avail = torch.cuda.device_count()
    DEVICE_IDS = _determine_device_ids(gpu_count, gpu_ids, num_gpus_avail)


def _parse_ids(input_ids):
    if isinstance(input_ids, int):
        return [input_ids]
    elif isinstance(input_ids, str):
        return [int(x) for x in input_ids.split(",")]
    else:
        raise Exception("Str or int need to be provided for gpu_ids in config")


def _determine_device_ids(gpu_count, gpu_ids, num_gpus_avail):
        if USE_CPU:
            return []
        elif gpu_ids is not None:
            assert max(gpu_ids) <= num_gpus_avail - 1
            return gpu_ids
        elif gpu_count > 1:
            assert gpu_count <= num_gpus_avail
            return list(range(gpu_count))
        else:
            return [torch.cuda.current_device()]


def _determine_torch_device(id=None):
    if USE_CPU or not torch.cuda.is_available():
        return torch.device("cpu")
    if id is None:
        if len(DEVICE_IDS) > 1:
            return torch.device("cuda")
        else:
            return torch.device("cuda:{}".format(DEVICE_IDS[0]))
    else:
        assert id in DEVICE_IDS
        return torch.device("cuda:{}".format(id))


def gpu_available():
    device = _determine_torch_device()
    return not USE_CPU and device.type == "cuda"


def model_to_device(model, id=None):
    device = _determine_torch_device(id)
    if len(DEVICE_IDS) > 1 and not SPLIT_GPUS:
        logger.info(f"{len(DEVICE_IDS)} gpus available. Using parallel batch processing")
        model = torch.nn.DataParallel(model, device_ids=DEVICE_IDS)
    return model.to(device)


def optimizer_to_device(optimizer):
    device = _determine_torch_device()
    optimizer_to(optimizer, device)


def tensor_to_device(data, id=None):
    device = _determine_torch_device(id)
    if isinstance(data, list):
        return [t.to(device) for t in data]
    return data.to(device)


def get_first_split_gpu_id():
    return DEVICE_IDS[0]


def get_last_split_gpu_id():
    return DEVICE_IDS[-1]


def get_device(id=None):
    return _determine_torch_device(id)
