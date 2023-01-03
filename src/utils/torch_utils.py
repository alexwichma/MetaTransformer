"""
Utility functions regarding PyTorch internals and tensor manipulation
"""


from __future__ import absolute_import, division, print_function

import torch
from torch import nn
from torch.functional import Tensor
from torch.nn.utils.rnn import pad_sequence

from utils.utils import SPECIAL_TOKENS_2_INDEX


# Borrowed from https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/3
# This is used to load the optimizer on the cpu from checkpoint to avoid going OOM on the GPU
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


# extract the original sequence lengths of sequence padded with 0s, needed for pack_padded_seqs in rnns
def extract_base_seq_len(x):
    #  Find the base sequence length of a zero padded sequence
    max_seq_length = x.shape[1]
    zero_mask = x == 0  # Mask zero elements
    _, max_idxs = torch.min(zero_mask, dim=1)  # along each row find first masked value (first 0 per seq)
    max_idxs[max_idxs == 0] = max_seq_length  # when max index is found at position 0, there was no padding element present => set it to max seq length
    return max_idxs.cpu()


# generic weight init for all model types base on uniform xavier initialization
def weights_init(m):
    if type(m) == nn.Embedding:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.Linear:
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
    if type(m) == nn.LSTM:
        for name, parameter in m.named_parameters():
            if "bias" in name:
                nn.init.zeros_(parameter)
            if "weight" in name:
                nn.init.xavier_uniform_(parameter)


def train_collate_fn_padded(x):
    """
    Collate function to pad sequences to same length. Additionally, multiple labels per instance
    are separated in to different tensors.
    """
    data = [instance[0] for instance in x]
    targets = [instance[1] for instance in x]
    # pad sequences so they all have the same length
    data_tensor = pad_sequence(data, batch_first=True, padding_value=SPECIAL_TOKENS_2_INDEX["<pad>"])
    targets_tensor = None
    # targets is just a list of integer labels, for normal prediction like standard collate function
    if isinstance(targets[0], int):
        targets_tensor = torch.tensor(targets, dtype=torch.long)
    # targets are list, used for multi-level prediction
    elif isinstance(targets[0], list):
        # seperate into lists per level, then transform each list to tensor
        targets_tensor = [torch.tensor(per_level_labels, dtype=torch.long) for per_level_labels in zip(*targets)]
    else:
        raise Exception("No valid type for targets provided. Targets must be either of type int or list[int].")
    return data_tensor, targets_tensor

def train_collate_single_fn_padded(x):
    """
    Collate function to pad sequences to same length. Additionally, multiple labels per instance
    are separated in to different tensors.
    """
    data = [instance[0] for instance in x]
    targets = [instance[1] for instance in x]
    file_ids = [instance[2] for instance in x]
    # pad sequences so they all have the same length
    data_tensor = pad_sequence(data, batch_first=True, padding_value=SPECIAL_TOKENS_2_INDEX["<pad>"])
    targets_tensor = None
    # targets is just a list of integer labels, for normal prediction like standard collate function
    if isinstance(targets[0], int):
        targets_tensor = torch.tensor(targets, dtype=torch.long)
    # targets are list, used for multi-level prediction
    elif isinstance(targets[0], list):
        # seperate into lists per level, then transform each list to tensor
        targets_tensor = [torch.tensor(per_level_labels, dtype=torch.long) for per_level_labels in zip(*targets)]
    else:
        raise Exception("No valid type for targets provided. Targets must be either of type int or list[int].")
    return data_tensor, targets_tensor, file_ids


def generate_square_subsequent_mask(sz: int) -> Tensor:
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def generate_cls_token_mask(sz: int) -> Tensor:
    mask = torch.zeros(sz, sz, dtype=torch.float32)
    mask[1:, 0] = float('-inf')  # no position should be able to attend the cls token, since this is just for aggregation 
    return mask


def generate_source_padding_mask(source: Tensor) -> Tensor:
    # dim(source) = (batch_size, seq_len)
    mask = torch.zeros_like(source, dtype=torch.int32)
    mask = mask.masked_fill_(source == SPECIAL_TOKENS_2_INDEX["<pad>"], 1)
    return mask.bool()


def resume_training(resume_path, model, opt_mgr, logger):
    logger.info("Resume path at location %s given. Trying to resume model.", resume_path)
    state = torch.load(resume_path, map_location='cpu')
    model.load_state_dict(state["model_state_dict"])
    opt_mgr.load_state_dict(state["optimizer_state_dict"])
    
    meta_infos = {
        "current_batch": state["batch_num"],
        "training_time": state["training_time"],
        "monitor_best": state["monitor_best"]
    }

    return model, opt_mgr, meta_infos


def freeze_model_backbone(net):
    for name, parameter in net.named_parameters():
        parameter.requires_grad = name.startswith("output_module")