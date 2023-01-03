from __future__ import absolute_import, division, print_function

import torch
import math
import logging
from torch.nn.modules.activation import Softmax

from layers import TransformerEncoder, PositionalEncoding2
from models.BaseModel import BaseModel
from utils.torch_utils import generate_cls_token_mask, generate_source_padding_mask
import utils.device_handler as DeviceHandler


class ClassificationTransformer(BaseModel):
    """
    Implementation of the transformer encoder based classification model
    """

    def __init__(self,
                 input_module,
                 output_module, 
                 dim_model, 
                 num_att_heads, 
                 dim_ff, 
                 num_encoder_blocks, 
                 dropout,
                 aggregation_mode,
                 split_gpu,
                 activation="relu", 
                 use_pad_mask=True):
        super(ClassificationTransformer, self).__init__()
        self.split_gpu = split_gpu
        self.input_module = input_module
        if self.split_gpu:
            self.first_gpu_id = DeviceHandler.get_first_split_gpu_id()
            self.last_gpu_id = DeviceHandler.get_last_split_gpu_id()
        self.output_module = output_module
        self.pos_enc = PositionalEncoding2(dim_model, dropout=dropout)
        self.transformer_enc = TransformerEncoder(dim_model, num_att_heads, dim_ff, num_encoder_blocks,
                                                  activation=activation, dropout=dropout)
        self.aggregation_mode = aggregation_mode
        self.use_pad_mask = use_pad_mask
        self.device = DeviceHandler.get_device(self.last_gpu_id) if self.split_gpu else DeviceHandler.get_device()
        self.dim_model = dim_model
        self.softmax = Softmax(dim=1)
        self.logger = logging.getLogger(self.__class__.__name__)

    def move_to_gpu(self):
        self.input_module = DeviceHandler.model_to_device(self.input_module, self.first_gpu_id)
        self.pos_enc = DeviceHandler.model_to_device(self.pos_enc, id=self.last_gpu_id)
        self.transformer_enc = DeviceHandler.model_to_device(self.transformer_enc, id=self.last_gpu_id)
        self.output_module = DeviceHandler.model_to_device(self.output_module, id=self.last_gpu_id)

    def forward(self, x):

        # dim(x) = (batch_size, seq_len)
        unembedded_x = x.clone()

        _, seq_len = x.size()
        
        x = self.input_module(x) * math.sqrt(self.dim_model)  # (batch_size, seq_len, d_model)

        # p2p transfer between gpus
        if self.split_gpu:
            x = DeviceHandler.tensor_to_device(x, id=self.last_gpu_id)

        x = self.pos_enc(x)
        # (seq_len, batch_size, d_model), needs to be transposed for transformer input
        # tranpose corrupts contiguous memory layout, make contiguous again 
        x = x.transpose(0, 1).contiguous()

        cls_mask = None
        if self.aggregation_mode == "cls":
            cls_mask = generate_cls_token_mask(seq_len)
            cls_mask = cls_mask.to(self.device)

        pad_mask = None
        if self.use_pad_mask:
            pad_mask = generate_source_padding_mask(unembedded_x)
            pad_mask = pad_mask.to(self.device)

        x = self.transformer_enc(x, src_mask=cls_mask, src_key_padding_mask=pad_mask)
        # make contiguous after re-transposing again
        x = x.transpose(0, 1).contiguous()   # (batch_size, seq_len, d_model)

        if self.aggregation_mode == "cls":
            x = x[:, 0, :]
        elif self.aggregation_mode == "mean":
            x = torch.mean(x, dim=1)
        else:
            raise Exception(f"Aggregation mode {self.aggregation_mode} is no valid aggregation mode")
        
        return self.output_module(x)