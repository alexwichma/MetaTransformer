from __future__ import absolute_import, division, print_function

from layers import BidirectionalLSTMLayer, MultiHeadAttentionLayer
from models.BaseModel import BaseModel
from utils.torch_utils import extract_base_seq_len


class EmbedLstmAttention(BaseModel):
    """
    Re-implementation of the EmbedLstmAttention architecture 
    """

    def __init__(self, input_module, output_module, input_dim, lstm_dim, att_dim, num_att_heads):
        super(EmbedLstmAttention, self).__init__()
        self.input_module = input_module
        self.output_module = output_module
        self.bi_lstm = BidirectionalLSTMLayer(input_dim, lstm_dim)
        self.multi_head_att = MultiHeadAttentionLayer(2 * lstm_dim, att_dim, num_att_heads)

    def move_to_gpu(self):
        class_name = self.__class__.__name__
        raise Exception(f"Model sharding not supported for model of type {class_name}")

    def forward(self, x, use_padding=False):
        # apparently applying "pack_padded_sequences" is slower than just leaving it as is
        base_seq_lengths = extract_base_seq_len(x) if use_padding else None
        x = self.input_module(x)
        x = self.bi_lstm(x, base_seq_lengths)
        x = self.multi_head_att(x)
        x = self.output_module(x)
        return x
