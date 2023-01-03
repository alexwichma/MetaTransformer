from __future__ import absolute_import, division, print_function

import torch
from torch import nn


class MultiHeadAttentionLayer(nn.Module):
    """
    Multi-head self-attention as described in the paper by Liang et al.
    """

    def __init__(self, in_dim, att_dim, num_att_heads):
        super(MultiHeadAttentionLayer, self).__init__()
        # Trainable matrix-mult without bias
        self.W_s1 = nn.Linear(in_dim, att_dim, bias=False)
        self.W_s2 = nn.Linear(att_dim, num_att_heads, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)  # Softmax will be used column-wise (along axis seq axis)

    def forward(self, x):
        # Dim(x) = (batch_size, seq_length, in_dim)
        x_ws1 = self.tanh(self.W_s1(x))
        a = self.W_s2(x_ws1)  # Dim(a) = (batch_size, seq_length, num_att_heads)
        a = self.softmax(a)  # Apply softmax columnwise, e.g. for each attention head
        x_T = torch.transpose(x, 1, 2)  # Swap to (batch_size, in_dim, seq_length) for weighted sum
        m = torch.bmm(x_T, a)  # Dim(m) = (batch_size, in_dim, num_att_heads)
        m = torch.transpose(m, 1, 2)  # Dim(m) = (batch_size, num_att_heads, in_dim) => needed for stacking the heads
        flattened_m = torch.flatten(m, start_dim=1)  # Flatten the vector to prepare for dense layer
        return flattened_m
