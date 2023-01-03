from __future__ import absolute_import, division, print_function

from torch import nn


class BidirectionalLSTMLayer(nn.Module):
    """Wrapper around a bi-directional LSTM module"""

    def __init__(self, input_dim, lstm_dim):
        super(BidirectionalLSTMLayer, self).__init__()
        self.num_stacks = 2
        self.num_dirs = 2
        self.bi_lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_dim,
                               num_layers=self.num_stacks, bidirectional=True, batch_first=True)

    def forward(self, x, unpadded_seq_lens):
        # Dim(outputs) = (batch_size, seq_length, num_hidden_vecs * embed_dim)
        # Forward and reverse hidden states are already concatenated
        # Feed packed batch to bi_lstm and unpack afterward
        self.bi_lstm.flatten_parameters()
        if unpadded_seq_lens is not None:
            packed_batch = nn.utils.rnn.pack_padded_sequence(x, lengths=unpadded_seq_lens, enforce_sorted=False,
                                                             batch_first=True)
            outputs, _ = self.bi_lstm(packed_batch)
            padded_out, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            return padded_out
        else:
            outputs, _ = self.bi_lstm(x)
            return outputs
