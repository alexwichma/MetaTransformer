"""
Positional encoding for the transformer encoder architecture
"""


from __future__ import absolute_import, division, print_function

from torch import nn
import torch
import math
from matplotlib import pyplot as plt


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512, dropout=0.0) -> None:
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # longest wavelength in the geometric progression
        PROGR_END = 10000.0
        # dim(pos_enc) = (max_len, d_model)
        pos_enc = torch.zeros(max_len, d_model)
        # dim(pos) = (max_len, 1)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # inverse is produced my multiplying with -1.0, base PROGR_END (default 10000.0) is produced by taking nat. log
        # dim(div_factor) = (d_model / 2)
        div_factor = torch.exp(-1.0 * torch.arange(0, d_model, 2, dtype=torch.float32) / d_model * math.log(PROGR_END)) 
        # apply sine at even dimensions, apply cosine att odd dimensions per positions
        pos_enc[:, 0::2] = torch.sin(pos * div_factor)
        pos_enc[:, 1::2] = torch.cos(pos * div_factor)
        # register as buffer to make it part of module in terms of state-dict, copy to device etc...
        self.register_buffer('pos_enc', pos_enc)
        self.dropout = nn.Dropout(p=dropout)

    def generate_heatmap(self, save_path=None) -> None:
        plt.style.use('seaborn')
        plt.pcolormesh(self.pos_enc.squeeze().numpy(), cmap="PuOr")
        plt.colorbar()
        plt.xlim(0, self.d_model)
        plt.xlabel("Dimension of positional vector")
        plt.ylabel("Position in the sequence")
        if save_path:
            plt.savefig(save_path, transparent=True)
        else:
            plt.show()

    def forward(self, x):
        x = x + self.pos_enc[:x.size(1), :]
        return self.dropout(x)


class PositionalEncoding2(nn.Module):

    def __init__(self, d_model, dropout=0, max_len=512):
        super(PositionalEncoding2, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)



if __name__ == "__main__":
    pos_enc = PositionalEncoding(32, 32)
    pos_enc.generate_heatmap("../../result_images/positional_enc_heatmap.pdf")