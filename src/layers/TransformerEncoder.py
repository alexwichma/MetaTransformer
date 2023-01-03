from __future__ import absolute_import, division, print_function

from torch import nn


class TransformerEncoder(nn.Module):
    """Wrapper around a transformer encoder block"""

    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout, activation="relu") -> None:
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_ff, dropout=dropout, activation=activation)
        layer_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=layer_norm)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        return self.encoder(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
