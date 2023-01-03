from __future__ import absolute_import, division, print_function

import torch
from torch import nn

from logging import getLogger
import utils.device_handler as DeviceHandler


class FcLayers(nn.Module):
    """Allows to create multiple fc-layers at once"""
    
    def __init__(self, dims):
        super(FcLayers, self).__init__()
        self.logger = getLogger(self.__class__.__name__)
        self.fc_layers = nn.ModuleList([nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(dims[:-1], dims[1:])])
        self.num_fc_layer = len(self.fc_layers)
        self.relu_activation = nn.ReLU()

    def forward(self, x):
        for i, fc_layer in enumerate(self.fc_layers):
            x = fc_layer(x)
            if i != (self.num_fc_layer - 1):
                x = self.relu_activation(x)
        return x


class DenseLayer(nn.Module):
    """Simple MLP"""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super(DenseLayer, self).__init__()
        self.fc_1 = nn.Linear(in_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc_1(x))
        x = self.relu(self.fc_2(x))
        return self.fc_3(x)


class MultiLevelClassificationLayer(nn.Module):
    """Multi-level interconnected classification heads as described in the publication by Rojas-Carulla et al."""
    
    def __init__(self, dims, classes_per_rank):
        super(MultiLevelClassificationLayer, self).__init__()
        self.logger = getLogger(self.__class__.__name__)
        self.device = DeviceHandler.get_device()
        
        # fc-layer for classification on each level L
        self.cls_layers = nn.ModuleList()
        # fc-layer to transform input from L-1 to L
        self.trans_layers = nn.ModuleList()
        
        # ranks are expected in ascneding order (e.g. species, genus, phylum...)
        # append -1 for transition to coarsest level and reverse list
        # since the start should be on the coarsest level
        full_classes_per_rank = classes_per_rank.copy()
        full_classes_per_rank.append(-1)
        full_classes_per_rank = list(reversed(full_classes_per_rank))

        for prev_num_classes, cur_num_classes in zip(full_classes_per_rank[:-1], full_classes_per_rank[1:]):
            cls_dims = dims.copy()
            cls_dims.append(cur_num_classes)
            cls_layer = nn.Sequential(*[nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(cls_dims[:-1], cls_dims[1:])])
            self.cls_layers.append(cls_layer)
            # in the first iteration, add identity function, since no higher level prediction can be passed
            trans_layer = nn.Linear(prev_num_classes, cur_num_classes) if prev_num_classes != -1 else nn.Identity()
            self.trans_layers.append(trans_layer)

        self.activation = nn.ReLU()

    def forward(self, x):
        # dim(x) = (batch_sz, pre_fc_hidden_dim)
        out = []
        current_prediction = torch.tensor([0.0], device=self.device)
        for cls_layer, trans_layer in zip(self.cls_layers, self.trans_layers):
            cur_lv_out = cls_layer(x)
            last_lv_out = trans_layer(current_prediction)
            current_prediction = self.activation(cur_lv_out) + self.activation(last_lv_out)
            # prepend elements since the actual prediction should returned in lowest-to-highest fashion
            out.insert(0, current_prediction)
        return out


if __name__ == "__main__":
    # Fc layers test
    fcl = FcLayers([2 * 300 * 2, 3000, 3000, 120])
    fcl(torch.randn((2048, 2 * 300 * 2)))
    print(fcl)

    # multi level layers test
    mlcl = MultiLevelClassificationLayer([512], [3000, 120, 40])
    rand_tensor = torch.rand((2048, 512))
    result = mlcl(rand_tensor)
    print(result)
    print(len(result))
    for lv_result in result:
        print(lv_result.shape)