import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all others models to provide commonly used helpers in a constant and
    desired way, like for example printing the model.
    """
    @abstractmethod
    def forward(self, *args):
        raise NotImplementedError

    @abstractmethod
    def move_to_gpu():
        raise NotImplementedError

    def __str__(self) -> str:
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        num_params = sum([np.prod(p.size()) for p in model_params])
        return super().__str__() + '\nNumber of trainable parameters: {}'.format(num_params)
