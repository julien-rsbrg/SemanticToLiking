
from typing import Any, Optional

import torch
from torch_geometric.nn.dense.linear import Linear, reset_bias_, reset_weight_
from torch_geometric.nn import inits

from torch import Tensor


def extended_reset_(fun_reset):
    def new_fun(value:Any,in_channels:int,initializer:Optional[str] = None) -> Tensor:
        if in_channels < 0 or initializer in {"glorot","uniform","kaiming_uniform",None}:
            return fun_reset(value,in_channels,initializer)
        elif initializer == "ones":
            inits.ones(value)
        else:
            raise RuntimeError(f"Initializer '{initializer}' not supported")

        return  value
    
    return new_fun

class myLinear(Linear):
    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        extended_reset_(reset_weight_)(self.weight, self.in_channels, self.weight_initializer)
        reset_bias_(self.bias, self.in_channels, self.bias_initializer)

        # print("self.__name__",self,"self.requires_grad_():",self.requires_grad_())
        #if not(self.requires_grad_()):
        #    self.weight.grad = None
        #    self.bias.grad = None       

    def set_requires_grad(self, value):
        self.weight.requires_grad = value

        if not(self.bias is None):
            self.bias.requires_grad = value



class MLPModel(torch.nn.Module):
    def __init__(self, c_in, c_hidden, c_out, num_layers=2, dp_rate=0.1):
        """MLPModel.

        Args:
            c_in: Dimension of input features
            c_hidden: Dimension of hidden features
            c_out: Dimension of the output features. Usually number of classes in classification
            num_layers: Number of hidden layers
            dp_rate: Dropout rate to apply throughout the network

        """
        super().__init__()
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers - 1):
            layers += [torch.nn.Linear(in_channels, out_channels), torch.nn.Sigmoid(), torch.nn.Dropout(dp_rate)]
            in_channels = c_hidden
        layers += [torch.nn.Linear(in_channels, c_out)]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        """Forward.

        Args:
            x: Input features per node

        """
        return self.layers(x)