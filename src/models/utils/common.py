from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import torch
from torch import Tensor



def replace_by_value(
        original_tensor: Tensor,
        mask_samples: Optional[Tensor] = None, 
        mask_attr: Optional[Tensor] = None,
        new_value: Optional[float] = 0.0) -> torch.Tensor:
    r"""Change the nodes attributes. This function is not inplace.

        Args:
            original_tensor (torch.Tensor): The nodes' attributes.
            mask_samples (torch.Tensor): The nodes to change (if :obj:`True`, the node is changed).
            mask_attr (torch.Tensor, optional): The attributes to change (if :obj:`True`, the attribute is changed). It compounds with mask_nodes to specifically target the node and attributes to change. If set to :obj:`None`, every attribute is targetted.
                (default: :obj:`None`)
            new_value (float, optional): The new value given to the nodes' attributes. (default: :obj:`0.0`)
    """
    assert len(original_tensor.size()) == 2, "Not meant for more than 2d tensors"

    if mask_samples is None:
        mask_samples = torch.ones(original_tensor.size(0),dtype=torch.bool, device=original_tensor.device) 


    if mask_attr is None:
        mask_attr = torch.ones(original_tensor.size(1),dtype=torch.bool, device=original_tensor.device) 

    new_tensor = torch.clone(original_tensor)
    new_tensor[mask_samples,mask_attr] = new_value
    
    return new_tensor



def log_norm(x, mu, std):
    """Compute the log pdf of x,
    under a normal distribution with mean mu and standard deviation std."""
    return -0.5 * torch.log(2*np.pi*std**2) - (0.5 * (1/(std**2))* (x-mu)**2)