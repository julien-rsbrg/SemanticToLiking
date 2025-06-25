from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pandas as pd

import torch
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros, ones
from torch import Tensor
import torch_geometric.nn
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils.sparse import set_sparse_value
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
    to_dense_adj
)
from torch_geometric.nn.dense.linear import Linear
from torch.autograd import Variable

from src.models.nn.bnn_layers import MeanFieldGaussianFeedForward, VIModule
from src.models.nn.nn_layers import myLinear

class MyGATConv(torch_geometric.nn.GATConv):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        src_content_mask: Optional[torch.Tensor] = None,
        src_edge_mask: Optional[torch.Tensor] = None,
        dst_content_mask: Optional[torch.Tensor] = None,
        dst_edge_mask: Optional[torch.Tensor] = None,
        src_content_require_grad: bool = True,
        src_edge_require_grad: bool = True,
        dst_content_require_grad: bool = True,
        dst_edge_require_grad: bool = True,
        src_content_weight_initializer: str = "glorot",
        src_edge_weight_initializer: str = "glorot",
        dst_content_weight_initializer: str = "glorot",
        dst_edge_weight_initializer: str = "glorot",
        edge_weight_initializer: str = "glorot",
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')

        # need before the reset
        self.lin = self.lin_src_content = self.lin_src_edge = self.lin_dst_content = self.lin_dst_edge = None

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=add_self_loops,
            edge_dim=edge_dim,
            fill_value=fill_value,
            bias=bias,
            **kwargs
        )

        self.n_free_params = 0

        self.lin_src = self.lin_dst = None
        if edge_dim is not None:
            self.lin_edge = myLinear(edge_dim, 
                                     heads * out_channels, 
                                     bias=False,
                                     weight_initializer=edge_weight_initializer) # for uniformity
            self.n_free_params += edge_dim*heads*out_channels


        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int) and (src_content_mask is None) and (src_edge_mask is None) and (dst_content_mask is None) and (dst_edge_mask is None):
            raise NotImplementedError() # should check forward() in that case
            #self.lin = Linear(in_channels, heads * out_channels, bias=False,weight_initializer='glorot')
        else:
            # TODO: create function for this setup repeated 4 times
            if src_content_mask is None:
                src_content_in_channels = in_channels[0]
                self.src_content_mask = torch.ones(src_content_in_channels,dtype=torch.bool)
            else:
                assert isinstance(src_content_mask,torch.Tensor) and src_content_mask.dtype == torch.bool

                src_content_in_channels = int(torch.sum(src_content_mask))
                self.src_content_mask = src_content_mask
            
            if src_edge_mask is None:
                src_edge_in_channels = in_channels[0]
                self.src_edge_mask = torch.ones(src_edge_in_channels,dtype=torch.bool)
            else:
                assert isinstance(src_edge_mask,torch.Tensor) and src_edge_mask.dtype == torch.bool
                src_edge_in_channels = int(torch.sum(src_edge_mask))
                self.src_edge_mask = src_edge_mask

            if dst_content_mask is None:
                dst_content_in_channels = in_channels[1]
                self.dst_content_mask = torch.ones(dst_content_in_channels,dtype=torch.bool)
            else:
                assert isinstance(dst_content_mask,torch.Tensor) and dst_content_mask.dtype == torch.bool
                dst_content_in_channels = int(torch.sum(dst_content_mask))
                self.dst_content_mask = dst_content_mask
            
            if dst_edge_mask is None:
                dst_edge_in_channels = in_channels[1]
                self.dst_edge_mask = torch.ones(dst_edge_in_channels,dtype=torch.bool)
            else:
                assert isinstance(dst_edge_mask,torch.Tensor) and dst_edge_mask.dtype == torch.bool
                dst_edge_in_channels = int(torch.sum(dst_edge_mask))
                self.dst_edge_mask = dst_edge_mask
                        
            self.lin_src_edge = myLinear(src_edge_in_channels, 
                                       heads * out_channels, 
                                       bias=False,
                                       weight_initializer=src_edge_weight_initializer)            
            self.lin_dst_edge = myLinear(dst_edge_in_channels, 
                                       heads * out_channels, 
                                       bias=False,
                                       weight_initializer=dst_edge_weight_initializer)
            self.n_free_params += src_edge_in_channels*heads*out_channels*int(src_edge_require_grad)
            self.n_free_params += dst_edge_in_channels*heads*out_channels*int(dst_edge_require_grad)

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            assert torch.all(src_content_mask == dst_content_mask) and src_content_require_grad == dst_content_require_grad and src_content_weight_initializer == dst_content_weight_initializer, "It is useless to distinguish the two if there will have the same model behind." 
            self.lin_src_content = myLinear(in_channels, heads * out_channels,
                                          bias=False, weight_initializer=src_content_weight_initializer)
            self.lin_dst_content = self.lin_src_content
            self.n_free_params += in_channels*heads*out_channels*int(src_content_require_grad)
            self.lin_src_content.set_requires_grad(src_content_require_grad)
        else:
            self.lin_src_content = myLinear(src_content_in_channels, heads * out_channels, bias=False,
                                          weight_initializer=src_content_weight_initializer)
            self.lin_dst_content = myLinear(dst_content_in_channels, heads * out_channels, bias=False,
                                          weight_initializer=dst_content_weight_initializer)

            self.n_free_params += src_content_in_channels*heads*out_channels*int(src_content_require_grad)
            self.n_free_params += dst_content_in_channels*heads*out_channels*int(dst_content_require_grad)

            self.lin_src_content.set_requires_grad(src_content_require_grad)
            self.lin_dst_content.set_requires_grad(dst_content_require_grad) 

        self.lin_src_edge.set_requires_grad(src_edge_require_grad) 
        self.lin_dst_edge.set_requires_grad(dst_edge_require_grad) 

        self.att_src.requires_grad = False
        self.att_dst.requires_grad = False

        if self.edge_dim is not None:
            self.att_edge.requires_grad = False

        self.reset_parameters()

        self.src_content_require_grad = src_content_require_grad
        self.dst_content_require_grad = dst_content_require_grad
        self.src_edge_require_grad = src_edge_require_grad
        self.dst_edge_require_grad = dst_edge_require_grad

        self.src_content_weight_initializer = src_content_weight_initializer
        self.src_edge_weight_initializer = src_edge_weight_initializer
        self.dst_content_weight_initializer = dst_content_weight_initializer
        self.dst_edge_weight_initializer = dst_edge_weight_initializer
        self.edge_weight_initializer = edge_weight_initializer
    
    def reset_parameters(self):
        super().reset_parameters()
        if self.lin_src_content is not None:
            self.lin_src_content.reset_parameters()
        if self.lin_src_edge is not None:
            self.lin_src_edge.reset_parameters()
        if self.lin_dst_content is not None:
            self.lin_dst_content.reset_parameters()
        if self.lin_dst_edge is not None:
            self.lin_dst_edge.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
            
        ones(self.att_src)
        ones(self.att_dst)
        if self.edge_dim is not None:
            ones(self.att_edge)
        zeros(self.bias)
        

    def forward(  # noqa: F811
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
        return_attention_weights: Optional[bool] = None
    ) -> Union[
            Tensor,
            Tuple[Tensor, Tuple[Tensor, Tensor]],
            Tuple[Tensor, SparseTensor],
    ]:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            size ((int, int), optional): The shape of the adjacency matrix.
                (default: :obj:`None`)
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
            mask_use_for_inference (torch.Tensor(torch.bool), optional):
                the nodes that we can use to infer about the other nodes. (default: :obj:`None`)

        """
        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src_edge = self.lin_src_edge(x[...,self.src_edge_mask]).view(-1, H, C)
            x_dst_edge = self.lin_dst_edge(x[...,self.dst_edge_mask]).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src_edge, x_dst_edge = x
            assert x_src_edge.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src_edge = self.lin_src_edge(x_src_edge[...,self.src_edge_mask]).view(-1, H, C)
            if x_dst_edge is not None:
                x_dst_edge = self.lin_dst_edge(x_dst_edge[...,self.dst_edge_mask]).view(-1, H, C)

        # CHANGE HERE: x features only for the message
        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src_content = self.lin_src_content(x[...,self.src_content_mask]).view(-1, H, C)
            x_dst_content = self.lin_dst_content(x[...,self.dst_content_mask]).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src_content, x_dst_content = x
            assert x_src_content.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src_content = self.lin_src_content(x_src_content[...,self.src_content_mask]).view(-1, H, C)
            if not(x_dst_content is  None):
                x_dst_content = self.lin_dst_content(x_dst_content[...,self.dst_content_mask]).view(-1, H, C)

        x_content = (x_src_content, x_dst_content)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src_edge * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst_edge is None else (x_dst_edge * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src_edge.size(0)
                if x_dst_edge is not None:
                    num_nodes = min(num_nodes, x_dst_edge.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, 
                                  alpha=alpha, 
                                  edge_attr=edge_attr,
                                  size=size)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        # CHANGE HERE: it is enough to propagate something different than the x used for the 
        # alpha for the purpose of dissociating the parameters involved in those two cases
        # (stronger than att_src and att_dst influences) 
        out = self.propagate(edge_index, x=x_content, alpha=alpha, size=size)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out
        
    
    def get_dict_parameters(self):
        def get_description_tensor(tensor_to_describe,tensor_name:str):
            if tensor_to_describe is None:
                param_id = 0
                return {f"{tensor_name}_{param_id:d}": [None]}
            
            parameters_spec = {}
            
            new_params = [float(param.squeeze().detach().cpu()) for param in tensor_to_describe.flatten()]

            if len(new_params) == 0:
                new_params.append(None)
            
            for param_id, param in enumerate(new_params):
                parameters_spec[f"{tensor_name}_{param_id:d}"] = [param]
                
            return parameters_spec
        

        def get_description_model(model, model_name:str):
            new_parameters_spec = [get_description_tensor(param,tensor_name="") for param in model.parameters()]

            parameters_spec = {}
            counter = 0
            for single_param_spec in new_parameters_spec:
                for lparam in single_param_spec.values():
                    parameters_spec[f"{model_name}_{counter:d}"] = lparam
                    counter += 1
                    
            return parameters_spec

        
            

        parameters_spec = {}

        if not(self.lin_src_content is None):
            new_parameters_spec = get_description_model(self.lin_src_content,"src_content")
            parameters_spec.update(new_parameters_spec)
        
        new_parameters_spec = get_description_tensor(self.att_src,"att_src_edge")
        parameters_spec.update(new_parameters_spec)

        if not(self.lin_src_edge is None):
            new_parameters_spec = get_description_model(self.lin_src_edge,"src_edge")
            if self.att_src.numel() == 1:
                full_parameters_spec = {}
                for k,v in new_parameters_spec.items():
                    if v[0] is not None:
                        full_parameters_spec["full_"+k] = float(self.att_src.squeeze())*v[0]
                    else:
                        full_parameters_spec["full_"+k] = None
            parameters_spec.update(new_parameters_spec)
            parameters_spec.update(full_parameters_spec)
        
        if not(self.lin_dst_content is None):
            new_parameters_spec = get_description_model(self.lin_dst_content,"dst_content")
            parameters_spec.update(new_parameters_spec)

        new_parameters_spec = get_description_tensor(self.att_dst,"att_dst_edge")
        parameters_spec.update(new_parameters_spec)

        if not(self.lin_dst_edge is None):
            new_parameters_spec = get_description_model(self.lin_dst_edge,"dst_edge")
            if self.att_dst.numel() == 1:
                full_parameters_spec = {}
                for k,v in new_parameters_spec.items():
                    if v[0] is not None:
                        full_parameters_spec["full_"+k] = float(self.att_dst.squeeze())*v[0]
                    else:
                        full_parameters_spec["full_"+k] = None
            parameters_spec.update(new_parameters_spec)
            parameters_spec.update(full_parameters_spec)

        new_parameters_spec = get_description_tensor(self.att_edge,"att_edge")
        parameters_spec.update(new_parameters_spec)

        if not(self.lin_edge is None):
            new_parameters_spec = get_description_model(self.lin_edge,"edge")
            if self.att_edge.numel() == 1:
                full_parameters_spec = {}
                for k,v in new_parameters_spec.items():
                    if v[0] is not None:
                        full_parameters_spec["full_"+k] = float(self.att_edge.squeeze())*v[0]
                    else:
                        full_parameters_spec["full_"+k] = None
            parameters_spec.update(new_parameters_spec)
            parameters_spec.update(full_parameters_spec)

        new_parameters_spec = get_description_tensor(self.bias,"bias")
        parameters_spec.update(new_parameters_spec)

        return pd.DataFrame(parameters_spec)


    def get_config(self):
        config = {
            "name":"MyGATConv",
            "parameters":{
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "heads": self.heads,
                "concat": self.concat,
                "negative_slope": self.negative_slope,
                "dropout": self.dropout,
                "add_self_loops": self.add_self_loops,
                "edge_dim": self.edge_dim,
                "fill_value": self.fill_value,
                "bias": not(self.bias is None),
                "src_content_mask": self.src_content_mask,
                "src_edge_mask": self.src_edge_mask,
                "dst_content_mask": self.dst_content_mask,
                "dst_edge_mask": self.dst_edge_mask,
                "src_content_require_grad": self.src_content_require_grad,
                "src_edge_require_grad": self.src_edge_require_grad,
                "dst_content_require_grad": self.dst_content_require_grad,
                "dst_edge_require_grad": self.dst_edge_require_grad,
                "src_content_weight_initializer": self.src_content_weight_initializer,
                "src_edge_weight_initializer": self.src_edge_weight_initializer,
                "dst_content_weight_initializer": self.dst_content_weight_initializer,
                "dst_edge_weight_initializer": self.dst_edge_weight_initializer,
                "edge_weight_initializer": self.edge_weight_initializer,
            },
            "n_free_params":self.n_free_params
        }
        return config
    

class MyGATConvNLeaps(MyGATConv):
    def __init__(self, n_leaps: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_leaps = n_leaps
    
    def forward(self,*args,**kwargs):
        complete_train_mask = kwargs["complete_train_mask"]
        del kwargs["complete_train_mask"]
        present_keys = set(kwargs.keys()).intersection({"x","edge_index","edge_attr","size","return_attention_weights"})
        kwargs = {k: kwargs[k] for k in present_keys}

        no_args = len(args) == 0
        if no_args:
            out = kwargs["x"].clone()
            kwargs["x"] = out
            for _ in range(self.n_leaps): 
                kwargs["x"][complete_train_mask] = out[complete_train_mask]
                out = super().forward(**kwargs) 
            kwargs["x"][complete_train_mask] = out[complete_train_mask]
            out = kwargs["x"]
        else:
            out = args[0].clone()
            args = tuple([out,*args[1:]])
            for _ in range(self.n_leaps): 
                args[0][complete_train_mask] = out[complete_train_mask]
                out = super().forward(*args,**kwargs)
            args[0][complete_train_mask] = out[complete_train_mask]
            out = args[0]

        return out
    
    def get_config(self):
        config = super().get_config()
        config["name"] = "MyGATConvNLeaps"
        config["parameters"]["n_leaps"] = self.n_leaps
        return config
    


class MyBGATConv(MyGATConv,VIModule):
    # TODO update
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        n_latent: int = 1,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        src_content_mask: Optional[torch.Tensor] = None,
        src_edge_mask: Optional[torch.Tensor] = None,
        dst_content_mask: Optional[torch.Tensor] = None,
        dst_edge_mask: Optional[torch.Tensor] = None,
        device = torch.device("cpu"),
        **kwargs,
    ):
        self.device = device

        VIModule.__init__(self)
        MyGATConv.__init__(self,
            in_channels = in_channels,
            out_channels = out_channels,
            heads = heads,
            concat = concat,
            negative_slope = negative_slope,
            dropout = dropout,
            add_self_loops = add_self_loops,
            edge_dim = edge_dim,
            fill_value = fill_value,
            bias = bias,
            src_content_mask = src_content_mask,
            src_edge_mask = src_edge_mask,
            dst_content_mask = dst_content_mask,
            dst_edge_mask = dst_edge_mask,
            **kwargs,
        )
        if isinstance(in_channels, int) and (src_content_mask is None) and (src_edge_mask is None) and (dst_content_mask is None) and (dst_edge_mask is None):
            raise NotImplementedError() # should check forward() in that case
            #self.lin = Linear(in_channels, heads * out_channels, bias=False,weight_initializer='glorot')
        else:
            # TODO: create function for this setup repeated 4 times
            if src_content_mask is None:
                src_content_in_channels = in_channels[0]
                self.src_content_mask = torch.ones(src_content_in_channels,dtype=torch.bool)
            else:
                assert isinstance(src_content_mask,torch.Tensor) and src_content_mask.dtype == torch.bool
                src_content_in_channels = int(torch.sum(src_content_mask))
                self.src_content_mask = src_content_mask
            
            if src_edge_mask is None:
                src_edge_in_channels = in_channels[0]
                self.src_edge_mask = torch.ones(src_edge_in_channels,dtype=torch.bool)
            else:
                assert isinstance(src_edge_mask,torch.Tensor) and src_edge_mask.dtype == torch.bool
                src_edge_in_channels = int(torch.sum(src_edge_mask))
                self.src_edge_mask = src_edge_mask

            if dst_content_mask is None:
                dst_content_in_channels = in_channels[1]
                self.dst_content_mask = torch.ones(dst_content_in_channels,dtype=torch.bool)
            else:
                assert isinstance(dst_content_mask,torch.Tensor) and dst_content_mask.dtype == torch.bool
                dst_content_in_channels = int(torch.sum(dst_content_mask))
                self.dst_content_mask = dst_content_mask
            
            if dst_edge_mask is None:
                dst_edge_in_channels = in_channels[1]
                self.dst_edge_mask = torch.ones(dst_edge_in_channels,dtype=torch.bool)
            else:
                assert isinstance(dst_edge_mask,torch.Tensor) and dst_edge_mask.dtype == torch.bool
                dst_edge_in_channels = int(torch.sum(dst_edge_mask))
                self.dst_edge_mask = dst_edge_mask
                        
            self.lin_src_edge = MeanFieldGaussianFeedForward(
                in_features=src_edge_in_channels, 
                out_features=heads * out_channels, 
                prior_weights_m = 1.0,
                prior_weights_s = 1.0,
                n_latent=n_latent,
                has_bias=False,
                device = self.device)            
            self.lin_dst_edge = MeanFieldGaussianFeedForward(
                in_features = dst_edge_in_channels, 
                out_features = heads * out_channels, 
                prior_weights_m = 1.0,
                prior_weights_s = 1.0,
                n_latent=n_latent,
                has_bias = False,
                device = self.device)   

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            self.lin_src_content = MeanFieldGaussianFeedForward(
                in_features = in_channels, 
                out_features = heads * out_channels, 
                prior_weights_m = 1.0,
                prior_weights_s = 1.0,
                n_latent=n_latent,
                has_bias = False,
                device = self.device)
            self.lin_dst_content = self.lin_src_content
        else:
            self.lin_src_content = MeanFieldGaussianFeedForward(
                in_features = src_content_in_channels, 
                out_features = heads * out_channels, 
                prior_weights_m = 1.0,
                prior_weights_s = 1.0,
                n_latent=n_latent,
                has_bias = False,
                device = self.device)
            self.lin_dst_content = MeanFieldGaussianFeedForward(
                in_features = dst_content_in_channels, 
                out_features = heads * out_channels, 
                prior_weights_m = 1.0,
                prior_weights_s = 1.0,
                n_latent=n_latent,
                has_bias = False,
                device = self.device)
            
        if self.edge_dim is not None:
            self.lin_edge = MeanFieldGaussianFeedForward(
                in_features = edge_dim, 
                out_features = heads * out_channels, 
                prior_weights_m = 1.0,
                prior_weights_s = 1.0,
                n_latent=n_latent,
                has_bias = False,
                device = self.device)
            
    
    def get_dict_parameters(self):
        def get_description_tensor(tensor_to_describe,tensor_name:str):
            if tensor_to_describe is None:
                param_id = 0
                return {f"{tensor_name}_{param_id:d}": [None]}
            
            parameters_spec = {}
            
            new_params = [float(param.squeeze().detach().cpu()) for param in tensor_to_describe.flatten()]

            if len(new_params) == 0:
                new_params.append(None)
            
            for param_id, param in enumerate(new_params):
                parameters_spec[f"{tensor_name}_{param_id:d}"] = [param]
                
            return parameters_spec
        

        def get_description_bayesian_linear(model, model_name:str):
            weights_parameters_spec = [get_description_tensor(param,tensor_name="") for param in (model.weights_m,model.weights_s)]
            
            if model.has_bias:
                bias_parameters_spec = []
                bias_parameters_spec.append(get_description_tensor(model.bias_m,tensor_name=""))
                bias_parameters_spec.append(get_description_tensor(model.bias_s,tensor_name=""))

            parameters_spec = {}
            for i,lparam in enumerate(weights_parameters_spec[0].values()):
                parameters_spec[f"{model_name}_weight_mean_{i:d}"] = lparam
            
            for i,lparam in enumerate(weights_parameters_spec[1].values()):
                parameters_spec[f"{model_name}_weight_std_{i:d}"] = lparam

            if model.has_bias:
                for i,lparam in enumerate(bias_parameters_spec[0].values()):
                    parameters_spec[f"{model_name}_bias_mean_{i:d}"] = lparam
                
                for i,lparam in enumerate(bias_parameters_spec[1].values()):
                    parameters_spec[f"{model_name}_bias_std_{i:d}"] = lparam
            
            return parameters_spec

        parameters_spec = {}

        if not(self.lin_src_content is None):
            new_parameters_spec = get_description_bayesian_linear(self.lin_src_content,"src_content")
            parameters_spec.update(new_parameters_spec)
        
        new_parameters_spec = get_description_tensor(self.att_src,"att_src_edge")
        parameters_spec.update(new_parameters_spec)

        if not(self.lin_src_edge is None):
            new_parameters_spec = get_description_bayesian_linear(self.lin_src_edge,"src_edge")
            full_parameters_spec = {}
            if self.att_src.numel() == 1:
                for k,v in new_parameters_spec.items():
                    if v[0] is not None:
                        full_parameters_spec["full_"+k] = float(self.att_src.squeeze())*v[0]
                    else:
                        full_parameters_spec["full_"+k] = None

            parameters_spec.update(new_parameters_spec)
            parameters_spec.update(full_parameters_spec)
        
        if not(self.lin_dst_content is None):
            new_parameters_spec = get_description_bayesian_linear(self.lin_dst_content,"dst_content")
            parameters_spec.update(new_parameters_spec)

        new_parameters_spec = get_description_tensor(self.att_dst,"att_dst_edge")
        parameters_spec.update(new_parameters_spec)

        if not(self.lin_dst_edge is None):
            new_parameters_spec = get_description_bayesian_linear(self.lin_dst_edge,"dst_edge")
            full_parameters_spec = {}
            if self.att_dst.numel() == 1:
                full_parameters_spec = {}
                for k,v in new_parameters_spec.items():
                    if v[0] is not None:
                        full_parameters_spec["full_"+k] = float(self.att_dst.squeeze())*v[0]
                    else:
                        full_parameters_spec["full_"+k] = None
            parameters_spec.update(new_parameters_spec)
            parameters_spec.update(full_parameters_spec)

        new_parameters_spec = get_description_tensor(self.att_edge,"att_edge")
        parameters_spec.update(new_parameters_spec)

        if not(self.lin_edge is None):
            new_parameters_spec = get_description_bayesian_linear(self.lin_edge,"edge")
            full_parameters_spec = {}
            if self.att_edge.numel() == 1:
                full_parameters_spec = {}
                for k,v in new_parameters_spec.items():
                    if v[0] is not None:
                        full_parameters_spec["full_"+k] = float(self.att_edge.squeeze())*v[0]
                    else:
                        full_parameters_spec["full_"+k] = None
            parameters_spec.update(new_parameters_spec)
            parameters_spec.update(full_parameters_spec)

        new_parameters_spec = get_description_tensor(self.bias,"bias")
        parameters_spec.update(new_parameters_spec)

        return pd.DataFrame(parameters_spec)