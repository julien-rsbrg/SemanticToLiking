from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros
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


class myGATConv(torch_geometric.nn.GATConv):
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
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
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

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            self.lin_content_src = Linear(in_channels, heads * out_channels,
                                          bias=False, weight_initializer='glorot')
            self.lin_content_dst = self.lin_content_src
        else:
            self.lin_content_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_content_dst = Linear(in_channels[1], heads * out_channels, False,
                                          weight_initializer='glorot')

        self.reset_parameters()

    def forward(  # noqa: F811
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
        return_attention_weights: Optional[bool] = None,
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
        """
        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_edge_src = x_edge_dst = self.lin_src(x).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_edge_src, x_edge_dst = x
            assert x_edge_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_edge_src = self.lin_src(x_edge_src).view(-1, H, C)
            if x_edge_dst is not None:
                x_edge_dst = self.lin_dst(x_edge_dst).view(-1, H, C)

        x_edge = (x_edge_src, x_edge_dst)

        # CHANGE HERE: x features only for the message
        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_content_src = x_content_dst = self.lin_content_src(x).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_content_src, x_content_dst = x
            assert x_content_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_content_src = self.lin_content_src(x_content_src).view(-1, H, C)
            if x_content_dst is not None:
                x_content_dst = self.lin_content_dst(x_content_dst).view(-1, H, C)

        x_content = (x_content_src, x_content_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (0*x_edge_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_edge_dst is None else (x_edge_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_edge_src.size(0)
                if x_edge_dst is not None:
                    num_nodes = min(num_nodes, x_edge_dst.size(0))
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
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr,
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
    
    def describe_parameters(self):
        lin_src_params = [param for param in self.lin_src.parameters()][0]
        lin_dst_params = [param for param in self.lin_dst.parameters()][0]

        src_att_lin_params = lin_src_params * self.att_src
        dst_att_lin_params = lin_dst_params * self.att_dst
        print("param_x_src, param_x_dst =", float(src_att_lin_params), float(dst_att_lin_params))


        lin_edge_params = [param for param in self.lin_edge.parameters()][0]
        att_lin_edge_params = lin_edge_params * self.att_edge
        print("params_edge", att_lin_edge_params.squeeze())

        bias = [param for param in self.bias][0]
        print(bias)

        print()