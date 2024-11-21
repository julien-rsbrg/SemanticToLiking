import numpy as np

import plotly.graph_objects as go
import plotly.express as px

import torch_geometric
from torch_geometric.utils import (
            add_self_loops,
            is_torch_sparse_tensor,
            remove_self_loops,
            softmax,
            to_dense_adj
        )

from src.models.frameworks import GNN_naive_framework


def plot_errors_labels_comparison(model:GNN_naive_framework,graph:torch_geometric.data.Data,plot_attention_weights=False):
    if plot_attention_weights:
        preds, (adj, alpha) = model.predict(graph.x,
                                    graph.edge_index,
                                    graph.edge_attr,
                                    return_attention_weights=True)
    else:
        # should remove the return_attentio_weights since the implementation only cares about the presence of a bool 
        preds = model.predict(graph.x,
                              graph.edge_index,
                              graph.edge_attr)
                
    preds = np.array(preds.detach().to("cpu"))
    preds = np.squeeze(preds)
    labels = np.array(graph.y)
    labels = np.squeeze(labels)

    errors = labels-preds


    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = labels,
        y = errors,
        mode = "markers",
        marker=dict(color=preds)
    ))
    fig.update_layout(
        title="Residual depending on label value",
        xaxis_title="Label",
        yaxis_title="Residual"
    )
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x = preds)
    )
    fig.update_layout(
        title="Residual depending on label value",
        xaxis_title="Label",
        yaxis_title="Residual"
    )
    fig.show()

    if plot_attention_weights:
        matrix_alpha = to_dense_adj(adj, edge_attr = alpha).cpu().detach()
        matrix_alpha = matrix_alpha.squeeze()
        fig = px.imshow(matrix_alpha)
        fig.update_layout(
            title="Alpha: the message passing strength between nodes"
        )
        fig.show()


def print_parameters_gat_layer(gat_layer):
    lin_src_params = [param for param in gat_layer.lin_src.parameters()][0]
    lin_dst_params = [param for param in gat_layer.lin_dst.parameters()][0]

    src_att_lin_params = lin_src_params * gat_layer.att_src
    dst_att_lin_params = lin_dst_params * gat_layer.att_dst
    #lin_params.squeeze()
    print("param_x_src, param_x_dst =", float(src_att_lin_params), float(dst_att_lin_params))


    lin_edge_params = [param for param in gat_layer.lin_edge.parameters()][0]
    att_lin_edge_params = lin_edge_params * gat_layer.att_edge
    print("params_edge", att_lin_edge_params.squeeze())

    bias = [param for param in gat_layer.bias][0]
    print(bias)
