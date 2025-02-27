from typing import Optional, Callable, Iterable

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


import torch_geometric
from torch_geometric.utils import (
            add_self_loops,
            is_torch_sparse_tensor,
            remove_self_loops,
            softmax,
            to_dense_adj
        )

from src.models.ML_frameworks import GNNFramework

def get_prediction_table(model:GNNFramework,graph:torch_geometric.data.Data):
    preds = model.predict(graph.x,
                          graph.edge_index,
                          graph.edge_attr)
                
    preds = np.array(preds.detach().to("cpu"))
    preds = np.squeeze(preds)
    labels = np.array(graph.y)
    labels = np.squeeze(labels)

    prediction_table = pd.DataFrame()
    prediction_table["prediction"] = preds
    prediction_table["label"] = labels  
    return prediction_table

def plot_errors_labels_comparison(model:GNNFramework,graph:torch_geometric.data.Data,plot_attention_weights=False):
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

    errors = labels - preds 


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
    fig.add_trace(go.Scatter(
        x = labels,
        y = preds,
        mode = "markers",
        marker=dict(color=preds)
    ))
    fig.update_layout(
        title="Predictions vs labels",
        xaxis_title="Label",
        yaxis_title="Prediction"
    )
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x = errors)
    )
    fig.update_layout(
        title="Residual hisogram",
        xaxis_title="Residual value",
        yaxis_title="Count"
    )
    fig.show()

    print("labels=",labels)
    print("preds=",preds)

    if plot_attention_weights:
        matrix_alpha = to_dense_adj(adj, edge_attr = alpha).cpu().detach()
        matrix_alpha = matrix_alpha.squeeze()
        fig = px.imshow(matrix_alpha)
        fig.update_layout(
            title="Alpha: the message passing strength between nodes"
        )
        fig.show()


def balance_nrows_ncols(n):
    nrows,ncols = 0, 0
    for i in range(n):
        nrows += i%2
        ncols += (i+1)%2
    return nrows,ncols


def plot_histograms(data:pd.DataFrame, 
                    vars_names:Optional[Iterable[str]] = None, 
                    mask:Optional[np.ndarray] = None,
                    mask_legend: Optional[str] = "selected by mask",
                    title: Optional[str] = ""):
    if vars_names is None:
        vars_names = data.columns.tolist()
    if mask is None:
        mask = np.ones(len(data),dtype=bool)

    nvars = len(vars_names) 
    nrows,ncols = balance_nrows_ncols(nvars)

    fig = make_subplots(rows=nrows, 
                        cols=ncols, 
                        subplot_titles=[str(v) for v in vars_names],
                        vertical_spacing = 0.3,
                        horizontal_spacing = 0.1)

    data_mask_selected = data[mask]
    data_not_mask_selected = data[~mask]
    for col_id, col_name in enumerate(vars_names):
        row,col = col_id%nrows + 1, col_id//nrows + 1, 
        
        fig.add_trace(go.Histogram(x=data_mask_selected[col_name],
                                   name=col_name,
                                   legendgroup=mask_legend,
                                   legendgrouptitle={"text":mask_legend},
                                   marker_color="#3dc244"),
                                   row=row,col=col)
        fig.add_trace(go.Histogram(x=data_not_mask_selected[col_name],
                                   name=col_name,
                                   legendgroup="not("+mask_legend+")",
                                   legendgrouptitle={"text":"not("+mask_legend+")"},
                                   marker_color='#cf0c1c'),
                                   row=row,col=col)
        fig.add_trace(go.Histogram(x=data[col_name],
                                   name=col_name,
                                   legendgroup="all samples",
                                   legendgrouptitle={"text":"all samples"},
                                   marker_color="#1331f6"),
                                   row=row,col=col)
        
        
        fig['layout'][f'xaxis{col_id+1:d}']['title']='value'
        fig['layout'][f'yaxis{col_id+1:d}']['title']='count'


    fig.update_layout(barmode='group', title=title)
    fig.update_traces(opacity=0.9)
    fig.show()