import torch_geometric
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import plotly.graph_objects as go

from typing import Optional, Iterable

from src.graph_analysis.clustering import cluster_nodes

def convert_float_to_hsl(l,min_l,max_l,s=100,h=235):
    if max_l < min_l:
        min_l, max_l = max_l, min_l
        print("Careful: min max lighting exchanged")

    if max_l != min_l:
        scaled_l = (l-min_l)/(max_l-min_l)*100
        scaled_l = min(int(scaled_l),90)
    else:
        scaled_l = 90
    
    return f"hsl({h:d},{s:d}%,{scaled_l:d}%)"


def convert_torch_to_networkx_graph(data: torch_geometric.data.Data, edge_attr_id: int = 0, to_undirected:bool|str = False):
    G = to_networkx(data, to_undirected=to_undirected)
    weights = {e_pair:float(data.edge_attr[e,edge_attr_id]) for e,e_pair in enumerate(G.edges())} 
    nx.set_edge_attributes(G, values = weights, name = 'weight')
    return G


def draw_torch_graph(data: torch_geometric.data.Data, 
               node_annotations: Optional[Iterable] = None,
               node_colors: Optional[Iterable] = None,
               node_color_label: Optional[str] = "",
               title: Optional[str] = ""):
    G = convert_torch_to_networkx_graph(data=data)
    draw_networkx_graph(G,
                        node_annotations=node_annotations,
                        node_colors=node_colors,
                        node_color_label=node_color_label,
                        title=title)

def draw_networkx_graph(
        G: nx.Graph, 
        node_annotations: Optional[Iterable] = None,
        node_colors: Optional[Iterable] = None,
        node_color_label: Optional[str] = "",
        title: Optional[str] = ""):
    
    # adapts: https://plotly.com/python/network-graphs/
    pos = nx.spring_layout(G, k=20*1/np.sqrt(G.number_of_nodes()), weight="weight",seed = 50, iterations=500)


    edge_attr = list(nx.get_edge_attributes(G,"weight").values())
    min_edge_attr,max_edge_attr = np.min(edge_attr),np.max(edge_attr)
    
    edge_traces = []
    for edge in G.edges():
        edge_x = []
        edge_y = []
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_y.append(y0)
        edge_y.append(y1)

        color = convert_float_to_hsl(float(G.get_edge_data(edge[0],edge[1])["weight"]),min_edge_attr,max_edge_attr)
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5,color=color),
            hoverinfo='none', #to change
            mode='lines'
        )
        edge_traces.append(edge_trace)
    

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=node_colors,
            size=10,
            colorbar=dict(
                thickness=15,
                title=dict(
                    text=node_color_label, # to change
                    side='right'
                ),
                xanchor='left',
            ),
            line_width=2))
    
    # colors
    node_text = []
    for node in G.nodes():
        # node_adjacencies.append(len(adjacencies[1]))
        node_text.append(str(node_annotations[node])+f":{node_colors[node]:.2f}")

    # node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    # plot
    fig = go.Figure(data=edge_traces+[node_trace],
             layout=go.Layout(
                title=dict(
                    text=title,
                    font=dict(
                        size=16
                    )
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    fig.show()




def draw_clustering(G:nx.Graph,
                    opposite_weight: bool = True,
                    upper_quantile_removed:float=0.8,
                    node_annotations: Optional[Iterable] = None,
                    node_colors: Optional[Iterable] = None,
                    node_color_label: Optional[str] = ""):
    connected_components_index, G_MST = cluster_nodes(G, opposite_weight = opposite_weight, upper_quantile_removed = upper_quantile_removed)

    for nodes_index in connected_components_index:
        new_component_nodes_txt = []
        for id in nodes_index:
            new_component_nodes_txt.append(node_annotations[id])
        print("Component's annotations =", new_component_nodes_txt)

        n_nodes = len(nodes_index)
        if n_nodes > 1: 
            component_subgraph = G_MST.subgraph(nodes_index)
            weights = list(nx.get_edge_attributes(component_subgraph,"weight").values())
            weight_message = f"min weight = {np.min(weights):.3f}, max weights = {np.max(weights):.3f}"

            draw_networkx_graph(
                component_subgraph,
                node_annotations=node_annotations,
                node_colors = node_colors,
                node_color_label = node_color_label,
                title=f"Component with {n_nodes:d} nodes ("+ weight_message +")")