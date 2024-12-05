import numpy as np
import networkx as nx

from typing import Optional, Callable, Iterable

def add_invert_edge_attr(G:nx.Graph,edge_attr_name:str,new_edge_attr_name:str):
    new_G = G.copy()
    edge_attr = nx.get_edge_attributes(new_G,name=edge_attr_name)
    new_edge_attr = {k:1/(v+1e-6) for k,v in edge_attr.items()}
    nx.set_edge_attributes(new_G, values = new_edge_attr, name = new_edge_attr_name)
    return new_G

def add_opposite_edge_attr(G:nx.Graph,edge_attr_name:str,new_edge_attr_name:str):
    new_G = G.copy() 
    edge_attr = nx.get_edge_attributes(new_G,name=edge_attr_name)
    new_edge_attr = {k:-v for k,v in edge_attr.items()}
    nx.set_edge_attributes(G, values = new_edge_attr, name = new_edge_attr_name)
    return new_G

def cluster_nodes(G:nx.Graph, opposite_weight:bool = False, upper_quantile_removed:float=.8):
    """
    Cluster nodes with Kruskal's algorithm. 
    The greater the weight the less likely the two nodes are in the same component.
    """
    edge_attr_name = "weight" 
    if opposite_weight:
        new_G = add_opposite_edge_attr(G,"weight","opposite_weight")
        edge_attr_name = "opposite_weight"

    G_MST = nx.minimum_spanning_tree(new_G, weight=edge_attr_name)

    opposite_weight = list(nx.get_edge_attributes(G_MST,name=edge_attr_name).values())
    q = np.quantile(opposite_weight,upper_quantile_removed)
    for edge,edge_opp_weight in nx.get_edge_attributes(G_MST,edge_attr_name).items():
        if edge_opp_weight >= q:
            G_MST.remove_edge(edge[0],edge[1])

    connected_components_index = sorted(nx.connected_components(G_MST), key=len, reverse=True)
    return connected_components_index, G_MST

def draw_clustering(G:nx.Graph,
                    opposite_weight: bool = True,
                    upper_quantile_removed:float=0.8,
                    node_annotations: Optional[Iterable] = None,
                    node_colors: Optional[Iterable] = None,
                    node_color_label: Optional[str] = ""):
    connected_components_index, G_MST = cluster_nodes(G, opposite_weight = opposite_weight, upper_quantile_removed = upper_quantile_removed)

    words = list(translator_word_to_index.keys())
    for nodes_index in connected_components_index:
        new_component_words = []
        for id in nodes_index:
            new_component_words.append(words[id])
        print("Component's words =", new_component_words)

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