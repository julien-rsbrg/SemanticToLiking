import pandas as pd
import numpy as np
import torch

import copy
from torch_geometric.data import Data
from typing import Iterable

import seaborn as sns
import plotly.graph_objects as go

import statsmodels.api as sm
import statsmodels.formula.api as smf

import src.data_handler as data_handler
from src.processing.raw_data_cleaning import prepare_graph_for_participant

from sentence_transformers import SentenceTransformer

def add_edge_L2_dist_to_graph(graph:Data, translator_word_to_index, word_to_embeddings):
    translator_index_to_word_emb = {v:np.array(word_to_embeddings[k]) for k,v in translator_word_to_index.items()} # will have to add .values when loading 
    
    edge_L2_dist = []
    for edge_i in range(graph.edge_index.size(1)):
        sender_id,receiver_id = graph.edge_index[:,edge_i]
        sender_id,receiver_id = int(sender_id), int(receiver_id)
        L2_dist = np.linalg.norm(translator_index_to_word_emb[receiver_id] - translator_index_to_word_emb[sender_id])
        edge_L2_dist.append([L2_dist])

    graph.edge_L2_dist = torch.Tensor(edge_L2_dist)
    return graph




def shortest_path_djikstra(node_start:int,edge_index:np.array,edge_weight:Iterable,num_nodes:int|None = None):
    assert node_start <= max(np.max(edge_index),num_nodes), f"There is no edge connected to node_start or the number of nodes ({num_nodes}) given is wrong..."
    
    _num_nodes = num_nodes
    if num_nodes is None:
        _num_nodes = np.max(edge_index)+1
        
    distance_from_start = np.inf * np.ones(_num_nodes)
    distance_from_start[node_start] = 0

    shortest_path_from_start = {node_id:{"edge_index":[],"edge_weight":[]} for node_id in range(_num_nodes)}

    segments_to_check = [(0,node_start)]
    while len(segments_to_check):
        #print("len(segments_to_check):",len(segments_to_check))
        selected_segment = segments_to_check.pop(-1)
        sending_edges_mask = edge_index[0,:] == selected_segment[1]

        receiving_neighbor_nodes = edge_index[1,sending_edges_mask] 
        neighbor_distances = edge_weight[sending_edges_mask]

        for edge_local_id,node_id in enumerate(receiving_neighbor_nodes):
            current_path_distance = selected_segment[0] + neighbor_distances[edge_local_id]
            if distance_from_start[node_id] > current_path_distance:
                distance_from_start[node_id] =  current_path_distance
                segments_to_check.append((distance_from_start[node_id],node_id))
                shortest_path_from_start[node_id]["edge_index"] = shortest_path_from_start[selected_segment[1]]["edge_index"] + [[selected_segment[1],node_id]] 
                shortest_path_from_start[node_id]["edge_weight"] = shortest_path_from_start[selected_segment[1]]["edge_weight"] + [neighbor_distances[edge_local_id]] 
    return distance_from_start, shortest_path_from_start



def shortest_path_to_cluster_djikstra(node_start:int,
                                      edge_index:np.array,
                                      edge_weight:Iterable,
                                      cluster:Iterable[bool],
                                      num_nodes:int|None = None,
                                      exclude_itself:bool=True,
                                      distances_from_start:np.ndarray|None = None,
                                      shortest_path_from_start:dict|None = None):
    """
    shortest path from node_start to a node belonging to cluster (set to True).

    Parameters
    ----------
    - exclude_itself: (bool)
        If True and node_start in cluster, removes node_start from the cluster identification and computes the distance to the nodes of the same cluster
    """
    _cluster = copy.copy(cluster)
    if exclude_itself:
        _cluster[node_start] = False

    node_ids_cluster = np.where(_cluster)[0]
    
    if node_start in node_ids_cluster:
        return 0, {node_start: {'edge_index': [], 'edge_weight': []}}
    elif len(node_ids_cluster) == 0:
        return np.nan, dict() 
    else:
        if (distances_from_start is None) or (shortest_path_from_start is None):
            _distances_from_start, _shortest_path_from_start = shortest_path_djikstra(node_start,edge_index,edge_weight,num_nodes=num_nodes)
        else:
            _distances_from_start, _shortest_path_from_start = distances_from_start, shortest_path_from_start 
        closest_node_id_cluster = node_ids_cluster[np.argmin(_distances_from_start[node_ids_cluster])]

        shortest_path_to_cluster = {closest_node_id_cluster: _shortest_path_from_start[closest_node_id_cluster]}
        return _distances_from_start[closest_node_id_cluster],shortest_path_to_cluster




def mean_shortest_distance_to_cluster_djikstra(node_start:int,edge_index:np.array,edge_weight:Iterable,cluster:Iterable[bool],num_nodes:int|None = None,exclude_itself:bool=True):
    """
    mean shortest distance from node_start to a node belonging to cluster (set to True).
    """
    _cluster = copy.copy(cluster)
    if exclude_itself:
        _cluster[node_start] = False
    node_ids_cluster = np.where(_cluster)[0]

    distances_from_start, _ = shortest_path_djikstra(node_start,edge_index,edge_weight,num_nodes=num_nodes)
    return np.mean(distances_from_start[node_ids_cluster])




def mean_leaps_shortest_path_to_cluster_djikstra(node_start:int,edge_index:np.array,edge_weight:Iterable,cluster:Iterable[bool],num_nodes:int|None = None,exclude_itself:bool=True):
    """
    mean shortest distance from node_start to a node belonging to cluster (set to True).
    """
    _cluster = copy.copy(cluster)
    if exclude_itself:
        _cluster[node_start] = False
    node_ids_cluster = np.where(_cluster)[0] 

    _, shortest_paths = shortest_path_djikstra(node_start,edge_index,edge_weight,num_nodes=num_nodes)

    leaps = []
    for path in shortest_paths.values():
        leaps.append(len(path["edge_index"]))
        
    return np.mean(np.array(leaps)[node_ids_cluster])



if __name__ == "__main__":

    data = data_handler.load_data()
    word_to_embeddings = pd.read_csv('data/processed/node_data/word_to_embeddings_MPNet.csv', index_col=0)


    all_node_data = None

    participant_ids = data["participant"].unique()
    n_participants = len(participant_ids)
    for participant_id in participant_ids:
        print(f"participant / n_participant: {participant_id} / {n_participants}")
        participant_graph, translator_word_to_index = prepare_graph_for_participant(data=data, 
                                                            participant_id=participant_id, 
                                                            sim_used="original")
        x = pd.DataFrame(participant_graph.x,columns=participant_graph.x_names)
        cluster_has_been_experienced = (x["experience"] > 0).values
        edge_index = participant_graph.edge_index.numpy() # check to remove
        participant_graph = add_edge_L2_dist_to_graph(participant_graph=participant_graph,translator_word_to_index=translator_word_to_index)
        edge_weight = participant_graph.edge_L2_dist.numpy()

        node_data = {"node_id":[],"word":[], "experienced": [], "min_to_exp":[],"min_to_not_exp":[],"mean_to_exp":[],"mean_to_not_exp":[],"n_leaps_min_to_exp":[],"n_leaps_min_to_not_exp":[],"mean_leaps_to_exp":[],"mean_leaps_to_not_exp":[]}
        for node_id in range(participant_graph.num_nodes):
            print(f"participant / n_participant: {participant_id} / {n_participants} :: node_id + 1 / num_nodes: {node_id+1} / {participant_graph.num_nodes}\n")
            # Careful: suboptimal, compute twice the shortest distance
            min_to_exp,shortest_path_to_exp = shortest_path_to_cluster_djikstra(node_id,edge_index,edge_weight,cluster_has_been_experienced)
            if len(shortest_path_to_exp) == 0:
                n_leaps_min_to_exp = np.nan
                mean_to_exp = np.nan
                mean_leaps_to_exp = np.nan
            else:
                n_leaps_min_to_exp = len(next(iter(shortest_path_to_exp.values()))["edge_index"])
                mean_to_exp = mean_shortest_distance_to_cluster_djikstra(node_id,edge_index,edge_weight,cluster_has_been_experienced)
                mean_leaps_to_exp = mean_leaps_shortest_path_to_cluster_djikstra(node_id,edge_index,edge_weight,cluster_has_been_experienced)

            min_to_not_exp,shortest_path_to_not_exp = shortest_path_to_cluster_djikstra(node_id,edge_index,edge_weight,~cluster_has_been_experienced)
            if len(shortest_path_to_not_exp) == 0:
                n_leaps_min_to_not_exp = np.nan
                mean_to_not_exp = np.nan
                mean_leaps_to_not_exp = np.nan
            else:
                n_leaps_min_to_not_exp = len(next(iter(shortest_path_to_not_exp.values()))["edge_index"])
                mean_to_not_exp = mean_shortest_distance_to_cluster_djikstra(node_id,edge_index,edge_weight,~cluster_has_been_experienced)
                mean_leaps_to_not_exp = mean_leaps_shortest_path_to_cluster_djikstra(node_id,edge_index,edge_weight,~cluster_has_been_experienced)

            node_data["node_id"].append(node_id)
            node_data["word"].append(list(translator_word_to_index.keys())[node_id])
            node_data["experienced"].append(cluster_has_been_experienced[node_id])

            node_data["min_to_exp"].append(min_to_exp)
            node_data["n_leaps_min_to_exp"].append(n_leaps_min_to_exp)
            node_data["mean_to_exp"].append(mean_to_exp)
            node_data["mean_leaps_to_exp"].append(mean_leaps_to_exp)

            node_data["min_to_not_exp"].append(min_to_not_exp)
            node_data["n_leaps_min_to_not_exp"].append(n_leaps_min_to_not_exp)
            node_data["mean_to_not_exp"].append(mean_to_not_exp)
            node_data["mean_leaps_to_not_exp"].append(mean_leaps_to_not_exp)

        node_data = pd.DataFrame(node_data)
        node_data["participant_id"] = participant_id
        
        if all_node_data is None:
            all_node_data = node_data
        else:
            all_node_data = pd.concat([all_node_data,node_data],axis=0)
            all_node_data.reset_index(inplace=True,drop=True)


    all_node_data.to_csv(f"data/processed/node_data/distance_cluster.csv")

