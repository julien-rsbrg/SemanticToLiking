import os
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import torch
import torch_geometric
from torch_geometric.data import Data

from src.models.nn.nn_layers import MLPModel
from src.models.nn.gnn_layers import myGATConv
from src.models.nn.ML_frameworks import GNN_naive_framework
import src.data_handler as data_handler
import src.preprocessing.processing as processing

from src.visualization.analyse_model import plot_errors_labels_comparison, get_prediction_table

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Graph
def prepare_graph_for_participant(data:pd.DataFrame, participant_id:any, sim_used:str) -> Data:
    # extraction
    subdata = data[data["participant"] == participant_id]

    # similarity
    if sim_used == "ones":
        subdata["test_sim"] = np.ones(len(subdata))
    elif sim_used == "random":
        subdata["test_sim"] = np.random.rand(len(subdata))
    elif sim_used == "original":
        subdata["test_sim"] = subdata["senenceBERT_mpnet_similarity"]
    else:
        raise NotImplementedError("Don't know sim_used =", sim_used)

    participant_graph,translator_word_to_index = processing.convert_table_to_graph(
        complete_data_table=subdata,
        node_attr_names=["liking"],
        node_label_names=["liking"],
        edge_attr_names=["test_sim"],
        return_word_to_index=True)

    # scaling only the liking
    node_table = processing.build_node_table(data,["liking"],distinct_id=["participant"])
    scaler_liking = StandardScaler()
    scaler_liking.fit(node_table.liking.to_numpy()[...,np.newaxis])
    transformed_50 = scaler_liking.transform(np.array([[50]]))

    # todo create specific scaler function
    participant_graph.x = scaler_liking.transform(participant_graph.x)
    participant_graph.x -= np.repeat(transformed_50,repeats=participant_graph.num_nodes,axis=0)
    participant_graph.x = torch.Tensor(participant_graph.x)

    participant_graph.y = scaler_liking.transform(participant_graph.y)
    participant_graph.y -= np.repeat(transformed_50,repeats=participant_graph.num_nodes,axis=0)
    participant_graph.y = torch.Tensor(participant_graph.y)

    return participant_graph, translator_word_to_index


def remove_node(graph:torch_geometric.data.Data,node_id:int):
    kept_nodes = torch.ones(graph.num_nodes,dtype=bool)
    kept_nodes[node_id] = False
    _graph = torch_geometric.data.Data()
    _graph.x = graph.x[kept_nodes]
    _graph.y = graph.y[kept_nodes]
    _graph.train_mask = torch.ones(_graph.num_nodes,dtype=bool)
    _graph.val_mask = torch.zeros(_graph.num_nodes,dtype=bool)

    kept_edges = (graph.edge_index == node_id).sum(0) == 0
    _graph.edge_index = graph.edge_index[:,kept_edges]
    _graph.edge_index = torch.where(_graph.edge_index > node_id,_graph.edge_index - 1, _graph.edge_index)
    _graph.edge_attr = graph.edge_attr[kept_edges,:]

    return _graph


def run_hold_one_out_test(sim_used:str = "original",
                          sham_node_value:float = -100,
                          dst_file_path:str|None = None):
    ## Data
    data = data_handler.load_data()

    def nor_function(a,b):
        return (a or b) and not(a and b)

    data["NoExp_Exp"] = data.apply(lambda row: nor_function(row["word1_experience"]>50,row["word2_experience"]>50),axis=1)


    results = {"participant":[],
               "node_i":[],
               "prediction":[],
               "label":[],
               "last_train_mae":[],
               "mean_residual":[],
               "std_residual":[],
               "min_residual":[],
               "max_residual":[]}

    for participant_id in data["participant"].unique():
        print(f"start participant_id:{participant_id:d}")
        participant_graph, translator_word_to_index = prepare_graph_for_participant(data=data, 
                                                                                    participant_id=participant_id, 
                                                                                    sim_used=sim_used)
        for node_i in range(participant_graph.num_nodes):
            print(f" === node_i/num_nodes-1:{node_i:d}/{participant_graph.num_nodes-1:d}, participant_id:{participant_id:d} === ")
            participant_subgraph = remove_node(participant_graph,node_id=node_i)
            print("participant_subgraph:",participant_subgraph)
            ## Model

            src_mask = torch.Tensor([True]).to(torch.bool)
            dst_mask = torch.Tensor([False]).to(torch.bool)
            my_module = myGATConv(
                in_channels=(1,1),
                out_channels=1,
                heads=1,
                negative_slope=0.0,
                add_self_loops=False,
                edge_dim=1,
                dropout=0.0,
                src_mask=src_mask,
                dst_mask=dst_mask)

            ## Training
            complete_model = GNN_naive_framework(my_module,device)

            opt = complete_model.configure_optimizer(lr=0.1)
            scheduler = complete_model.configure_scheduler(opt,0.1,0.1,10)

            history = complete_model.train([participant_subgraph],10000,1,opt,scheduler,"train_loss",200)

            ## Results
            saved_node_value = copy.deepcopy(participant_graph.x[node_i])
            participant_graph.x[node_i] = sham_node_value
            print("participant_graph.x[node_i]",participant_graph.x[node_i])
            prediction_table = get_prediction_table(complete_model,participant_graph)
            node_prediction_table = prediction_table.iloc[node_i]
            participant_graph.x[node_i] = saved_node_value
            print("participant_graph.x[node_i]",participant_graph.x[node_i])
            residual_table = prediction_table["label"] - prediction_table["prediction"]
            residual_table = pd.concat([residual_table.iloc[:node_i],residual_table.iloc[node_i+1:]],axis=0)
            print(residual_table)
            mean_residual = residual_table.mean()
            std_residual = residual_table.std()
            max_residual = residual_table.max()
            min_residual = residual_table.min()

            ## Save
            results["participant"].append(participant_id)
            results["node_i"].append(node_i)
            results["prediction"].append(node_prediction_table["prediction"])
            results["label"].append(node_prediction_table["label"])
            results["last_train_mae"].append(float(history["train_mae"][-1]))
            results["mean_residual"].append(mean_residual)
            results["std_residual"].append(std_residual)
            results["min_residual"].append(min_residual)
            results["max_residual"].append(max_residual)
            
            
            print("-- Results --")
            print("participant_id =",results["participant"])
            print("node_i =",results["node_i"])
            print("prediction =",results["prediction"])
            print("label =",results["label"])
            print("label - prediction =",results["label"][0] - results["prediction"][0])
            print("last_train_mae =",results["last_train_mae"])
            print("mean_residual =",results["mean_residual"])
            print("min_residual =",results["min_residual"])
            print("max_residual =",results["max_residual"])
            print()

            if not(dst_file_path is None): 
                pd.DataFrame(results).to_csv(dst_file_path,index=False)
            return
        
        print(f"end participant_id:{participant_id:d}")

    return results

if __name__ == "__main__":
    run_hold_one_out_test(sim_used="original",sham_node_value=-1e1,dst_file_path="test/result_hold_one_out.csv")