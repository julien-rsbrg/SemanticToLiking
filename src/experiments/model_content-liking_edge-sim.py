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
import src.loading as loading
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


def run(sim_used:str = "original",dst_file_path:str|None = None):
    ## Data
    data = loading.load_data()

    def nor_function(a,b):
        return (a or b) and not(a and b)

    data["NoExp_Exp"] = data.apply(lambda row: nor_function(row["word1_experience"]>50,row["word2_experience"]>50),axis=1)


    results = {"participant":[],
               "min_train_loss":[],
               "min_train_mae":[]}

    participant_indices = data["participant"].unique()
    for participant_id in participant_indices:
        print(f"start participant_id/n_participants-1:{participant_id:d}/{len(participant_indices-1):d}")
        participant_graph, translator_word_to_index = prepare_graph_for_participant(data=data, 
                                                                                    participant_id=participant_id, 
                                                                                    sim_used=sim_used)
        
        src_content_mask = torch.Tensor([True]).to(torch.bool)
        src_edge_mask = torch.Tensor([False]).to(torch.bool)
        dst_mask = torch.Tensor([False]).to(torch.bool)
        my_module = myGATConv(
            in_channels=(1,1),
            out_channels=1,
            heads=1,
            negative_slope=0.0,
            add_self_loops=False,
            edge_dim=1,
            dropout=0.0,
            src_content_mask=src_content_mask,
            src_edge_mask=src_edge_mask,
            dst_content_mask=dst_mask,
            dst_edge_mask=dst_mask)

        ## Training
        complete_model = GNN_naive_framework(my_module,device)
        complete_model.predict(participant_graph.x,participant_graph.edge_index,participant_graph.edge_attr)
        description_parameters_init = complete_model.update_node_module.get_description_parameters().copy()

        opt = complete_model.configure_optimizer(lr=0.1)
        scheduler = complete_model.configure_scheduler(opt,0.1,0.1,10)

        history = complete_model.train([participant_graph],10000,1,opt,scheduler,"train_loss",200)
        complete_model.predict(participant_graph.x,participant_graph.edge_index,participant_graph.edge_attr)
        description_parameters_trained = complete_model.update_node_module.get_description_parameters().copy()

        ## Save
        results["participant"].append(participant_id)
        results["min_train_loss"].append(np.min(history["train_loss"]))
        results["min_train_mae"].append(np.min(history["train_mae"]))
        for param_name in description_parameters_init.columns:
            if "init_"+param_name in results:
                results["init_"+param_name].append(description_parameters_init[param_name][0])
                results["trained_"+param_name].append(description_parameters_trained[param_name][0])
            else:
                results["init_"+param_name] = [description_parameters_init[param_name][0]]
                results["trained_"+param_name] = [description_parameters_trained[param_name][0]]
        

        pd.DataFrame(results).to_csv(dst_file_path,index=False)
            
        print(f"end participant_id/n_participants-1:{participant_id:d}/{len(participant_indices-1):d}")

    return results

if __name__ == "__main__":
    run(sim_used="original",dst_file_path="src/experiments/results/model_content-liking_edge-sim_epochs-10000.csv")