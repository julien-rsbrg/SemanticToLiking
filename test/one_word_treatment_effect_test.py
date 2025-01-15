import os
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import torch
import torch_geometric
from torch_geometric.data import Data

from src.models.nn_layers import MLPModel
from src.models.gnn_layers import myGATConv
from src.models.frameworks import GNN_naive_framework
import src.loading as loading
import src.processing as processing

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


def run_one_word_treatment(sim_used:str = "original",
                           sham_node_value:float = -100,
                           dst_file_path:str|None = None):
    ## Data
    data = loading.load_data()

    def nor_function(a,b):
        return (a or b) and not(a and b)

    data["NoExp_Exp"] = data.apply(lambda row: nor_function(row["word1_experience"]>50,row["word2_experience"]>50),axis=1)


    results = {"participant":[],
               "node_i":[],
               "prediction_real_attr":[],
               "prediction_sham_attr":[],
               "label":[],
               "last_train_mae":[]}

    participant_indices = data["participant"].unique()
    for participant_id in participant_indices:
        print(f"start participant_id/n_participants-1:{participant_id:d}/{len(participant_indices-1):d}")
        participant_graph, translator_word_to_index = prepare_graph_for_participant(data=data, 
                                                                                    participant_id=participant_id, 
                                                                                    sim_used=sim_used)
        
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

        history = complete_model.train([participant_graph],100,1,opt,scheduler,"train_loss",200)

        for node_i in range(participant_graph.num_nodes):
            print(f" === node_i/num_nodes-1:{node_i:d}/{participant_graph.num_nodes-1:d}, participant_id/n_participants-1:{participant_id:d}/{len(participant_indices-1):d} === ")

            saved_node_value = copy.deepcopy(participant_graph.x[node_i])
            
            # sham attribute
            participant_graph.x[node_i] = sham_node_value
            print("participant_graph.x[node_i]",participant_graph.x[node_i])
            prediction_table = get_prediction_table(complete_model,participant_graph)
            node_prediction_sham_attr_table = prediction_table.iloc[node_i]
            participant_graph.x[node_i] = saved_node_value
            print("participant_graph.x[node_i]",participant_graph.x[node_i])

            # real attribute
            prediction_table = get_prediction_table(complete_model,participant_graph)
            node_prediction_real_attr_table = prediction_table.iloc[node_i]

            ## Save
            results["participant"].append(participant_id)
            results["node_i"].append(node_i)
            results["prediction_sham_attr"].append(node_prediction_sham_attr_table["prediction"])
            results["prediction_real_attr"].append(node_prediction_real_attr_table["prediction"])
            results["label"].append(node_prediction_real_attr_table["label"])
            results["last_train_mae"].append(float(history["train_mae"][-1]))
            
            
            print("-- Results --")
            print("participant_id =",results["participant"][-1])
            print("node_i =",results["node_i"][-1])
            print("prediction_sham_attr =",results["prediction_sham_attr"][-1])
            print("prediction_real_attr =",results["prediction_real_attr"][-1])
            print("label =",results["label"][-1])
            print("prediction_sham_attr - prediction_real_attr =",results["prediction_sham_attr"][-1] - results["prediction_real_attr"][-1])
            print("last_train_mae =",results["last_train_mae"][-1])
            print()

            if not(dst_file_path is None): 
                pd.DataFrame(results).to_csv(dst_file_path,index=False)
            
        print(f"end participant_id/n_participants-1:{participant_id:d}/{len(participant_indices-1):d}")

    return results

if __name__ == "__main__":
    run_one_word_treatment(sim_used="original",sham_node_value=10,dst_file_path="test/result_one_btreatment_effect_test_sham_10.csv")