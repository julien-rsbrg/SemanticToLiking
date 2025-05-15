import os
import copy
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import torch
import torch_geometric
from torch_geometric.data import Data

from src.models.nn_layers import MLPModel
from src.models.gnn_layers import MyGATConv
from src.models.ML_frameworks import GNNFramework
import src.loading as loading
import src.processing.raw_data_cleaning as raw_data_cleaning

from src.visualization.analyse_model import plot_errors_labels_comparison, get_prediction_table

from src.pipeline import GeneralizerRun 
import src.processing.preprocessing as preprocessing
import src.utils

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

    participant_graph,translator_word_to_index = raw_data_cleaning.convert_table_to_graph(
        complete_data_table=subdata,
        node_attr_names=["liking","experience"],
        node_label_names=["liking"],
        edge_attr_names=["test_sim"],
        return_word_to_index=True)

    # scaling only the liking
    node_table = raw_data_cleaning.build_node_table(data,["liking"],distinct_id=["participant"])
    scaler_liking = StandardScaler()
    scaler_liking.fit(node_table.liking.to_numpy()[...,np.newaxis])
    transformed_50 = scaler_liking.transform(np.array([[50]]))

    # todo create specific scaler function
    new_x_liking = scaler_liking.transform(participant_graph.x[:,:1])
    new_x_liking -= np.repeat(transformed_50,repeats=participant_graph.num_nodes,axis=0)
    participant_graph.x[:,:1] = torch.Tensor(new_x_liking)

    participant_graph.y = scaler_liking.transform(participant_graph.y)
    participant_graph.y -= np.repeat(transformed_50,repeats=participant_graph.num_nodes,axis=0)
    participant_graph.y = torch.Tensor(participant_graph.y)

    return participant_graph, translator_word_to_index





def run(sim_used:str = "original", dst_folder_path:str|None = None):
    dst_history_folder_path = os.path.join(dst_folder_path,"models_history")
    src.utils.recursive_mkdirs(dst_history_folder_path)

    ## Data
    data = loading.load_data()

    def nor_function(a,b):
        return (a or b) and not(a and b)


    results = {"participant":[],
               "min_train_loss":[],
               "min_train_mae":[],
               "min_train_loss_epoch":[],
               "min_train_mae_epoch":[],
               "end_epoch":[]}

    participant_indices = data["participant"].unique()
    for participant_id in participant_indices:
        time_start_participant = time.time()
        print(f"start participant_id/n_participants-1:{participant_id:d}/{len(participant_indices-1):d}")
        
        ## PREPROCESSING ##
        participant_graph, translator_word_to_index = prepare_graph_for_participant(data=data, 
                                                                                    participant_id=participant_id, 
                                                                                    sim_used=sim_used)
        
        # preprocessing 1 - experienced to not experienced
        print("Preprocessing 1")
        preprocessing_cut = preprocessing.CutGroupSendersToGroupReceivers(
            group_senders_mask_fn= lambda x: x["experience"] > 0,
            group_receivers_mask_fn= lambda x: x["experience"] <= 0,
        )
        # TODO: use data conversion tools/fns instead

        new_edge_index, new_edge_attr, new_x, new_y = preprocessing_cut.fit_transform(
            edge_index=participant_graph.edge_index.data.numpy(),
            edge_attr=pd.DataFrame(participant_graph.edge_attr.data.numpy(),columns=["test_sim"]),
            x=pd.DataFrame(participant_graph.x.data.numpy(),columns=["liking","experience"]),
            y=pd.DataFrame(participant_graph.y.data.numpy(),columns=["liking"])
        )

        participant_graph.edge_index = torch.Tensor(new_edge_index).to(dtype=torch.int64)
        participant_graph.edge_attr = torch.Tensor(new_edge_attr.values) 
        # participant_graph.edge_attr = None
        participant_graph.x = torch.Tensor(new_x.values) 
        participant_graph.y = torch.Tensor(new_y.values) 

        # preprocessing 2 - only not experienced predictions in training
        print("Preprocessing 2")
        node_train_mask = torch.ones(len(participant_graph.x),dtype=torch.bool)
        node_train_mask[participant_graph.x[:,1]>0] = False
        participant_graph.train_mask = node_train_mask
        participant_graph.val_mask = torch.zeros(len(participant_graph.x),dtype=torch.bool)
        
        # preprocessing 3 - two distinct parameters for liking positive and liking negative
        #print("Preprocessing 3")
        # preprocessing_separate_features = preprocessing.SeparatePositiveNegative(verbose=True, feature_separated="liking")
        #x, _ = preprocessing_separate_features.fit_transform(
        #    x = pd.DataFrame(participant_graph.x.data.numpy(),columns=["liking","experience"])
        #)

        # preprocessing 3' - remove experience from liking 
        participant_graph.x = participant_graph.x[:,:1]
               
        ## MODEL ##
        print("Model")
        src_content_mask = torch.Tensor([True]).to(torch.bool)
        src_edge_mask = torch.Tensor([False]).to(torch.bool)
        dst_content_mask = torch.Tensor([False]).to(torch.bool)
        dst_edge_mask = torch.Tensor([False]).to(torch.bool)
        my_module = MyGATConv(
            in_channels=(1,1),
            out_channels=1,
            heads=1,
            negative_slope=0.0,
            add_self_loops=False,
            edge_dim=1,
            dropout=0.0,
            bias=False,
            src_content_mask=src_content_mask,
            src_edge_mask=src_edge_mask,
            dst_content_mask=dst_content_mask,
            dst_edge_mask=dst_edge_mask,
            src_content_require_grad=False,
            src_content_weight_initializer="ones",
            edge_weight_initializer="ones")

        
        
        ## Training
        complete_model = GNNFramework(my_module,device)
        complete_model.predict(participant_graph.x,
                               participant_graph.edge_index,
                               participant_graph.edge_attr)

        opt = complete_model.configure_optimizer(lr=10)
        scheduler = complete_model.configure_scheduler(opt,0.1,0.1,10)
        weight_constrainer = complete_model.configure_weight_constrainer("clipper",0,100)
        
        description_parameters_init = complete_model.update_node_module.get_description_parameters().copy()

        if participant_graph.train_mask.sum()>0:
            history = complete_model.train([participant_graph],
                                        10000,
                                        1,
                                        opt,
                                        scheduler,
                                        weight_constrainer=weight_constrainer,
                                        val_dataset=None,
                                        early_stopping_monitor="train_loss",
                                        patience=200,
                                        l2_reg=1e-2)
        else:
            history = {"epoch":[],"train_loss":[],"train_mae":[]}

        dst_history_path = os.path.join(dst_history_folder_path, f"model_history_participant_{participant_id}.csv")
        pd.DataFrame(history).to_csv(dst_history_path,index=False)

        complete_model.predict(participant_graph.x,participant_graph.edge_index,participant_graph.edge_attr)
        description_parameters_trained = complete_model.update_node_module.get_description_parameters().copy()

        ## Save
        results["participant"].append(participant_id)
        results["min_train_loss"].append(np.min(history["train_loss"]) if len(history["train_loss"]) else None)
        results["min_train_mae"].append(np.min(history["train_mae"]) if len(history["train_mae"]) else None)
        results["min_train_loss_epoch"].append(np.argmin(history["train_loss"]) if len(history["train_loss"]) else None)
        results["min_train_mae_epoch"].append(np.argmin(history["train_mae"]) if len(history["train_mae"]) else None)
        results["end_epoch"].append(history["epoch"][-1] if len(history["epoch"]) else None)

        for param_name in description_parameters_init.columns:
            if "init_"+param_name in results:
                results["init_"+param_name].append(description_parameters_init[param_name][0])
                results["trained_"+param_name].append(description_parameters_trained[param_name][0])
            else:
                results["init_"+param_name] = [description_parameters_init[param_name][0]]
                results["trained_"+param_name] = [description_parameters_trained[param_name][0]]


        dst_results_path = os.path.join(dst_folder_path, "results_model_pipeline_content-identity_edge-sim_epochs-10000.csv")
        pd.DataFrame(results).to_csv(dst_results_path,index=False)
            
        print(f"end participant_id/n_participants-1:{participant_id:d}/{len(participant_indices-1):d}")
        time_take_participant = (time.time()-time_start_participant)
        print("time taken: {}h - {}m - {}s".format(time_take_participant//(3600),((time_take_participant%3600)//60),((time_take_participant%3600)%60)))

    return results





if __name__ == "__main__":
    run(sim_used="original",dst_folder_path="src/experiments/results/2025-05-13_19-38_results")