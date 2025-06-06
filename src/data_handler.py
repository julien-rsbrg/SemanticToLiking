import os
import numpy as np
import pandas as pd

from typing import Iterable

from src.processing.raw_data_cleaning import build_node_table
from src.utils import recursive_mkdirs, read_yaml, save_yaml, flatten_dict, convert_dataframe_to_dict, turn_dict_values_tensor_to_list

def load_data():
    print("== Load Data: start ==")
    data = pd.read_excel("data/paired_data_newSim.xlsx")
    _temp = data["word_pair"].str.split(".")
    data["word1"] = _temp.apply(func=lambda x: x[0])
    data["word2"] = _temp.apply(func=lambda x: x[1][1:])

    print("== Load Data: end ==")
    return data

def get_participant_data(raw_data:pd.DataFrame) -> pd.DataFrame:
    participant_features = ["participant","depression","depressionCont","female","age"]
    participant_data = raw_data[participant_features].drop_duplicates(inplace=False).reset_index(drop=True)

    node_data_table = build_node_table(raw_data,["liking","experience"],["participant"])
    node_data_table["experienced"] = node_data_table["experience"] > 0
    node_data_table["one"] = 1 
    participant_n_not_experienced = node_data_table.groupby("participant")["one"].sum() - node_data_table.groupby("participant")["experienced"].sum()
    participant_n_not_experienced = participant_n_not_experienced.rename("n_not_experienced")

    participant_data = pd.merge(participant_data,participant_n_not_experienced,on="participant")
    return participant_data


def postprocess(src_folder_path,dst_folder_path):
    """
    Assumes a given structure for the raw data's folder structure 
    """

    # aggregate participant-wise
    for participant_folder_name in os.listdir(src_folder_path):
        participant_folder_path = os.path.join(src_folder_path,participant_folder_name)

        models_params_init = []
        models_params_trained = [] 
        summary = {}
        for graph_folder_name in os.listdir(participant_folder_path):
            graph_folder_path = os.path.join(participant_folder_path,graph_folder_name)
            if os.path.isdir(graph_folder_path):
                pred_table = pd.read_csv(os.path.join(graph_folder_path,"prediction_table.csv"),index_col=0)
                train_diff = pred_table[pred_table["train_mask"]]["pred_values"] - pred_table[pred_table["train_mask"]]["true_values"]
                val_diff = pred_table[pred_table["val_mask"]]["pred_values"] - pred_table[pred_table["val_mask"]]["true_values"]
                summary["train_avg_pred-true"] = [np.mean(train_diff)]
                summary["train_avg_MAE"] = [np.mean(np.abs(train_diff))]

                summary["val_avg_pred-true"] = [np.mean(val_diff)]
                summary["val_avg_MAE"] = [np.mean(np.abs(val_diff))]

                models_params_init.append(pd.read_csv(os.path.join(graph_folder_path,"model_params_init.csv"),index_col=0))
                models_params_trained.append(pd.read_csv(os.path.join(graph_folder_path,"model_params_trained.csv"),index_col=0))
        
        summary = pd.DataFrame(summary)

        model_params = []
        if len(models_params_init):
            models_params_init = pd.concat(models_params_init,axis=0)
            models_params_init_mean = models_params_init.mean().rename(index=lambda name: name + "_mean_init")
            models_params_init_mean = pd.DataFrame(models_params_init.mean()).T
            models_params_init_std = models_params_init.std().rename(index=lambda name: name + "_std_init")
            models_params_init_std = pd.DataFrame(models_params_init.std()).T

            models_params_init = pd.concat([models_params_init_mean,models_params_init_std],axis=1)
            model_params.append(models_params_init)
        
        if len(models_params_trained):
            models_params_trained = pd.concat(models_params_trained,axis=0)
            models_params_trained_mean = models_params_trained.mean().rename(index=lambda name: name + "_mean_trained")
            models_params_trained_mean = pd.DataFrame(models_params_trained_mean).T
            models_params_trained_std = models_params_trained.std().rename(index=lambda name: name + "_std_trained")
            models_params_trained_std = pd.DataFrame(models_params_trained_std).T
            models_params_trained = pd.concat([models_params_trained_mean,models_params_trained_std],axis=1)
            model_params.append(models_params_trained)
        
        if len(model_params):
            model_params = pd.concat(model_params,axis=1)
            summary = pd.concat([summary,model_params],axis=1)
        
        recursive_mkdirs(os.path.join(dst_folder_path,participant_folder_name))
        summary.to_csv(os.path.join(dst_folder_path, participant_folder_name, "summary.csv"))
    

    # aggregate across participants
    overall_summaries = []
    config_params = []
    for participant_folder_name in os.listdir(src_folder_path):
        config = read_yaml(os.path.join(src_folder_path,participant_folder_name,"config.yml"))
        config = pd.DataFrame(flatten_dict(config),index=[0])
        config_params = config.columns.tolist()
        config["participant_folder_name"] = participant_folder_name

        summary = pd.read_csv(os.path.join(dst_folder_path,participant_folder_name,"summary.csv"), index_col=0)
        
        overall_summary = pd.concat([config,summary],axis=1)

        overall_summaries.append(overall_summary)
    

    overall_summaries = pd.concat(overall_summaries,axis=0)
    overall_summaries.reset_index(inplace=True,drop=True)
    overall_summaries.to_csv(os.path.join(dst_folder_path,"overall_summaries.csv"))

    # save externally the constant config params
    constant_params = []
    for param_name in config_params:
        if len(np.unique(overall_summaries[param_name].dropna(axis=0))) == 1:
            constant_params.append(param_name)
    constant_config = pd.DataFrame(overall_summaries)[constant_params].iloc[0].to_dict()
    varying_params = list(set(config_params) - set(constant_params))
    constant_config["varying_params"] = varying_params
    save_yaml(constant_config,os.path.join(dst_folder_path,"constant_config.yml"))
    

if __name__ == "__main__":
    postprocess("src/data_generation/examples/raw/study_0","src/data_generation/examples/processed/study_0")
    postprocess("src/data_generation/examples/raw/study_1","src/data_generation/examples/processed/study_1")
    postprocess("src/data_generation/examples/raw/study_2","src/data_generation/examples/processed/study_2")