import os
import numpy as np
import pandas as pd

from typing import Iterable

from src.processing.raw_data_cleaning import build_node_table
from src.utils import recursive_mkdirs, read_yaml, save_yaml, flatten_dict, convert_dataframe_to_dict, turn_dict_values_tensor_to_list, check_model_normality, compute_log_likelihood_normal, compute_BIC, compute_AIC, compute_AICc

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
        # print("participant_folder_name:",participant_folder_name)
        participant_folder_path = os.path.join(src_folder_path,participant_folder_name)

        n_params = read_yaml(os.path.join(participant_folder_path,"config.yml"))["model"]["n_free_params"]

        models_params_init = []
        models_params_trained = [] 
        summary = {
            "graph":[],
            "train_mean_pred-true":[],
            "train_MAE":[],
            "train_MSE":[],
            "train_log_likelihood":[],
            "train_residuals_shapiro_statistic":[],
            "train_residuals_shapiro_pvalue":[],
            "val_mean_pred-true":[],
            "val_MAE":[],
            "val_MSE":[],
            "val_log_likelihood":[],
            "val_residuals_shapiro_statistic":[],
            "val_residuals_shapiro_pvalue":[],
            }
        for graph_folder_name in os.listdir(participant_folder_path):
            # print("graph_folder_name:",graph_folder_name)
            graph_folder_path = os.path.join(participant_folder_path,graph_folder_name)
            if os.path.isdir(graph_folder_path):
                summary["graph"].append(graph_folder_name)

                pred_table = pd.read_csv(os.path.join(graph_folder_path,"prediction_table.csv"),index_col=0)
                
                # regular metrics
                train_diff = pred_table[pred_table["train_mask"]]["pred_values"] - pred_table[pred_table["train_mask"]]["true_values"]
                val_diff = pred_table[pred_table["val_mask"]]["pred_values"] - pred_table[pred_table["val_mask"]]["true_values"]
                
                summary["train_mean_pred-true"].append(np.mean(train_diff))
                summary["train_MSE"].append(np.mean(np.power(train_diff, 2)))
                summary["train_MAE"].append(np.mean(np.abs(train_diff)))
                summary["train_log_likelihood"].append(compute_log_likelihood_normal(y_pred=pred_table[pred_table["train_mask"]]["pred_values"].values[...,np.newaxis],
                                                                                     y_true=pred_table[pred_table["train_mask"]]["true_values"].values[...,np.newaxis])[0])

                summary["val_mean_pred-true"].append(np.mean(val_diff))
                summary["val_MSE"].append(np.mean(np.power(val_diff,2)))
                summary["val_MAE"].append(np.mean(np.abs(val_diff)))
                summary["val_log_likelihood"].append(compute_log_likelihood_normal(y_pred=pred_table[pred_table["val_mask"]]["pred_values"].values[...,np.newaxis],
                                                                                   y_true=pred_table[pred_table["val_mask"]]["true_values"].values[...,np.newaxis])[0])

                # normality test
                train_results_residuals_shapiro = check_model_normality(y_pred=pred_table[pred_table["train_mask"]]["pred_values"].values,
                                                                        y_true=pred_table[pred_table["train_mask"]]["true_values"].values,
                                                                        dst_folder_path=graph_folder_path)
                summary["train_residuals_shapiro_statistic"].append(train_results_residuals_shapiro["residuals_shapiro_statistic"][0])
                summary["train_residuals_shapiro_pvalue"].append(train_results_residuals_shapiro["residuals_shapiro_pvalue"][0])
                
                val_results_residuals_shapiro = check_model_normality(y_pred=pred_table[pred_table["val_mask"]]["pred_values"].values,
                                                                      y_true=pred_table[pred_table["val_mask"]]["true_values"].values,
                                                                      dst_folder_path=graph_folder_path)
                summary["val_residuals_shapiro_statistic"].append(val_results_residuals_shapiro["residuals_shapiro_statistic"][0])
                summary["val_residuals_shapiro_pvalue"].append(val_results_residuals_shapiro["residuals_shapiro_pvalue"][0])

                # information criteria
                train_BIC = compute_BIC(y_pred = pred_table[pred_table["train_mask"]]["pred_values"].values[...,np.newaxis],
                                        y_true = pred_table[pred_table["train_mask"]]["true_values"].values[...,np.newaxis],
                                        n_params = n_params)

                val_BIC = compute_BIC(y_pred = pred_table[pred_table["val_mask"]]["pred_values"].values[...,np.newaxis],
                                      y_true = pred_table[pred_table["val_mask"]]["true_values"].values[...,np.newaxis],
                                      n_params = n_params)
                
                train_AIC = compute_AIC(y_pred = pred_table[pred_table["train_mask"]]["pred_values"].values[...,np.newaxis],
                                        y_true = pred_table[pred_table["train_mask"]]["true_values"].values[...,np.newaxis],
                                        n_params = n_params)

                val_AIC = compute_AIC(y_pred = pred_table[pred_table["val_mask"]]["pred_values"].values[...,np.newaxis],
                                      y_true = pred_table[pred_table["val_mask"]]["true_values"].values[...,np.newaxis],
                                      n_params = n_params)
                
                train_AICc = compute_AICc(y_pred = pred_table[pred_table["train_mask"]]["pred_values"].values[...,np.newaxis],
                                        y_true = pred_table[pred_table["train_mask"]]["true_values"].values[...,np.newaxis],
                                        n_params = n_params)

                val_AICc = compute_AICc(y_pred = pred_table[pred_table["val_mask"]]["pred_values"].values[...,np.newaxis],
                                      y_true = pred_table[pred_table["val_mask"]]["true_values"].values[...,np.newaxis],
                                      n_params = n_params)


                # bof, you know that the output is 1d otherwise val_MSE_0,val_MSE_1... too
                for i in range(len(train_BIC)):
                    if not(f"train_BIC_{i}" in summary):
                        summary[f"train_BIC_{i}"] = []
                        summary[f"val_BIC_{i}"] = []

                        summary[f"train_AIC_{i}"] = []
                        summary[f"val_AIC_{i}"] = []

                        summary[f"train_AICc_{i}"] = []
                        summary[f"val_AICc_{i}"] = []
                    
                    summary[f"train_BIC_{i}"].append(train_BIC[i])
                    summary[f"val_BIC_{i}"].append(val_BIC[i])

                    summary[f"train_AIC_{i}"].append(train_AIC[i])
                    summary[f"val_AIC_{i}"].append(val_AIC[i])

                    summary[f"train_AICc_{i}"].append(train_AICc[i])
                    summary[f"val_AICc_{i}"].append(val_AICc[i])
                
                models_params_init.append(pd.read_csv(os.path.join(graph_folder_path,"model_params_init.csv"),index_col=0))
                models_params_trained.append(pd.read_csv(os.path.join(graph_folder_path,"model_params_trained.csv"),index_col=0))

        summary = pd.DataFrame(summary)

        model_params = []
        if len(models_params_init):
            models_params_init = pd.concat(models_params_init,axis=0)
            _models_params_init = models_params_init.copy().rename(columns=lambda name: name + "_init")
            model_params.append(_models_params_init)
        
        if len(models_params_trained):
            models_params_trained = pd.concat(models_params_trained,axis=0)
            _models_params_trained = models_params_trained.copy().rename(columns=lambda name: name + "_trained")
            model_params.append(_models_params_trained)
        
        if len(model_params):
            model_params = pd.concat(model_params, axis = 1).reset_index(drop=True)
            model_params = model_params.loc[:,~model_params.columns.duplicated()].copy()

            summary = pd.concat([summary,model_params], axis = 1)

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
    overall_summaries = overall_summaries.ffill()
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
    print("MAIN RUN data_handler.py")
    study_name = "2025-06-27_13-02__GAT_liking_sim_amp_3NN_3ExpNN_no_val_bias-False_att-liking-False_amp-liking-True_sim-Zeros"
    print(study_name)
    for participant_folder_name in os.listdir(f"experiments_results/no_validation/{study_name}/raw"):
        config_path = os.path.join(f"experiments_results/no_validation/{study_name}/raw",participant_folder_name,"config.yml")
        config = read_yaml(config_path)
        config["model"]["n_free_params"] = 0 + 0 + 1 + 0 # bias, X att liking, amp liking, edge 
        save_yaml(config,config_path)

    postprocess(f"experiments_results/no_validation/{study_name}/raw",f"experiments_results/no_validation/{study_name}/processed")
    # postprocess("src/data_generation/examples/raw/study_1","src/data_generation/examples/processed/study_1")
    # postprocess("src/data_generation/examples/raw/study_2","src/data_generation/examples/processed/study_2")