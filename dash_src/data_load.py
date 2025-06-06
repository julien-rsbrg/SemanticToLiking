"""
Handle all data loading
"""

import os
import numpy as np
import pandas as pd

import dash_src.utils as utils

from dash_src.configs.main import config
from src.utils import read_yaml
import src.data_handler as src_data_handler

def load_configs(result_folder_path:str)->pd.DataFrame:
    """
    Load the configs in <result_folder_path>/config

    Arguments
    ---------
    - result_folder_path: (str)
        Path to the result folder
    
    Returns
    -------
    - df_config: (pd.DataFrame)
        Contains the mean and standard deviation in each item in a string format for representation
    """
    all_configs = []
    for exp_name in os.listdir(os.path.join(result_folder_path,"config")):
        src_path = os.path.join(result_folder_path,"config",exp_name,"processed","df_config.csv")
        new_config = pd.read_csv(src_path,index_col=0)[["exp_name"]+config["main_config_params"]]
        new_config[new_config.select_dtypes(bool).columns] = new_config.select_dtypes(bool).astype(np.int32)

        all_configs.append(new_config)
    
    return pd.concat(all_configs,axis=0)



def load_data(exp_path:str,participant_data_path:str):
    id_to_models_info = {}
    for i, study_name in enumerate(os.listdir(exp_path)):
        constant_config = read_yaml(os.path.join(exp_path,study_name,"processed","constant_config.yml"))          
        
        note_path = os.path.join(exp_path,study_name,"processed","note.txt")
        if os.path.exists(note_path):
            f = open(note_path)
            note = f.read()
        else:
            note = ""

        id_to_models_info[i] = {}
        id_to_models_info[i]["model_name"] = constant_config["other.model_name"]
        id_to_models_info[i]["note"] = note
        id_to_models_info[i]["constant_config"] = constant_config
        id_to_models_info[i]["study_name"] = study_name

    assert len(id_to_models_info), f"No study stored in {exp_path}"

    # ensure different model names
    for i in range(len(id_to_models_info)):
        for j in range(i+1,len(id_to_models_info)):
            if id_to_models_info[i]["model_name"] == id_to_models_info[j]["model_name"]:
                id_to_models_info[i]["model_id"] = id_to_models_info[i]["model_name"] + id_to_models_info[i]["study_name"] # can be applied several times
                id_to_models_info[j]["model_id"] = id_to_models_info[j]["model_name"] + id_to_models_info[j]["study_name"] # can be applied several times

    for i in id_to_models_info.keys():
        if not("model_id" in id_to_models_info[i]):
            id_to_models_info[i]["model_id"] = id_to_models_info[i]["model_name"]

    # load

    all_studies_summaries = []
    for i,study_name in enumerate(os.listdir(os.path.join(exp_path))):
        study_overall_summaries = pd.read_csv(os.path.join(exp_path,study_name,"processed","overall_summaries.csv"),index_col = 0)
        study_overall_summaries["model_id"] = id_to_models_info[i]["model_id"]

        all_studies_summaries.append(study_overall_summaries) 

    all_studies_summaries = pd.concat(all_studies_summaries,axis=0)
    all_studies_summaries.reset_index(inplace=True,drop=True)

    # get participant data
    all_studies_summaries["participant"] = all_studies_summaries["participant_folder_name"].apply(lambda x: int(x.split("_")[-1]))
    
    participant_data = pd.read_csv(participant_data_path,index_col=0)
    all_studies_summaries = pd.merge(all_studies_summaries,participant_data,how="left",on="participant")
    
    return id_to_models_info, all_studies_summaries