"""
Handle all data loading
"""

import os
import numpy as np
import pandas as pd

import dash_src.utils as utils

from dash_src.configs.main import config

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