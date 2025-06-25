from collections.abc import MutableMapping
from typing import Iterable

import os
import yaml

import numpy as np
import pandas as pd
import torch

def recursive_mkdirs(folder_path:str)->None:
    """
    Recursively create the folders leading to folder_path

    Arguments
    ---------
    - folder_path: (str)
      path to folder to create as well as its ancestors
    """
    parent_folder_path = os.path.dirname(folder_path)
    if not (os.path.exists(parent_folder_path)) and parent_folder_path != "":
        recursive_mkdirs(parent_folder_path)
        if not(os.path.exists(folder_path)):
            os.makedirs(folder_path)
    else:
        if not(os.path.exists(folder_path)):
            os.makedirs(folder_path)



def locate_in_list(var,list):
    for i in range(len(list)):
        if var == list[i]:
            return i
    return None


def _flatten_dict_gen(d, parent_key, sep):
    for k, v in d.items():
        new_key = str(parent_key) + sep + str(k) if parent_key else k
        if isinstance(v, MutableMapping):
            yield from flatten_dict(v, new_key, sep=sep).items()
        elif isinstance(v, list):
            yield from flatten_dict({i:v[i] for i in range(len(v))},new_key,sep=sep).items()
        else:
            yield new_key, v


def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str = '.'):
    return dict(_flatten_dict_gen(d, parent_key, sep))


def _turn_dict_values_tensor_to_list_gen(d):
    for k, v in d.items():
        if isinstance(v, MutableMapping):
            yield k, turn_dict_values_tensor_to_list(v)
        elif isinstance(v, torch.Tensor):
            yield k, v.detach().cpu().numpy().tolist()
        else:
            yield k, v

def turn_dict_values_tensor_to_list(d: MutableMapping):
    return dict(_turn_dict_values_tensor_to_list_gen(d))

def _turn_dict_values_iterable_to_list_gen(d: MutableMapping):
    for k, v in d.items():
        if isinstance(v, MutableMapping):
            yield k, turn_dict_values_iterable_to_list(v)
        elif isinstance(v, Iterable) and not(isinstance(v,str)):
            yield k, list(turn_dict_values_iterable_to_list({i:v for i,v in enumerate(v)}).values())
        else:
            yield k, v

def turn_dict_values_iterable_to_list(d: MutableMapping):
    _d = turn_dict_values_tensor_to_list(d)
    return dict(_turn_dict_values_iterable_to_list_gen(_d))


def ensure_one_dim_dict(d: MutableMapping):
    new_d = {}
    for k,v in d.items():
        if not(isinstance(v,Iterable)):
            new_d[k] = [v]
        elif isinstance(v,np.ndarray) and len(v.shape) != 1:
            if len(v.shape) == 0:
                new_d[k] = np.expand_dims(v,0)
            else:
                new_d[k] = v.flatten()    
        elif isinstance(v,torch.Tensor) and len(v.size()) != 1:
            new_d[k] = v.flatten()   
        else:
            new_d[k] = v
    return new_d


def convert_dataframe_to_dict(df:pd.DataFrame):
    raw_d = df.to_dict()
    processed_d = {k:list(v.values()) for k,v in raw_d.items()}
    return processed_d


def read_yaml(src_path:str) -> dict:
    assert os.path.splitext(src_path)[1] in [".yml",".yaml"], f"Wrong extension for the file {src_path}: should be .yml or .yaml"

    with open(src_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def save_yaml(data:dict,dst_path:str):
    assert os.path.splitext(dst_path)[1] in [".yml",".yaml"], f"Wrong extension for the file {dst_path}: should be .yml or .yaml"
    assert os.path.dirname(dst_path) == "" or os.path.exists(os.path.dirname(dst_path)), f"Directory {os.path.dirname(dst_path)} does not exist"

    with open(dst_path, 'w') as file:
        yaml.dump(data, file)


def compute_BIC(y_pred,y_true,n_params):
    # https://en.wikipedia.org/wiki/Bayesian_information_criterion "Under the assumption that the model errors or disturbances are independent and identically distributed according to a normal distribution and the boundary condition that the derivative of the log likelihood with respect to the true variance is zero"
    # https://stats.stackexchange.com/questions/455592/function-for-bayesian-information-criterion-bic: "You can check if a Gaussian model is reasonable or not by a quantile plot of the residuals for example."
    
    assert len(y_pred.shape) == 2, y_pred.shape
    assert len(y_true.shape) == 2, y_true.shape 
    
    n_samples = y_pred.shape[0]
    RSS = np.sum(np.power(y_true - y_pred, 2),axis=0)
    return n_samples * np.log(RSS / n_samples) + n_params * np.log(n_samples)


if __name__ == "__main__":
    import pandas as pd
    save_yaml(pd.DataFrame({"a":[0],"b":[3]}).iloc[0].to_dict(),"here.yml")
    print(read_yaml("here.yml"))


