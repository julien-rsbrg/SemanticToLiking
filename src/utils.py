from collections.abc import MutableMapping
from typing import Iterable

import os
import copy
import yaml

import numpy as np
import pandas as pd
import torch

import statsmodels.api as sm
import scipy.stats as stats

import matplotlib.pyplot as plt


def handle_plot_or_save(dst_file_path=None):
    if dst_file_path is not None:
        dst_folder_path = os.path.dirname(dst_file_path)
        if not (os.path.exists(dst_folder_path)):
            os.makedirs(dst_folder_path)
        plt.savefig(dst_file_path)
        plt.close()
    else:
        plt.show()


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

## on model predictions

def check_model_normality(y_pred:np.ndarray, y_true:np.ndarray, dst_folder_path: str|None = None, save_shapiro:bool = False):
    """
    Check normality of the predictions out of a model

    Parameters
    ----------
    - y_pred : (np.ndarray) 
        model's predictions

    - y_true : (np.ndarray)
        true values to predict

    - dst_folder_path : (str|None)
        if given, will save different plots and data in the folder under "QQ_plot_residuals.png", "histo_residuals.png", "residual_vs_prediction.png", "residuals_shapiro_test.csv"
    
    - save_shapiro : (bool)
        if False, doesn't save "shapiro_test.csv" and returns it
    
    """
    if len(y_pred) == 0 or len(y_true) == 0:
        result_shapiro = {"residuals_shapiro_statistic":[np.nan],"residuals_shapiro_pvalue":[np.nan]}
        if save_shapiro and not(dst_folder_path is None):
            pd.DataFrame(result_shapiro).to_csv(os.path.join(dst_folder_path,"residuals_shapiro_test.csv"))
        else:
            return result_shapiro

    res = y_pred - y_true

    fig = sm.qqplot(res, stats.norm, fit=True, line="45")
    plt.title("QQ plot")
    dst_file_path = os.path.join(dst_folder_path,"QQ_plot_residuals.png") if not(dst_folder_path is None) else None
    handle_plot_or_save(dst_file_path)

    plt.hist(res,bins=40)
    plt.xlabel("Residual value")
    plt.ylabel("Count")
    plt.title("Residuals histogram")
    dst_file_path = os.path.join(dst_folder_path,"histo_residuals.png") if not(dst_folder_path is None) else None
    handle_plot_or_save(dst_file_path)

    pred = np.random.randn(len(res))

    plt.scatter(x=pred,y=res)
    plt.title("Plot residual vs prediction")
    plt.xlabel("Prediction")
    plt.ylabel("Residual")
    dst_file_path = os.path.join(dst_folder_path,"residual_vs_prediction.png") if not(dst_folder_path is None) else None
    handle_plot_or_save(dst_file_path)

    result_shapiro = stats.shapiro(res)
    result_shapiro = {"residuals_shapiro_statistic":[result_shapiro.statistic],"residuals_shapiro_pvalue":[result_shapiro.pvalue]}

    if save_shapiro and not(dst_folder_path is None):
        pd.DataFrame(result_shapiro).to_csv(os.path.join(dst_folder_path,"residuals_shapiro_test.csv"))
    else:
        return result_shapiro

def compute_log_likelihood_normal(y_pred,y_true):
    assert len(y_pred.shape) == 2, y_pred.shape
    assert len(y_true.shape) == 2, y_true.shape 

    n_samples = y_pred.shape[0]
    if n_samples == 0:
        return np.array([np.nan]*y_pred.shape[1])
    
    RSS = np.sum(np.power(y_true - y_pred, 2),axis=0)
    
    # to avoid a numpy error message:
    if RSS == 0.:
        return np.array([np.infty]*y_pred.shape[1])
    
    log_likelihood = - n_samples/2 * np.log(2*np.pi) - n_samples/2 * np.log(RSS/n_samples) - n_samples/2
    return log_likelihood


def compute_BIC(y_pred,y_true,n_params):
    n_samples = y_pred.shape[0]
    if n_samples == 0:
        return np.array([np.nan]*y_pred.shape[1])
     
    log_likelihood = compute_log_likelihood_normal(y_pred,y_true)
    BIC = -2*log_likelihood + n_params*np.log(n_samples)
    return BIC



def compute_AIC(y_pred,y_true,n_params):
    n_samples = y_pred.shape[0]
    if n_samples == 0:
        return np.array([np.nan]*y_pred.shape[1])
     
    log_likelihood = compute_log_likelihood_normal(y_pred,y_true)
    AIC = -2*log_likelihood + 2*n_params
    return AIC



def compute_AICc(y_pred,y_true,n_params):
    n_samples = y_pred.shape[0]
    if n_samples == 0:
        return np.array([np.nan]*y_pred.shape[1])
     
    AIC = compute_AIC(y_pred,y_true,n_params)
    if n_samples > n_params + 1:
        AICc = AIC + 2 * n_params * (n_params + 1) / (n_samples - n_params - 1)
    else:
        # penalization term is actually becoming beneficial, which is absurd
        AICc = AIC
    return AICc


## on graphs

def compute_vector_distance(A,B,ord=1):
    """
    res[i,j] = ||A[i,:] - B[j,:]||_ord
    """
    assert len(A.shape) == 2, A.shape
    assert len(B.shape) == 2, B.shape
    assert A.shape[1] == B.shape[1], (A.shape, B.shape)
    m, d = A.shape
    n, d = B.shape

    aug_A = np.einsum("mdj,ndj->mnd", A[...,np.newaxis],np.ones((n,d,1)))
    aug_B = np.einsum("ndj,mdj->mnd", B[...,np.newaxis],np.ones((m,d,1)))
    res = np.power(np.sum(np.power(np.abs(aug_A - aug_B),ord),axis = -1),1/ord)
    return res


if __name__ == "__main__":
    import pandas as pd
    save_yaml(pd.DataFrame({"a":[0],"b":[3]}).iloc[0].to_dict(),"here.yml")
    print(read_yaml("here.yml"))


