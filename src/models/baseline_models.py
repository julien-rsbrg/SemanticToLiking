import os

import torch
from torch import Tensor
from torch_geometric.nn import aggr
from torch_geometric.nn.conv import SimpleConv

from src.models.generic_model import GenericModel
from src.utils import read_yaml,save_yaml


class RandomNormalModel(GenericModel):
    def __init__(self,dim_out:int,mean:float,std:float):
        self.dim_out = dim_out
        self.mean = mean
        self.std = std

    def fit(self,dataset,val_dataset = None, **kwargs) -> tuple[any,dict]:
        """fit the model to the dataset"""
        return {}
    
    
    def predict(self,node_attr:Tensor, edge_index:Tensor, edge_attr:Tensor|None = None,**kwargs):
        """use the model to predict"""
        n_nodes = node_attr.size(0)
        model_out = self.std * torch.randn(n_nodes,self.dim_out) + self.mean
        return model_out  
    
    def save(self,dst_path:str):
        """save the model
        
        Parameters
        ----------
        dst_path : str
            Path to save the model. No extension.
        """
        save_yaml(data = self.get_config(),dst_path=dst_path+".yml")


    def load(self,src_path:str):
        """load the model"""
        config = read_yaml(src_path=src_path)
        self = RandomNormalModel(**config["parameters"])
        return self

    def reset_parameters(self):
        """reset parameters of the model"""
        pass

    
    def get_config(self):
        """get configuration for the model"""
        config = {
            "name":"RandomNormalModel",
            "parameters":{
                "dim_out":self.dim_out,
                "mean":self.mean,
                "std":self.std
            }
        }
        return config
    
    def get_dict_params(self):
        return {}


class SimpleConvModel(GenericModel):
    def __init__(self,aggr = "mean"):
        self.aggr = aggr
        self.simple_conv = SimpleConv(aggr = self.aggr)
        

    def fit(self,dataset,val_dataset = None, **kwargs) -> tuple[any,dict]:
        """fit the model to the dataset"""
        return {}
    
    def predict(self,node_attr:Tensor, edge_index:Tensor, edge_attr:Tensor|None = None,**kwargs):
        """use the model to predict"""
        model_out = self.simple_conv.forward(x=node_attr,edge_index=edge_index)
        return model_out
    
    
    def save(self,dst_path:str):
        """save the model
        
        Parameters
        ----------
        dst_path : str
            Path to save the model. No extension.
        
        dst_path : str
            Path to save the model. No extension.
        """
        save_yaml(data = self.get_config(),dst_path=dst_path+".yml")


    def load(self,src_path:str):
        """load the model"""
        config = read_yaml(src_path=src_path)
        self = SimpleConvModel(**config["parameters"])
        return self

    def reset_parameters(self):
        """reset parameters of the model"""
        pass

    
    def get_config(self):
        """get configuration for the model"""
        config = {
            "name":"SimpleConvModel",
            "parameters":{
                "aggr":self.aggr
            }
        }
        return config
    
    def get_dict_params(self):
        return {}