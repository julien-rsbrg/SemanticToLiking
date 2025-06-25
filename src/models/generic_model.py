import collections.abc
from abc import ABC, abstractmethod

import os
import pandas as pd

from src.utils import flatten_dict, ensure_one_dim_dict


class FilterKwargs():
    def __init__(self,kept_keywords):
        self.kept_keywords = kept_keywords
    
    def __call__(self,func):
        def wrapper(*args,**kwargs):
            for k in list(kwargs.keys()):
                if k not in self.kept_keywords:
                    del kwargs[k]
            return func(*args,**kwargs)
        return wrapper

class ForceFitOutputs():
    """Force a fit function to return model and history.
    If it is impossible, model is replaced by None and history by {}
    """
    def __call__(self,fit_func):
        def wrapper(*args,**kwargs):
            fit_out = fit_func(*args,**kwargs)
            if fit_out is None:
                return None,None
            if isinstance(fit_out,collections.abc.Iterable):
                if len(fit_out)==1:
                    if isinstance(fit_out,dict):
                        # surely history
                        return None,fit_out
                    else:
                        # surely model
                        return fit_out,{}
                elif len(fit_out)==2:
                    return fit_out
            else:
                # surely model
                return fit_out,{}
                # raise NotImplementedError("Issue with ForceFitOutputs. Fit function returns too many outputs: only need (model,history).")
                
        return wrapper
    

class GenericModel(ABC):
    @abstractmethod
    def fit(self,dataset,val_dataset = None, **kwargs) -> tuple[any,dict]:
        """fit the model to the dataset"""
        pass
    
    
    @abstractmethod
    def predict(self,node_attr, edge_index, edge_attr = None, **kwargs):
        """use the model to predict"""
        pass
    
    
    @abstractmethod
    def save(self,dst_path:str):
        """save the model
        
        Parameters
        ----------
        dst_path : str
            Path to save the model. No extension.
        """
        pass

    
    @abstractmethod
    def load(self,src_path:str):
        """load the model
        
        Parameters
        ----------
        src_path : str
            source path to the configuration / save of the model

        Returns
        -------
        self : GenericModel
            a new version of the model
        """
        pass
    

    def save_parameters(self,dst_path):
        """save the model's parameters under a .csv format
        
        Parameters
        ----------
        dst_path : str
            Path to save the model's parameters. No extension.
        """
        dict_params = flatten_dict(self.get_dict_params())
        dict_params = ensure_one_dim_dict(dict_params)
        pd.DataFrame(dict_params).to_csv(dst_path + ".csv")


    @abstractmethod
    def reset_parameters(self):
        """reset parameters of the model"""
        pass

    
    @abstractmethod
    def get_config(self):
        """get configuration for the model. Should provide n_free_params"""
        pass

    @abstractmethod
    def get_dict_params(self):
        """get a dictionary of the model's parameters"""
        pass



################
"""
class GNNFrameworkWrapper(GenericModel):
    def __init__(self,
                 model:GNNFramework,
                 optimizer_lr:float=1e-3,
                 scheduler_start_lr:float=1e-3,
                 scheduler_end_lr:float=1e-3,
                 scheduler_total_ites:int=1
                 ):
        self.model = model
        self.optimizer = self.model.configure_optimizer(lr=optimizer_lr)
        self.scheduler = self.model.configure_scheduler(self.optimizer, scheduler_start_lr, scheduler_end_lr, scheduler_total_ites)

        self.fit = FilterKwargs(kept_keywords={"dataset","val_dataset"})(self.fit)
        self.fit = ForceFitOutputs()(self.fit)
        self.predict = FilterKwargs(kept_keywords=self.predict_param_names)(self.model.predict)


    def fit(self,dataset,val_dataset,**kwargs):
        self.model.train(
            dataset=dataset,
            val_dataset=val_dataset,
            epochs=kwargs["epochs"] 
        )


    def predict(self,data):
        pass
"""