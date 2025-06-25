import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

import torch
import torch_geometric.data 
from torch_geometric.data import Data

from abc import ABC, abstractmethod
from typing import List,Tuple,Callable
import copy



# Node attribute transformation

class NodeTransformator(ABC):
    """
    Operate on the nodes' attributes of a graph 

    (Respect sklearn BaseEstimator and TransformerMixin baseline)

    (Machine library agnostic)

    Attributes
    ----------
    input_types : list[str]

    verbose : bool

    _feature_names_in : list[str]

    _feature_names_out : list[str]


    Methods
    -------
    fit()

    transform()
    
    fit_transform(X,y)
    
    """

    def __init__(self,  verbose: str):
        """Instantiate a base selector"""
        self.verbose = verbose
        self._feature_names_in = None
        self._feature_names_out = None

    @property
    def main_name(self) -> str:
        """Main name of the transform function"""
        pass
    
    @property
    def params_repr(self) -> str:
        """Representation of the parameters of the transform function"""
        pass

    @property
    def name(self) -> str:
        """Name of the transform function used for authentificate"""
        return self.main_name + self.params_repr

    def __repr__(self):
        return self.name

    @abstractmethod
    def get_config(self):
        """Get the configuration of the validation handler: {"name": ..., "parameters": ...}"""
        pass


    def _check_types(self, x: pd.DataFrame,  y: pd.DataFrame | None = None):
        assert (
            isinstance(x, pd.DataFrame)
        ), "Please input a DataFrame or let x None"
        assert (
            isinstance(y, pd.DataFrame) or y is None
        ), "Please input a DataFrame or let y None"



    def fit(self, 
            edge_index: np.ndarray, 
            edge_attr: pd.DataFrame,
            x: pd.DataFrame,  
            y: pd.DataFrame | None = None, 
            edge_locked: np.ndarray | None = None,
            **kwargs):
        """
        Receive a graph and fit instantiation's attributes for edge selection.

        Parameters
        ----------

        Returns
        -------

        """
        self._check_types(x,y)
        self._feature_names_in = x.columns.tolist()
        return self


    def transform(
            self, 
            edge_index: np.ndarray, 
            edge_attr: pd.DataFrame,
            x: pd.DataFrame,  
            y: pd.DataFrame | None = None, 
            edge_locked: np.ndarray | None = None,
            **kwargs) -> tuple[pd.DataFrame,pd.DataFrame|None]:
        """Directly returns the best features from what was found during fit and that are present in X.

        Parameters
        ----------
        x : 
            --

        Returns
        -------
        """
        self._check_types(x,y)
        return edge_index, edge_attr, x, y, edge_locked


    def fit_transform(
        self, 
        edge_index: np.ndarray, 
        edge_attr: pd.DataFrame,
        x: pd.DataFrame,  
        y: pd.DataFrame | None = None,
        edge_locked: np.ndarray | None = None,
        **kwargs
    ) -> tuple[pd.DataFrame,pd.DataFrame|None]:
        """Receive new graph and remove some of its edges 

        Parameters
        ----------

        Returns
        -------
        """
        self._check_types(x,y)

        self.fit(edge_index, edge_attr,x,y,edge_locked)
        edge_index, edge_attr, x, y, edge_locked = self.transform(edge_index,edge_attr,x,y,edge_locked)
        return edge_index, edge_attr, x, y, edge_locked


class SeparatePositiveNegative(NodeTransformator):
    """
    Create 2 new features based on feature_separated in x's features. One keep only positive values (named: feature_separated + "_pos") 
    and the other only negative values (named: feature_separated + "_neg") 

    Attributes
    ----------
    verbose : ...
    
    feature_separated : str
    ...

    """
    def __init__(self, verbose, feature_separated: str):
        super().__init__(verbose)
        
        self.feature_separated = feature_separated

    @property
    def main_name(self) -> str:
        """Main name of the transform function"""
        return "SeparatePositiveNegative"
    
    @property
    def params_repr(self) -> str:
        """Representation of the parameters of the transform function"""
        return self.feature_separated
    
    def get_config(self):
        """Get the configuration of the validation handler: {"name": ..., "parameters": ...}"""
        return {"name":"SeparatePositiveNegative", 
                "parameters":{"verbose":self.verbose,
                              "feature_separated":self.feature_separated}}

    def fit(self, 
            edge_index: np.ndarray, 
            edge_attr: pd.DataFrame, 
            x: pd.DataFrame,  
            y: pd.DataFrame | None = None, 
            edge_locked: np.ndarray | None = None,
            **kwargs):
        super().fit(edge_index, edge_attr,x,y,edge_locked)
        self._feature_names_out = self._feature_names_in + [self.feature_separated+"_pos", self.feature_separated+"_neg"]
        return self

    def transform(self, 
                  edge_index: np.ndarray, 
                  edge_attr: pd.DataFrame, 
                  x: pd.DataFrame,  
                  y: pd.DataFrame | None = None, 
                  edge_locked: np.ndarray | None = None,
                  **kwargs):
        edge_index,edge_attr,x,y,edge_locked = super().transform(edge_index,edge_attr,x,y,edge_locked)

        def transform_separate_feature(data:pd.DataFrame, feature:str, replace_value:float = 0.0,threshold:float = 0.0):
            new_data = copy.deepcopy(data)
            new_data[feature+"_pos"] = replace_value
            new_data[feature+"_neg"] = replace_value

            mask_pos = new_data[feature] > threshold
            new_data.loc[mask_pos,feature+"_pos"] = new_data.loc[mask_pos,feature]
            new_data.loc[~mask_pos,feature+"_neg"] = new_data.loc[~mask_pos,feature] 

            return new_data

        new_x = transform_separate_feature(x, self.feature_separated)
        self._feature_names_out = new_x.columns.tolist()

        return edge_index, edge_attr, new_x, y, edge_locked
    


class KeepNodeFeaturesSelector(NodeTransformator):
    """
    Direct feature selector of node features that keep only the ones given.
    """
    def __init__(self,verbose, feature_names_kept:list[str]):
        """
        
        Parameters
        ----------
        verbose : bool

        feature_names_kept : list[str]
        """
        super().__init__(verbose)
        self.feature_names_kept = feature_names_kept

    @property
    def main_name(self) -> str:
        """Main name of the transform function"""
        return "KeepFeaturesSelector"
    
    @property
    def params_repr(self) -> str:
        """Representation of the parameters of the transform function"""
        return ""
    
    def get_config(self):
        """Get the configuration of the validation handler: {"name": ..., "parameters": ...}"""
        return {"name":"KeepFeaturesSelector", 
                "parameters":{"verbose":self.verbose,
                              "feature_names_kept":self.feature_names_kept}}
    
    def fit(self, 
            edge_index: np.ndarray, 
            edge_attr: pd.DataFrame, 
            x: pd.DataFrame,  
            y: pd.DataFrame | None = None, 
            edge_locked: np.ndarray | None = None,
            **kwargs):
        super().fit(edge_index, edge_attr,x,y,edge_locked)
        return self

    def transform(self, 
                  edge_index: np.ndarray, 
                  edge_attr: pd.DataFrame, 
                  x: pd.DataFrame,  
                  y: pd.DataFrame | None = None, 
                  edge_locked: np.ndarray | None = None,
                  **kwargs):
        edge_index,edge_attr,new_x,new_y,edge_locked = super().transform(edge_index, edge_attr, x, y, edge_locked)
        new_x = x[self.feature_names_kept]
        return edge_index, edge_attr, new_x, new_y, edge_locked


class FillFeature(NodeTransformator):
    """
    Direct feature selector of node features that keep only the ones given.
    """
    def __init__(self,verbose, feature_name:str, mask_fn: callable, fill_value: float = 0.0):
        """
        
        Parameters
        ----------
        verbose : bool

        feature_names_kept : list[str]
        """
        super().__init__(verbose)
        self.feature_name = feature_name
        self.mask_fn = mask_fn
        self.fill_value = fill_value

    @property
    def main_name(self) -> str:
        """Main name of the transform function"""
        return "FillFeature"
    
    @property
    def params_repr(self) -> str:
        """Representation of the parameters of the transform function"""
        return ""
    
    def get_config(self):
        """Get the configuration of the validation handler: {"name": ..., "parameters": ...}"""
        return {"name":"FillFeature", 
                "parameters":{"verbose":self.verbose,
                              "feature_name":self.feature_names_kept,
                              "mask_fn":self.mask_fn.__name__,
                              "fill_value":self.fill_value}}
    
    def fit(self, 
            edge_index: np.ndarray, 
            edge_attr: pd.DataFrame, 
            x: pd.DataFrame,  
            y: pd.DataFrame | None = None, 
            edge_locked: np.ndarray | None = None,
            **kwargs):
        super().fit(edge_index, edge_attr,x,y,edge_locked)
        return self

    def transform(self, 
                  edge_index: np.ndarray, 
                  edge_attr: pd.DataFrame, 
                  x: pd.DataFrame,  
                  y: pd.DataFrame | None = None, 
                  edge_locked: np.ndarray | None = None,
                  **kwargs):
        edge_index,edge_attr,new_x,new_y,edge_locked = super().transform(edge_index, edge_attr, x, y, edge_locked)
        new_x = copy.deepcopy(new_x)
        new_x.loc[self.mask_fn(new_x),self.feature_name] = self.fill_value
        return edge_index, edge_attr, new_x, new_y, edge_locked



class PolynomialFeatureGenerator(NodeTransformator):
    """
    Create polynomial and interaction features
    
    """
    def __init__(self,
                 verbose: bool,
                 feature_names_involved:list[str],
                 degree: int = 2,
                 interaction_only:bool = False,
                 include_bias:bool = False):
        """
        
        Parameters
        ----------
        verbose : bool

        feature_names_kept : list[str]

        inputs_selected : bool
            if True, feature in x are changed; otherwise features in y
        """
        super().__init__(verbose)
        self.feature_names_involved = feature_names_involved
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.poly_generator = PolynomialFeatures(degree=degree,interaction_only=interaction_only,include_bias=include_bias)

        self._feature_names_out = None # instance to fit fist 

    @property
    def main_name(self) -> str:
        """Main name of the transform function"""
        return "PolynomialFeatureGenerator"
    
    @property
    def params_repr(self) -> str:
        """Representation of the parameters of the transform function"""
        return f"-Deg{self.degree}-InterOnly{self.interaction_only}-Bias{self.include_bias}"
    
    def get_config(self):
        """Get the configuration of the validation handler: {"name": ..., "parameters": ...}"""
        config = {"name":"PolynomialFeatureGenerator", 
                  "parameters":{"verbose":self.verbose,
                                "feature_names_involved":self.feature_names_involved,
                                "degree":self.degree,
                                "interaction_only":self.interaction_only,
                                "include_bias":self.include_bias
                                }
        }
        return config
    

    def fit(self, 
            edge_index: np.ndarray, 
            edge_attr: pd.DataFrame, 
            x: pd.DataFrame,  
            y: pd.DataFrame | None = None, 
            edge_locked: np.ndarray | None = None,
            **kwargs) -> Tuple[pd.DataFrame,pd.DataFrame]:
        super().fit(edge_index, edge_attr, x, y, edge_locked)
        
        x_involved = x[self.feature_names_involved].copy()
        self.poly_generator.fit(X=x_involved.to_numpy())

        poly_feature_names = self.poly_generator.get_feature_names_out(input_features=self.feature_names_involved)
        mock_new_df = pd.DataFrame(columns=poly_feature_names).drop(self.feature_names_involved, axis = 1)
        new_features = mock_new_df.columns.tolist()
        self._feature_names_out = self._feature_names_in + new_features

        return self

    def transform(self, 
                  edge_index: np.ndarray, 
                  edge_attr: pd.DataFrame, 
                  x: pd.DataFrame,  
                  y: pd.DataFrame | None = None, 
                  edge_locked: np.ndarray | None = None,
                  **kwargs):
        edge_index,edge_attr,x,y = super().transform(edge_index, edge_attr,x, y, edge_locked)

        x_involved = x[self.feature_names_involved].copy()
        new_x = self.poly_generator.transform(x_involved)
        new_x = pd.DataFrame(new_x,columns=self.poly_generator.get_feature_names_out(input_features=self.feature_names_involved))
        new_x = new_x.drop(self.feature_names_involved, axis = 1) # no double column

        new_x = x.join(new_x)

        return edge_index, edge_attr, new_x, y, edge_locked



# Edge Selectors

class EdgeTransformator(ABC):
    """
    Operate on the connections of a graph to remove them 

    (Respect sklearn BaseEstimator and TransformerMixin baseline)

    (Machine library agnostic)

    Attributes
    ----------
    input_types : list[str]
    
    verbose : bool

    _feature_names_in : list[str]
    
    _feature_names_out : list[str]
    

    Methods
    -------
    fit()
        --
    transform()
        --
    fit_transform(X,y)
        --
    """

    def __init__(self,  verbose: str):
        """Instantiate a base selector"""
        self.verbose = verbose

    @property
    def main_name(self) -> str:
        """Main name of the transform function"""
        pass
    
    @property
    def params_repr(self) -> str:
        """Representation of the parameters of the transform function"""
        pass

    @property
    def name(self) -> str:
        """Name of the transform function used for authentificate"""
        return self.main_name + self.params_repr

    def __repr__(self):
        return self.name
    
    @abstractmethod
    def get_config(self):
        """Get the configuration of the validation handler: {"name": ..., "parameters": ...}"""
        pass


    def __repr__(self):
        return self.name


    def _check_types(self, 
                     edge_index: np.ndarray, 
                     edge_attr: pd.DataFrame | None = None, 
                     x: pd.DataFrame | None = None, 
                     y: pd.DataFrame | None = None,
                     edge_locked: np.ndarray | None = None):
        assert (
            isinstance(edge_index,np.ndarray)
        ), ("Please input an np.ndarray for edge_index", (edge_index,type(edge_index)))
        assert (
            isinstance(edge_attr, pd.DataFrame) or edge_attr is None
        ), "Please input a DataFrame or let edge_attr None"
        assert (
            isinstance(x, pd.DataFrame) or x is None
        ), "Please input a DataFrame or let x None"
        assert (
            isinstance(y, pd.DataFrame) or y is None
        ), "Please input a DataFrame or let y None"
        assert (
            isinstance(edge_locked, np.ndarray) or edge_locked is None
        ), "Please input a np.ndarray or let edge_locked be None"


    def fit(self, 
            edge_index: np.ndarray, 
            edge_attr: pd.DataFrame | None = None, 
            x: pd.DataFrame | None = None, 
            y: pd.DataFrame | None = None,
            edge_locked: np.ndarray | None = None):
        """
        Receive a graph and fit instantiation's attributes for edge selection.

        Parameters
        ----------

        Returns
        -------

        """
        self._check_types(edge_index,edge_attr,x,y,edge_locked)

        return self

    def transform(self, 
                  edge_index: np.ndarray, 
                  edge_attr: pd.DataFrame | None = None, 
                  x: pd.DataFrame | None = None,  
                  y: pd.DataFrame | None = None,
                  edge_locked: np.ndarray | None = None) -> tuple[np.ndarray,pd.DataFrame|None,pd.DataFrame|None,pd.DataFrame|None]:
        """Directly returns the best features from what was found during fit and that are present in X.

        Parameters
        ----------
        x : 
            --

        Returns
        -------
        """
        self._check_types(edge_index,edge_attr,x,y,edge_locked)

        if edge_locked is None:
            edge_locked = np.zeros(edge_index.shape[-1],dtype=bool)
        edge_locked = copy.deepcopy(edge_locked)

        return edge_index, edge_attr, x, y, edge_locked

    def fit_transform(
        self, 
        edge_index: np.ndarray, 
        edge_attr: pd.DataFrame | None = None, 
        x: pd.DataFrame | None = None,  
        y: pd.DataFrame | None = None,
        edge_locked: np.ndarray | None = None
    ) -> tuple[np.ndarray,pd.DataFrame|None,pd.DataFrame|None,pd.DataFrame|None]:
        """Receive new graph and remove some of its edges 

        Parameters
        ----------

        Returns
        -------
        """
        self._check_types(edge_index,edge_attr,x,y,edge_locked)

        self.fit(edge_index,edge_attr,x,y,edge_locked)
        new_edge_index, new_edge_attr, x, y, new_edge_locked = self.transform(edge_index,edge_attr,x,y,edge_locked)
        return new_edge_index, new_edge_attr, x, y, new_edge_locked

class KeepEdgeFeaturesSelector(EdgeTransformator):
    """
    Direct feature selector of edge features that keep only the ones given.
    """
    def __init__(self,verbose, feature_names_kept:list[str]):
        """
        
        Parameters
        ----------
        verbose : bool

        feature_names_kept : list[str]
        """
        super().__init__(verbose)
        self.feature_names_kept = feature_names_kept

    @property
    def main_name(self) -> str:
        """Main name of the transform function"""
        return "KeepEdgeFeaturesSelector"
    
    @property
    def params_repr(self) -> str:
        """Representation of the parameters of the transform function"""
        return ""
    
    def get_config(self):
        """Get the configuration of the validation handler: {"name": ..., "parameters": ...}"""
        return {"name":"KeepEdgeFeaturesSelector", 
                "parameters":{"verbose":self.verbose,
                              "feature_names_kept":self.feature_names_kept}}
    
    def fit(self, 
            edge_index: np.ndarray, 
            edge_attr: pd.DataFrame, 
            x: pd.DataFrame,  
            y: pd.DataFrame | None = None, 
            edge_locked: np.ndarray | None = None,
            **kwargs):
        super().fit(edge_index, edge_attr,x,y,edge_locked)
        return self

    def transform(self, 
                  edge_index: np.ndarray, 
                  edge_attr: pd.DataFrame, 
                  x: pd.DataFrame,  
                  y: pd.DataFrame | None = None, 
                  edge_locked: np.ndarray | None = None,
                  **kwargs):
        edge_index,edge_attr,x,y,edge_locked = super().transform(edge_index, edge_attr, x, y, edge_locked)
        new_edge_attr = copy.deepcopy(edge_attr)
        new_edge_attr = new_edge_attr[self.feature_names_kept]
        return edge_index, new_edge_attr, x, y, edge_locked


class FilterGroupSendersToGroupReceivers(EdgeTransformator):
    """
    Careful wisely name parameters since the names are used in configuration saving 
    
    
    """
    def __init__(
        self,
        group_senders_mask_fn: callable,
        group_receivers_mask_fn: callable,
        group_senders_thresholding: bool = False,
        group_receivers_thresholding: bool = False,
        group_senders_mask_threshold_fn: Callable|None = None,
        group_receivers_mask_threshold_fn: Callable|None = None,
        verbose: bool = True,
        keep: bool = True
    ):
        super().__init__(verbose)
        self.group_senders_mask_fn = group_senders_mask_fn
        self.group_receivers_mask_fn = group_receivers_mask_fn

        self.group_senders_thresholding = group_senders_thresholding
        self.group_receivers_thresholding = group_receivers_thresholding

        self.group_senders_mask_threshold_fn = group_senders_mask_threshold_fn
        self.group_receivers_mask_threshold_fn = group_receivers_mask_threshold_fn

        self.keep = keep # else remove
    

    @property
    def main_name(self) -> str:
        """Main name of the transform function"""
        return "FilterGroupSendersToGroupReceivers"
    
    @property
    def params_repr(self) -> str:
        """Representation of the parameters of the transform function"""
        return ""

    @property
    def name(self) -> str:
        """Name of the transform function used for authentificate"""
        return self.main_name + self.params_repr
    
    def get_config(self):
        """Get the configuration of the validation handler: {"name": ..., "parameters": ...}"""
        config = {"name":"FilterGroupSendersToGroupReceivers", 
                  "parameters":{"verbose":self.verbose,
                                "group_senders_mask_fn":self.group_senders_mask_fn.__name__,
                                "group_receivers_mask_fn":self.group_receivers_mask_fn.__name__,
                                "group_senders_thresholding":self.group_senders_thresholding,
                                "group_receivers_thresholding":self.group_receivers_thresholding,
                                "group_senders_mask_threshold_fn":self.group_senders_mask_threshold_fn.__name__ if not(self.group_senders_mask_threshold_fn is None) else None,
                                "group_receivers_mask_threshold_fn":self.group_receivers_mask_threshold_fn.__name__ if not(self.group_receivers_mask_threshold_fn is None) else None,
                                "keep":self.keep
                                }
        }
        return config
    
    def transform(self, 
                  edge_index:np.ndarray, 
                  edge_attr: pd.DataFrame | None = None, 
                  x: pd.DataFrame = None, 
                  y: pd.DataFrame | None = None,
                  edge_locked: np.ndarray | None = None) -> tuple[np.ndarray,pd.DataFrame|None,pd.DataFrame|None,pd.DataFrame|None]:
        edge_index, edge_attr, x, y, edge_locked = super().transform(edge_index, edge_attr, x, y, edge_locked)

        if self.group_senders_thresholding:
            threshold_sender = self.group_senders_mask_threshold_fn(x)
            is_sender_selected = self.group_senders_mask_fn(x,threshold_sender)
        else:
            is_sender_selected = self.group_senders_mask_fn(x)

        if self.group_receivers_thresholding:
            threshold_receiver = self.group_receivers_mask_threshold_fn(x)
            is_receiver_selected = self.group_receivers_mask_fn(x,threshold_receiver)
        else:
            is_receiver_selected = self.group_receivers_mask_fn(x)
        

        if self.verbose:
            print("Number of Edges before transform:", edge_index.shape[1])
                  
        is_edge_selected = np.zeros(edge_index.shape,dtype=bool)
        is_edge_selected[0,:] = is_sender_selected[edge_index[0,:]]
        is_edge_selected[1,:] = is_receiver_selected[edge_index[1,:]]
        is_edge_selected = is_edge_selected[0,:] * is_edge_selected[1,:]

        if not(self.keep):
            is_edge_selected = ~is_edge_selected

        new_edge_index = copy.deepcopy(edge_index)[:,is_edge_selected+edge_locked]
        if not(edge_attr is None):
            new_edge_attr = copy.deepcopy(edge_attr).iloc[is_edge_selected+edge_locked]
        else:
            new_edge_attr = None
        
        new_x = copy.deepcopy(x)
        new_y = copy.deepcopy(y)
        new_edge_locked = copy.deepcopy(edge_locked)

        if self.verbose:
            print("Number of Edges after transform:", new_edge_index.shape[1])

        return new_edge_index, new_edge_attr, new_x, new_y, new_edge_locked




class KeepKNearestNeighbors(EdgeTransformator):
    def __init__(
        self,
        k: int,
        edge_attr_names_used: list[str],
        ord: int = 1,
        mode: str = "min",
        verbose: bool = True
    ):
        """
        Parameters
        ----------
        - mode : (str)
            Can be "max" or "min"
        """
        super().__init__(verbose)
        self.k = k
        self.edge_attr_names_used = edge_attr_names_used
        self.ord = ord
        self.mode = mode

        assert ord != 0 or len(edge_attr_names_used) == 1

    @property
    def main_name(self) -> str:
        """Main name of the transform function"""
        return "KeepKNearestNeighbors"
    
    @property
    def params_repr(self) -> str:
        """Representation of the parameters of the transform function"""
        return str(self.k)

    @property
    def name(self) -> str:
        """Name of the transform function used for authentificate"""
        return self.main_name + self.params_repr
    
    def get_config(self):
        """Get the configuration of the validation handler: {"name": ..., "parameters": ...}"""
        config = {"name":"KeepKNearestNeighbors", 
                  "parameters":{"verbose":self.verbose,
                                "k":self.k,
                                "edge_attr_names_used":self.edge_attr_names_used,
                                "ord":self.ord,
                                "mode":self.mode}
        }
        return config

    def transform(self, 
                  edge_index:np.ndarray, 
                  edge_attr: pd.DataFrame | None = None, 
                  x: pd.DataFrame | None = None, 
                  y: pd.DataFrame | None = None,
                  edge_locked: np.ndarray | None = None) -> tuple[np.ndarray,pd.DataFrame|None,pd.DataFrame|None,pd.DataFrame|None]:
        
        edge_index, edge_attr, x, y, edge_locked = super().transform(edge_index, edge_attr, x, y, edge_locked)

        if self.verbose:
            print("Number of Edges before transform:", edge_index.shape[1])

        if len(self.edge_attr_names_used) >= 1:
            if self.ord == 0:
                edge_dist = edge_attr[self.edge_attr_names_used].values.flatten()
            else:
                edge_dist = np.linalg.norm(edge_attr[self.edge_attr_names_used], ord=self.ord, axis=1)
            
            edge_kept = np.zeros(edge_index.shape[1],dtype=bool)
            
            # from receiver point of view
            for node_id in np.unique(edge_index[1,:]):
                ids_node_receiver = np.where(edge_index[1,:] == node_id)[0]
                node_dist_from_sender = edge_dist[ids_node_receiver]

                if self.mode == "min":
                    kept_k_neigbors_edges = np.argsort(node_dist_from_sender)[:self.k]
                else:
                    kept_k_neigbors_edges = np.argsort(node_dist_from_sender)[-self.k:]
                edge_kept[ids_node_receiver[kept_k_neigbors_edges]] = True
            
        else:
            edge_kept = np.zeros(edge_index.shape[1],dtype=bool)
        
        new_edge_index = copy.deepcopy(edge_index)[:,edge_kept+edge_locked]
        new_edge_attr = copy.deepcopy(edge_attr).iloc[edge_kept+edge_locked]
        new_x = copy.deepcopy(x)
        new_y = copy.deepcopy(y)
        new_edge_locked = copy.deepcopy(edge_locked)
        
        if self.verbose:
            print("Number of Edges after transform:", new_edge_index.shape[1])

        return new_edge_index, new_edge_attr, new_x, new_y, new_edge_locked


class KeepNeighborsDistThreshold(EdgeTransformator):
    def __init__(
        self,
        edge_attr_names_used: str,
        threshold: float,
        ord: int = 1,
        upper: bool = False,
        verbose: bool = True
    ):
        """
        Parameters:
        -----------
        - upper : (bool)
            if True, keep edges with distance greater or equal to threshold. Else, lower or equal to threshold.
        """
        super().__init__(verbose)
        self.edge_attr_names_used = edge_attr_names_used
        self.threshold = threshold
        self.ord = ord
        self.upper = upper

    @property
    def main_name(self) -> str:
        """Main name of the transform function"""
        return "KeepNeighborsDistThreshold"
    
    @property
    def params_repr(self) -> str:
        """Representation of the parameters of the transform function"""
        comparison = ">=" if self.upper else "<="
        return comparison + str(self.threshold)

    @property
    def name(self) -> str:
        """Name of the transform function used for authentificate"""
        return self.main_name + self.params_repr
    
    def get_config(self):
        """Get the configuration of the validation handler: {"name": ..., "parameters": ...}"""
        config = {"name":"KeepNeighborsDistThreshold", 
                  "parameters":{"verbose":self.verbose,
                                "edge_attr_names_used":self.edge_attr_names_used,
                                "threshold":self.threshold,
                                "ord":self.ord,
                                "upper":self.upper}
        }
        return config

    def transform(self, 
                  edge_index:np.ndarray, 
                  edge_attr: pd.DataFrame | None = None, 
                  x: pd.DataFrame | None = None, 
                  y: pd.DataFrame | None = None,
                  edge_locked: np.ndarray | None = None) -> tuple[np.ndarray,pd.DataFrame|None,pd.DataFrame|None,pd.DataFrame|None]:
        edge_index, edge_attr, x, y, edge_locked = super().transform(edge_index, edge_attr, x, y, edge_locked)

        if self.verbose:
            print("Number of Edges before transform:", edge_index.shape[1])
        
        if len(self.edge_attr_names_used) >= 1:
            edge_dist = np.linalg.norm(edge_attr[self.edge_attr_names_used], ord=self.ord, axis=1)

            if self.upper:
                edge_kept = edge_dist >= self.threshold
            else:
                edge_kept = edge_dist <= self.threshold
            
        else:
            edge_kept = np.zeros(edge_index.shape[1],dtype=bool)
        
        
        new_edge_index = copy.deepcopy(edge_index)[:,edge_kept+edge_locked]
        new_edge_attr = copy.deepcopy(edge_attr).iloc[edge_kept+edge_locked]
        new_x = copy.deepcopy(x)
        new_y = copy.deepcopy(y)
        new_edge_locked = copy.deepcopy(edge_locked)
        
        if self.verbose:
            print("Number of Edges after transform:", new_edge_index.shape[1])

        return new_edge_index, new_edge_attr, new_x, new_y, new_edge_locked



class KeepMonotonousNodeAttr(EdgeTransformator):
    def __init__(
        self,
        node_attr_name_used: str,
        ascending: bool = True, # else decreasing
        strict: bool = True,
        verbose: bool = True
    ):
        """
        Parameters:
        -----------
        - upper : (bool)
            if True, keep edges with distance greater or equal to threshold. Else, lower or equal to threshold.
        """
        super().__init__(verbose)
        self.node_attr_name_used = node_attr_name_used
        self.ascending = ascending
        self.strict = strict

    @property
    def main_name(self) -> str:
        """Main name of the transform function"""
        return "KeepMonotonousNodeAttr"
    
    @property
    def params_repr(self) -> str:
        """Representation of the parameters of the transform function"""
        txt = ""
        if self.strict:
            txt += "Strict"

        if self.ascending:
            txt += "Ascending"
        else:
            txt += "Descending"
        
        return txt

    @property
    def name(self) -> str:
        """Name of the transform function used for authentificate"""
        return self.main_name + "-" + self.params_repr
    
    def get_config(self):
        """Get the configuration of the validation handler: {"name": ..., "parameters": ...}"""
        config = {"name":"KeepMonotonousNodeAttr", 
                  "parameters":{"verbose":self.verbose,
                                "node_attr_name_used":self.node_attr_name_used,
                                "ascending":self.ascending,
                                "strict":self.threshold}
        }
        return config

    def transform(self, 
                  edge_index:np.ndarray, 
                  edge_attr: pd.DataFrame | None = None, 
                  x: pd.DataFrame | None = None, 
                  y: pd.DataFrame | None = None,
                  edge_locked: np.ndarray | None = None) -> tuple[np.ndarray,pd.DataFrame|None,pd.DataFrame|None,pd.DataFrame|None]:
        edge_index, edge_attr, x, y, edge_locked = super().transform(edge_index, edge_attr, x, y, edge_locked)

        if self.verbose:
            print("Number of Edges before transform:", edge_index.shape[1])
        
        sender_ids = edge_index[0,:]
        receiver_ids = edge_index[1,:]

        sender_values = x[self.node_attr_name_used].values[sender_ids]
        receiver_values = x[self.node_attr_name_used].values[receiver_ids]

        diff = sender_values - receiver_values

        if self.ascending:
            if self.strict:
                edge_kept = diff < 0
            else:
                edge_kept = diff <= 0
        else:
            if self.strict:
                edge_kept = diff > 0
            else:
                edge_kept = diff >= 0        
        
        new_edge_index = copy.deepcopy(edge_index[:,edge_kept+edge_locked])
        new_edge_attr = copy.deepcopy(edge_attr[edge_kept+edge_locked])
        new_x = copy.deepcopy(x)
        new_y = copy.deepcopy(y)
        new_edge_locked = copy.deepcopy(edge_locked)
        
        if self.verbose:
            print("Number of Edges after transform:", new_edge_index.shape[1])

        return new_edge_index, new_edge_attr, new_x, new_y, new_edge_locked



class LockKNearestGroupSendersToGroupReceivers(EdgeTransformator):
    def __init__(
        self,
        group_senders_mask_fn: callable,
        group_receivers_mask_fn: callable,
        k: int,
        edge_attr_names_used: list[str],
        ord: int = 1,
        mode: str = "min",
        verbose: bool = True
    ):
        """
        Parameters:
        -----------
        - upper : (bool)
            if True, keep edges with distance greater or equal to threshold. Else, lower or equal to threshold.
        
        - mode : (str)
            Can be "max" or "min"
        """
        super().__init__(verbose)

        self.group_senders_mask_fn = group_senders_mask_fn
        self.group_receivers_mask_fn = group_receivers_mask_fn

        self.k = k
        self.edge_attr_names_used = edge_attr_names_used
        self.ord = ord
        self.mode = mode
        assert ord != 0 or len(edge_attr_names_used) == 1

    @property
    def main_name(self) -> str:
        """Main name of the transform function"""
        return "LockKNearestGroupSendersToGroupReceivers"
    
    @property
    def params_repr(self) -> str:
        """Representation of the parameters of the transform function"""        
        return str(self.k)

    @property
    def name(self) -> str:
        """Name of the transform function used for authentificate"""
        return self.main_name + "-" + self.params_repr
    
    def get_config(self):
        """Get the configuration of the validation handler: {"name": ..., "parameters": ...}"""
        config = {"name":"LockKNearestGroupSendersToGroupReceivers", 
                  "parameters":{"verbose":self.verbose,
                                "group_senders_mask_fn":self.group_senders_mask_fn.__name__,
                                "group_receivers_mask_fn":self.group_receivers_mask_fn.__name__,
                                "k":self.k,
                                "edge_attr_names_used":self.edge_attr_names_used,
                                "ord":self.ord,
                                "mode":self.mode
                                }
        }
        return config

    def transform(self, 
                  edge_index:np.ndarray, 
                  edge_attr: pd.DataFrame | None = None, 
                  x: pd.DataFrame | None = None, 
                  y: pd.DataFrame | None = None,
                  edge_locked: np.ndarray | None = None) -> tuple[np.ndarray,pd.DataFrame|None,pd.DataFrame|None,pd.DataFrame|None]:
        edge_index, edge_attr, x, y, new_edge_locked = super().transform(edge_index, edge_attr, x, y, edge_locked)

        if self.verbose:
            print("Number of Edges Locked before transform:", new_edge_locked.sum())

        is_sender_selected = self.group_senders_mask_fn(x)
        is_receiver_selected = self.group_receivers_mask_fn(x)
                  
        is_edge_selected = np.zeros(edge_index.shape,dtype=bool)
        is_edge_selected[0,:] = is_sender_selected[edge_index[0,:]]
        is_edge_selected[1,:] = is_receiver_selected[edge_index[1,:]]
        is_edge_selected = is_edge_selected[0,:] * is_edge_selected[1,:]


        if len(self.edge_attr_names_used) >= 1:
            if self.ord == 0:
                edge_dist = edge_attr[self.edge_attr_names_used].values.flatten()
            else:
                edge_dist = np.linalg.norm(edge_attr[self.edge_attr_names_used], ord=self.ord, axis=1)

            # from receiver point of view
            receiver_ids = np.where(is_receiver_selected)[0]
            for node_id in np.unique(receiver_ids):
                edge_node_receiver = edge_index[1,:] == node_id
                edge_from_sender_to_node_receiver = edge_node_receiver * is_edge_selected
                edge_candidate_ids = np.where(edge_from_sender_to_node_receiver)[0]
                node_dist_from_sender = edge_dist[edge_candidate_ids]
                if self.mode == "min":
                    kept_k_sender_edges = np.argsort(node_dist_from_sender)[:self.k]
                else:
                    kept_k_sender_edges = np.argsort(node_dist_from_sender)[-self.k:]
                new_edge_locked[edge_candidate_ids[kept_k_sender_edges]] = True

        new_edge_index = copy.deepcopy(edge_index)
        new_edge_attr = copy.deepcopy(edge_attr)
        new_x = copy.deepcopy(x)
        new_y = copy.deepcopy(y)
        
        if self.verbose:
            print("Number of Edges Locked after transform:", new_edge_locked.sum())

        return new_edge_index, new_edge_attr, new_x, new_y, new_edge_locked


class LockGroupSendersToGroupReceivers(EdgeTransformator):
    def __init__(
        self,
        group_senders_mask_fn: callable,
        group_receivers_mask_fn: callable,
        verbose: bool = True,
        unlock: bool = False
    ):
        """
        Parameters:
        -----------
        - upper : (bool)
            if True, keep edges with distance greater or equal to threshold. Else, lower or equal to threshold.
        """
        super().__init__(verbose)

        self.group_senders_mask_fn = group_senders_mask_fn
        self.group_receivers_mask_fn = group_receivers_mask_fn
        self.unlock = unlock

    @property
    def main_name(self) -> str:
        """Main name of the transform function"""
        return "LockGroupSendersToGroupReceivers"
    
    @property
    def params_repr(self) -> str:
        """Representation of the parameters of the transform function"""        
        return str(self.k)

    @property
    def name(self) -> str:
        """Name of the transform function used for authentificate"""
        return self.main_name + "-" + self.params_repr
    
    def get_config(self):
        """Get the configuration of the validation handler: {"name": ..., "parameters": ...}"""
        config = {"name":"LockGroupSendersToGroupReceivers", 
                  "parameters":{"verbose":self.verbose,
                                "group_senders_mask_fn":self.group_senders_mask_fn.__name__,
                                "group_receivers_mask_fn":self.group_receivers_mask_fn.__name__,
                                "unlock":self.unlock
                                }
        }
        return config

    def transform(self, 
                  edge_index:np.ndarray, 
                  edge_attr: pd.DataFrame | None = None, 
                  x: pd.DataFrame | None = None, 
                  y: pd.DataFrame | None = None,
                  edge_locked: np.ndarray | None = None) -> tuple[np.ndarray,pd.DataFrame|None,pd.DataFrame|None,pd.DataFrame|None]:
        edge_index, edge_attr, x, y, new_edge_locked = super().transform(edge_index, edge_attr, x, y, edge_locked)

        if self.verbose:
            print("Number of Edges Locked before transform:", new_edge_locked.sum())

        is_sender_selected = self.group_senders_mask_fn(x)
        is_receiver_selected = self.group_receivers_mask_fn(x)
                  
        is_edge_selected = np.zeros(edge_index.shape,dtype=bool)
        is_edge_selected[0,:] = is_sender_selected[edge_index[0,:]]
        is_edge_selected[1,:] = is_receiver_selected[edge_index[1,:]]
        is_edge_selected = is_edge_selected[0,:] * is_edge_selected[1,:]
        
        if self.unlock:
            new_edge_locked[is_edge_selected] = False
        else:
            new_edge_locked[is_edge_selected] = True
        
        new_edge_index = copy.deepcopy(edge_index)
        new_edge_attr = copy.deepcopy(edge_attr)
        new_x = copy.deepcopy(x)
        new_y = copy.deepcopy(y)

        if self.verbose:
            print("Number of Edges Locked after transform:", new_edge_locked.sum())

        return new_edge_index, new_edge_attr, new_x, new_y, new_edge_locked


#### Training samples selection

class MaskSelector(ABC):
    def __init__(self,verbose:bool = True,**kwargs):
        self.verbose = verbose
        pass


    @property
    def main_name(self) -> str:
        """Main name of the mask selector"""
        pass


    @property
    def params_repr(self) -> str:
        """Representation of the parameters of the mask selector"""
        pass


    @property
    def name(self) -> str:
        """Name of the mask selector used for authentificate"""
        return self.main_name + self.params_repr
    
    def __repr__(self):
        return self.name
    
    @abstractmethod
    def get_config(self):
        """Get the configuration of the validation handler: {"name": ..., "parameters": ...}"""
        pass
    
    @abstractmethod
    def apply(self, x: pd.DataFrame) -> torch.Tensor:
        pass




class MaskLowerThanSelector(MaskSelector):
    def __init__(self, feature_name:str, threshold:float, **kwargs):
        super().__init__(**kwargs)
        self.feature_name = feature_name
        self.threshold = threshold
    

    @property
    def main_name(self) -> str:
        """Main name of the mask selector"""
        return "mask_"
    

    @property
    def params_repr(self) -> str:
        """Representation of the parameters of the mask selector"""
        return f"{self.feature_name}<={self.threshold}"

    
    def get_config(self):
        """Get the configuration of the validation handler: {"name": ..., "parameters": ...}"""
        config = {"name":"MaskLowerThanSelector", 
                  "parameters":{"verbose":self.verbose,
                                "feature_name":self.feature_name,
                                "threshold":self.threshold
                                }
        }
        return config


    def apply(self, x: pd.DataFrame) -> torch.Tensor:
        mask = torch.zeros(len(x),dtype=torch.bool)
        mask[x[self.feature_name]<=0] = True
        return mask


class MaskKeepQuantile(MaskSelector):
    def __init__(self, feature_name:str, q:float, mode:str = "lower", **kwargs):
        """
        Parameters
        ----------
        mode : (str)
        - keep in ["lower", "upper"] takes lower or upper portion of the data
        
        """
        super().__init__(**kwargs)
        self.feature_name = feature_name
        self.q = q
        self.mode = mode

    @property
    def main_name(self) -> str:
        """Main name of the mask selector"""
        return "mask_quantile_"
    

    @property
    def params_repr(self) -> str:
        """Representation of the parameters of the mask selector"""
        comparator = "<=" if self.mode == "lower" else ">="
        return f"{self.feature_name}"+comparator+"{self.q}%"

    
    def get_config(self):
        """Get the configuration of the validation handler: {"name": ..., "parameters": ...}"""
        config = {"name":"MaskKeepQuantile", 
                  "parameters":{"verbose":self.verbose,
                                "feature_name":self.feature_name,
                                "q":self.q,
                                "mode":self.mode
                                }
        }
        return config


    def apply(self, x: pd.DataFrame) -> torch.Tensor:
        mask = torch.zeros(len(x),dtype=torch.bool)
        quantile = torch.quantile(torch.Tensor(x[self.feature_name]),self.q)
        quantile = float(quantile)
        if self.mode == "lower":
            mask[x[self.feature_name]<=quantile] = True
        else:
            mask[x[self.feature_name]>=quantile] = True
        return mask


#### ValidationHandlers

class ValidationHandler(ABC):
    def __init__(self):
        pass
    
    @property
    def main_name(self) -> str:
        """Main name of the validation handler"""
        pass


    @property
    def params_repr(self) -> str:
        """Representation of the parameters of the validation handler"""
        pass
    
    
    @property
    def name(self) -> str:
        """Name of the validation handler used for authentificate"""
        return self.main_name + self.params_repr
    
    def __repr__(self):
        return self.name
    
    @abstractmethod
    def get_config(self):
        """Get the configuration of the validation handler: {"name": ..., "parameters": ...}"""
        pass


    @abstractmethod
    def apply(self, mask:torch.Tensor) -> list[Tuple[torch.Tensor,torch.Tensor]]:
        """
        Takes the truth values of the mask and parcelates it

        Parameters
        ----------
        mask : torch.Tensor

        Returns
        -------
        train_val_sets : list[Tuple[torch.Tensor,torch.Tensor]]
        """
        pass




class CrossValidationHandler(ValidationHandler):
    """
    If the number of partitions is too large compared to the available data, only one partition is made with an empy validation set 
    
    """
    def __init__(self,n_partition:int = 10):
        super().__init__()
        self.n_partition = n_partition

    @property
    def main_name(self) -> str:
        """Main name of the validation handler"""
        return "crossvalidation"


    @property
    def params_repr(self) -> str:
        """Representation of the parameters of the validation handler"""
        return str(self.n_partition)
    

    def get_config(self):
        """Get the configuration of the validation handler: {"name": ..., "parameters": ...}"""
        config = {"name":"CrossValidationHandler", 
                  "parameters":{"n_partition":self.n_partition}
        }
        return config
    
    def apply(self, mask:torch.Tensor) -> list[Tuple[torch.Tensor,torch.Tensor]]:
        """
        Takes the truth values of the mask and parcelates it. Create a n_partition partition

        Parameters
        ----------
        mask : torch.Tensor

        Returns
        -------
        train_val_sets : list[Tuple[torch.Tensor,torch.Tensor]]
        """
        train_val_sets = []
        poss_val_ids = mask.clone()

        n_val_samples_per_partition = int(mask.sum())//self.n_partition
        while int(poss_val_ids.sum()) > 0:
            n_val_samples_per_partition = min(n_val_samples_per_partition,int(poss_val_ids.sum()))

            if n_val_samples_per_partition:
                val_ids = torch.multinomial(poss_val_ids.to(torch.float64), n_val_samples_per_partition, replacement=False)
                poss_val_ids[val_ids] = False
            else:
                val_ids = []
                poss_val_ids[:] = False

            new_train_mask = mask.clone()
            new_train_mask[val_ids] = False

            new_val_mask = torch.zeros(len(mask),dtype=torch.bool)
            new_val_mask[val_ids] = True

            train_val_sets.append((new_train_mask,new_val_mask))

        return train_val_sets
        


class HoldPOutValidationHandler(ValidationHandler):
    def __init__(self, p: int = 1):
        assert p > 0
        super().__init__()
        self.p = p

    @property
    def main_name(self) -> str:
        """Main name of the validation handler"""
        return "HoldPOutValidation"


    @property
    def params_repr(self) -> str:
        """Representation of the parameters of the validation handler"""
        return str(self.p)
    
    def get_config(self):
        """Get the configuration of the validation handler: {"name": ..., "parameters": ...}"""
        config = {"name":"HoldPOutValidationHandler", 
                  "parameters":{"p":self.p}
        }
        return config

    def apply(self, mask:torch.Tensor) -> list[Tuple[torch.Tensor,torch.Tensor]]:
        """
        Takes the truth values of the mask and parcelates it. Create several partitions with p samples in validation each

        Parameters
        ----------
        mask : torch.Tensor

        Returns
        -------
        train_val_sets : list[Tuple[torch.Tensor,torch.Tensor]]
        """
        train_val_sets = []
        poss_val_ids = mask.clone()

        while poss_val_ids.sum() > 0:
            n_val_samples_per_partition = min(self.p,int(poss_val_ids.sum()))
            val_ids = torch.multinomial(poss_val_ids.to(torch.float64), n_val_samples_per_partition, replacement=False)

            new_train_mask = mask.clone().to(torch.bool)
            new_train_mask[val_ids] = False
            print("mask",mask)
            print("new_train_mask",new_train_mask)

            new_val_mask = torch.zeros(len(mask),dtype=torch.bool)
            new_val_mask[val_ids] = True

            train_val_sets.append((new_train_mask,new_val_mask))

            poss_val_ids[val_ids] = False
            
        return train_val_sets


class NoValidationHandler(ValidationHandler):
    @property
    def main_name(self) -> str:
        """Main name of the validation handler"""
        return "NoValidation"


    @property
    def params_repr(self) -> str:
        """Representation of the parameters of the validation handler"""
        return ""
    
    def get_config(self):
        """Get the configuration of the validation handler: {"name": ..., "parameters": ...}"""
        config = {"name":"NoValidationHandler", 
                  "parameters":{}
        }
        return config

    def apply(self, mask:torch.Tensor) -> list[Tuple[torch.Tensor,torch.Tensor]]:
        """
        Takes the truth values of the mask and parcelates it. Create several partitions with p samples in validation each

        Parameters
        ----------
        mask : torch.Tensor

        Returns
        -------
        train_val_sets : list[Tuple[torch.Tensor,torch.Tensor]]
        """
        new_train_mask = mask.clone()
        new_val_mask = torch.zeros(len(mask),dtype=torch.bool)
        train_val_sets = [(new_train_mask,new_val_mask)]
        return train_val_sets


# Pipeline

class PreprocessingPipeline():
    """
    Parameters
    ----------
    transformators
    """
    def __init__(
        self,
        complete_train_mask_selector: MaskSelector,
        transformators: list[EdgeTransformator|NodeTransformator],
        validation_handler:ValidationHandler,
        verbose: bool = True
    ):
        self.complete_train_mask_selector = complete_train_mask_selector
        self.transformators = transformators
        self.validation_handler = validation_handler
        self.verbose = verbose

    def get_train_val_sets(self,x):
        complete_train_mask = self.complete_train_mask_selector.apply(x)
        train_val_sets = self.validation_handler.apply(complete_train_mask)
        return train_val_sets
    
    def fit(self, 
            edge_index:np.ndarray, 
            edge_attr: pd.DataFrame | None = None, 
            x: pd.DataFrame | None = None, 
            y: pd.DataFrame | None = None,
            edge_locked: np.ndarray | None = None,):
        
        for transformator in self.transformators:
            print("PreprocessingPipeline fits",transformator)
            transformator.fit(
                edge_index=edge_index,
                edge_attr=edge_attr,
                x=x,
                y=y,
                edge_locked=edge_locked
            )
            # still need to transform for the next transformators that may depend on previous transformations
            edge_index,edge_attr,x,y, edge_locked = transformator.transform(
                edge_index=edge_index,
                edge_attr=edge_attr,
                x=x,
                y=y,
                edge_locked=edge_locked
            )
        
        return self

    def transform(self, 
                  edge_index:np.ndarray, 
                  edge_attr: pd.DataFrame | None = None, 
                  x: pd.DataFrame | None = None, 
                  y: pd.DataFrame | None = None,
                  edge_locked: np.ndarray | None = None) -> tuple[np.ndarray,pd.DataFrame|None,pd.DataFrame|None,pd.DataFrame|None]:
        
        for transformator in self.transformators:
            if self.verbose: 
                print("PreprocessingPipeline transforms data with",transformator)
                print("x.columns.to_list()",x.columns.to_list())
            
            edge_index,edge_attr,x,y = transformator.transform(
                edge_index=edge_index,
                edge_attr=edge_attr,
                x=x,
                y=y,
                edge_locked=edge_locked
            )
        
        if self.verbose: 
                print("PreprocessingPipeline transform done")
        return edge_index,edge_attr,x,y

    def fit_transform(self, 
                      edge_index:np.ndarray, 
                      edge_attr: pd.DataFrame | None = None, 
                      x: pd.DataFrame | None = None,
                      y: pd.DataFrame | None = None,
                      edge_locked: np.ndarray | None = None,) -> tuple[np.ndarray,pd.DataFrame|None,pd.DataFrame|None,pd.DataFrame|None]:
        self.fit(
            edge_index=edge_index,
            edge_attr=edge_attr,
            x=x,
            y=y,
            edge_locked=edge_locked
        )
        edge_index, edge_attr, x, y = self.transform(
            edge_index=edge_index,
            edge_attr=edge_attr,
            x=x,
            y=y,
            edge_locked=edge_locked
        )
        return edge_index,edge_attr,x,y,edge_locked
    
    

    def get_dataset_from_raw_data(self,
                                  edge_index:np.ndarray, 
                                  edge_attr: pd.DataFrame | None = None, 
                                  x: pd.DataFrame | None = None,
                                  y: pd.DataFrame | None = None,
                                  edge_locked: np.ndarray | None = None,) -> list[Data]:
        """
        Warning: complete_train_mask_selector is applied before any transformator
        """
        train_val_sets = self.get_train_val_sets(x)
        edge_index,edge_attr,x,y = self.fit_transform(edge_index=edge_index,
                                                      edge_attr=edge_attr,
                                                      x=x,
                                                      y=y,
                                                      edge_locked=edge_locked)
        
        dataset = []
        for i in range(len(train_val_sets)):
            data_graph = Data(
                x = torch.Tensor(x.values), 
                x_names = x.columns.tolist(),
                edge_index = torch.Tensor(edge_index).to(torch.int64),
                edge_attr = torch.Tensor(edge_attr.values) if not (edge_attr is None) else None,
                edge_attr_names = edge_attr.columns.tolist() if not (edge_attr is None) else None,
                y = torch.Tensor(y.values), 
                y_names = y.columns.tolist(),
                train_mask = torch.Tensor(train_val_sets[i][0]), 
                val_mask = torch.Tensor(train_val_sets[i][1]),
                edge_locked = torch.Tensor(edge_locked).to(torch.bool)
            )
            dataset.append(data_graph)
        
        return dataset

    def get_config(self):
        config = {
            "transformators": [],
            "complete_train_mask_selector": self.complete_train_mask_selector.get_config(),
            "validation_handler": self.complete_train_mask_selector.get_config()  
        }

        for transformator in self.transformators:
            config["transformators"].append(transformator.get_config())
        
        return config
