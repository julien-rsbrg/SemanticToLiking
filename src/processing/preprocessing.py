import numpy as np
import pandas as pd

import torch
import torch_geometric.data 

from abc import ABC, abstractmethod
from typing import List
import copy

# Edge Selectors

class EdgeSelector(ABC):
    """
    Operate on the connections of a graph to remove them 

    (Respect sklearn BaseEstimator and TransformerMixin baseline)

    (Machine library agnostic)

    Attributes
    ----------
    input_types : list[str]
        --
    verbose : bool
        --
    feature_names_in_ : list[str]
        --
    feature_names_out_ : list[str]
        --

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
        return self.name() 


    def _check_types(self, edge_index: np.ndarray, edge_attr: pd.DataFrame | None = None, x: pd.DataFrame | None = None,  y: pd.DataFrame | None = None):
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



    def fit(self, edge_index: np.ndarray, edge_attr: pd.DataFrame | None = None, x: pd.DataFrame | None = None,  y: pd.DataFrame | None = None):
        """
        Receive a graph and fit instantiation's attributes for edge selection.

        Parameters
        ----------

        Returns
        -------

        """
        self._check_types(edge_index,edge_attr,x,y)

        return self

    def transform(self, edge_index: np.ndarray, edge_attr: pd.DataFrame | None = None, x: pd.DataFrame | None = None,  y: pd.DataFrame | None = None) -> tuple[np.ndarray,pd.DataFrame|None,pd.DataFrame|None,pd.DataFrame|None]:
        """Directly returns the best features from what was found during fit and that are present in X.

        Parameters
        ----------
        x : 
            --

        Returns
        -------
        """
        self._check_types(edge_index,edge_attr,x,y)

        return edge_index, edge_attr, x, y

    def fit_transform(
        self, edge_index: np.ndarray, edge_attr: pd.DataFrame | None = None, x: pd.DataFrame | None = None,  y: pd.DataFrame | None = None
    ) -> tuple[np.ndarray,pd.DataFrame|None,pd.DataFrame|None,pd.DataFrame|None]:
        """Receive new graph and remove some of its edges 

        Parameters
        ----------

        Returns
        -------
        """
        self._check_types(edge_index,edge_attr,x,y)

        self.fit(edge_index,edge_attr,x,y)
        new_edge_index, edge_attr, x, y = self.transform(edge_index,edge_attr,x,y)
        return new_edge_index, edge_attr, x, y


class CutGroupSendersToGroupReceivers(EdgeSelector):
    def __init__(
        self,
        group_senders_mask_fn: callable,
        group_receivers_mask_fn: callable,
        verbose: bool = True
    ):
        super().__init__(verbose)
        self.group_senders_mask_fn = group_senders_mask_fn
        self.group_receivers_mask_fn = group_receivers_mask_fn
    

    @property
    def main_name(self) -> str:
        """Main name of the transform function"""
        return "CutGroupSendersToGroupReceivers"
    
    @property
    def params_repr(self) -> str:
        """Representation of the parameters of the transform function"""
        return ""

    @property
    def name(self) -> str:
        """Name of the transform function used for authentificate"""
        return self.main_name + self.params_repr
    
    def transform(self, 
                  edge_index:np.ndarray, 
                  edge_attr: pd.DataFrame | None = None, 
                  x: pd.DataFrame | None = None, 
                  y: pd.DataFrame | None = None) -> tuple[np.ndarray,pd.DataFrame|None,pd.DataFrame|None,pd.DataFrame|None]:
        super().transform(edge_index, edge_attr, x, y)
        is_sender_selected = self.group_senders_mask_fn(x)
        is_receiver_selected = self.group_receivers_mask_fn(x)
        
        if self.verbose:
            print("Number of Edges before transform:", edge_index.shape[1])
                  
        is_edge_selected = np.zeros(edge_index.shape,dtype=bool)
        is_edge_selected[0,:] = is_sender_selected[edge_index[0,:]]
        is_edge_selected[1,:] = is_receiver_selected[edge_index[1,:]]
        is_edge_selected = is_edge_selected[0,:] * is_edge_selected[1,:]

        new_edge_index = copy.deepcopy(edge_index)[:,~is_edge_selected]
        new_edge_attr = copy.deepcopy(edge_attr).iloc[~is_edge_selected]
        new_x = copy.deepcopy(x)
        new_y = copy.deepcopy(y)

        if self.verbose:
            print("Number of Edges after transform:", new_edge_index.shape[1])

        return new_edge_index, new_edge_attr, new_x, new_y


class KeepKNearestNeighbors(EdgeSelector):
    def __init__(
        self,
        k: int,
        edge_attr_names_used: list[str],
        ord: int = 1,
        verbose: bool = True
    ):
        super().__init__(verbose)
        self.k = k
        self.edge_attr_names_used = edge_attr_names_used
        self.ord = ord

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
    
    def transform(self, 
                  edge_index:np.ndarray, 
                  edge_attr: pd.DataFrame | None = None, 
                  x: pd.DataFrame | None = None, 
                  y: pd.DataFrame | None = None) -> tuple[np.ndarray,pd.DataFrame|None,pd.DataFrame|None,pd.DataFrame|None]:
        super().transform(edge_index, edge_attr, x, y)

        if self.verbose:
            print("Number of Edges before transform:", edge_index.shape[1])

        if len(self.edge_attr_names_used) >= 1:
            edge_dist = np.linalg.norm(edge_attr[self.edge_attr_names_used], ord=self.ord, axis=1)
            edge_kept = np.zeros(edge_index.shape[1],dtype=bool)
            
            # from receiver point of view
            for node_id in np.unique(edge_index[1,:]):
                ids_node_receiver = np.where(edge_index[1,:] == node_id)[0]
                node_dist_from_sender = edge_dist[ids_node_receiver]
                kept_k_neigbors_edges = np.argsort(node_dist_from_sender)[:self.k]
                edge_kept[ids_node_receiver[kept_k_neigbors_edges]] = True
            
        else:
            edge_kept = np.zeros(edge_index.shape[1],dtype=bool)
        
        new_edge_index = copy.deepcopy(edge_index)[:,edge_kept]
        new_edge_attr = copy.deepcopy(edge_attr).iloc[edge_kept]
        new_x = copy.deepcopy(x)
        new_y = copy.deepcopy(y)
        
        if self.verbose:
            print("Number of Edges after transform:", new_edge_index.shape[1])

        return new_edge_index, new_edge_attr, new_x, new_y

# Pipeline

class PreprocessingPipeline():
    """
    restricted to edge selection for now
    
    """
    def __init__(
        self,
        edge_selectors: list[EdgeSelector],
        verbose: bool = True
    ):
        self.edge_selectors = edge_selectors
        self.verbose = verbose

    def fit(self, 
            edge_index:np.ndarray, 
            edge_attr: pd.DataFrame | None = None, 
            x: pd.DataFrame | None = None, 
            y: pd.DataFrame | None = None):
        
        for edge_selector in self.edge_selectors:
            print("PreprocessingPipeline fits",edge_selector)
            edge_selector.fit(
                edge_index=edge_index,
                edge_attr=edge_attr,
                x=x,
                y=y
            )
        
        return self

    def transform(self, 
                  edge_index:np.ndarray, 
                  edge_attr: pd.DataFrame | None = None, 
                  x: pd.DataFrame | None = None, 
                  y: pd.DataFrame | None = None) -> tuple[np.ndarray,pd.DataFrame|None,pd.DataFrame|None,pd.DataFrame|None]:
        for edge_selector in self.edge_selectors:
            if self.verbose: 
                print("PreprocessingPipeline transforms data with",edge_selector)
            
            edge_index,edge_attr,x,y = edge_selector.transform(
                edge_index=edge_index,
                edge_attr=edge_attr,
                x=x,
                y=y
            )
        
        if self.verbose: 
                print("PreprocessingPipeline transform done")
        return edge_index,edge_attr,x,y

    def fit_transform(self, 
                      edge_index:np.ndarray, 
                      edge_attr: pd.DataFrame | None = None, 
                      x: pd.DataFrame | None = None,
                      y: pd.DataFrame | None = None) -> tuple[np.ndarray,pd.DataFrame|None,pd.DataFrame|None,pd.DataFrame|None]:
        self.fit(
            edge_index=edge_index,
            edge_attr=edge_attr,
            x=x,
            y=y
        )
        edge_index, edge_attr, x, y = self.transform(
            edge_index=edge_index,
            edge_attr=edge_attr,
            x=x,
            y=y
        )
        return edge_index,edge_attr,x,y