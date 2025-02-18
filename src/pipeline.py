import copy
import numpy as np
import pandas as pd

from typing import Tuple, List

from src.processing.preprocessing import EdgeSelector

class GeneralizerRun:
    """
    Complete generalizing model from graphs
    """
    def __init__(
            self,
            model,
            graph_preprocessing: list = [],
            dataset_preprocessing: list = []):
        self.model = model
        self.graph_preprocessing = graph_preprocessing
        self.dataset_preprocessing = dataset_preprocessing

        # for reset
        self._graph_preprocessing_unfitted = copy.deepcopy(self.graph_preprocessing)
        self._dataset_preprocessing_unfitted = copy.deepcopy(self.dataset_preprocessing)
        self._model_unfitted = copy.deepcopy(self.model)
    
    # TODO: all the graph preprocessors should be standardized for the same starting inputs (x,edge_index,edge_attr,y)
    def preprocess_graph(self,graph):
        new_graph = copy.deepcopy(graph)
        for i,preprocessing in enumerate(self.graph_preprocessing):
            new_graph = preprocessing.fit_transform(*new_graph)
            if not(isinstance(new_graph,tuple)) and i != len(self.graph_preprocessing)-1 :
                new_graph = (new_graph,)
        
        return new_graph

    def generate_dataset(self,graphs):
        dataset = copy.deepcopy(graphs)
        for preprocessing in self.dataset_preprocessing:
            dataset = preprocessing.fit_transform(dataset)
        return dataset
    
    def generate_dataset_from_scratch(self,data:list[tuple[pd.DataFrame]],):
        raw_dataset = []
        n_graphs = len(data)
        for i in range(n_graphs):
            graph = data[i]
            graph = self.preprocess_graph(graph)
            raw_dataset.append(graph)
        
        dataset = self.generate_dataset(raw_dataset)
        return dataset

    def fit_model(self,
                  data:list[tuple[pd.DataFrame]],
                  val_data: list[tuple[pd.DataFrame]]= None,
                  **fit_model_kwargs):
        
        dataset = self.generate_dataset_from_scratch(data)
        
        val_dataset = None
        if not(val_data is None):
            val_dataset = self.generate_dataset_from_scratch(val_data) 
        
        _,history = self.model.fit(dataset = dataset,val_dataset = val_dataset,**fit_model_kwargs)
        return self.model,history
    
    def predict(self,
                data:tuple[pd.DataFrame],
                **kwargs):
        
        # TODO: standardize model.predict to x,edge_index,edge_attr,y
        processed_data = self.preprocess_graph(data)
        if isinstance(processed_data,tuple):
            preds = self.model.predict(*processed_data,**kwargs)
        else:
            preds = self.model.predict(processed_data,**kwargs)
        return preds


    def reset(self):
        """Reset GeneralizerRun instance to the initial state (in place)

        Returns
        -------
        self : GeneralizerRun
            The reset generalizer run
        """
        self.graph_preprocessing = copy.deepcopy(self._graph_preprocessing_unfitted)
        self.dataset_preprocessing = copy.deepcopy(self._dataset_preprocessing_unfitted)
        self.model = copy.deepcopy(self._model_unfitted)
        return self