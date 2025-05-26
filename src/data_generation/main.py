import os
import numpy as np
import pandas as pd

import torch
from torch_geometric.data import Data

from src.processing.preprocessing import PreprocessingPipeline
from src.models.generic_model import GenericModel
from src.utils import recursive_mkdirs, read_yaml, save_yaml

from src.models.model_pipeline import ModelPipeline
from src.processing.preprocessing import NoValidationHandler


class DataGenerator():
    def __init__(self,
                 model_pipeline:ModelPipeline,
                 dst_folder_path:str,
                 n_nodes:int|tuple[int],
                 n_graphs:int,
                 x_var_names:list[str],
                 y_var_names:list[str],
                 edge_attr_names:list[str]
                 ):
        """
        dst_folder_path overwrites dst_folder_path in model_pipeline

        NoValidationHandler is overwritten on model_pipeline.preprocessing_pipeline.validation_handler

        Parameters
        ----------
        n_nodes: (int | tuple[int])
        - if tuple, samples from [n_nodes[0],n_nodes[1]) to decide how many nodes to include in the graph


        """
        model_pipeline.preprocessing_pipeline.validation_handler = NoValidationHandler()

        self.model_pipeline = model_pipeline
        self.model_pipeline.dst_folder_path = dst_folder_path

        self.dst_folder_path = dst_folder_path
        recursive_mkdirs(self.dst_folder_path)

        self.n_nodes = n_nodes
        if isinstance(self.n_nodes,int):
            self.n_nodes = (self.n_nodes,self.n_nodes+1) 

        self.n_graphs = n_graphs
        self.x_var_names = x_var_names
        self.y_var_names = y_var_names
        self.edge_attr_names = edge_attr_names

    
    def save_config(self,supplementary_config_model,supplementary_config_generator, file_name = "data_generator_config.yml"):
        self.model_pipeline.save_config(supplementary_config=supplementary_config_model,file_name = "model_pipeline_config.yml")

        config = {}
        config["other"] = supplementary_config_generator
        save_yaml(config, dst_path=os.path.join(self.dst_folder_path,file_name))


    def create_random_fully_connected_graph(self):
        n_nodes = np.random.randint(self.n_nodes[0],self.n_nodes[1])
        raw_ids = np.arange(n_nodes)[...,np.newaxis].repeat(n_nodes,axis=1)
        edge_index = np.stack([raw_ids.flatten(),raw_ids.flatten(order="F")])

        edge_attr = pd.DataFrame(np.random.randn(n_nodes,len(self.edge_attr_names)),columns=self.edge_attr_names)
        x = pd.DataFrame(np.random.randn(n_nodes,len(self.x_names)),columns=self.x_names)
        y = pd.DataFrame(np.random.randn(n_nodes,len(self.y_names)),columns=self.y_names)

        return edge_index,edge_attr,x,y


    def include_prediction(self, graphs_dataset):
        new_graphs_dataset = []
        for graph in graphs_dataset:
            new_graph = graph.clone()
            pred_values = self.model.predict(node_attr=graph.x, 
                                             edge_index=graph.edge_index, 
                                             edge_attr=graph.edge_attr)
            new_graph.y = pred_values
            new_graphs_dataset.append(new_graph)

        return new_graphs_dataset


    def run(self):
        for i in range(self.n_graphs):
            graph = self.create_random_fully_connected_graph()
            graphs_dataset = self.model_pipeline.run_preprocessing(graph=graph)
            assert len(graphs_dataset) == 1, graphs_dataset
            
            graphs_dataset = self.include_prediction(graphs_dataset)

            torch.save(graphs_dataset, f = os.path.join(self.dst_folder_path,f"graph_{i}"))



def load_generated_dataset(src_folder_path):
    graphs_dataset = []
    for f in os.listdir(src_folder_path):
        if f.split("_")[0] == "graph":
            src_file_path = os.path.join(src_folder_path,f)
            graphs_dataset.append(torch.load(src_file_path))
    return graphs_dataset



