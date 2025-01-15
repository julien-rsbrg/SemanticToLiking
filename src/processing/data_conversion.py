import numpy as np
import pandas as pd

import torch
import torch_geometric.data 

from abc import ABC, abstractmethod
from typing import List
import copy

# utils functions


# Converter
class DataConverter(ABC):    
    @abstractmethod
    def transform():
        pass

class ConverterDataFrameToTorchData(DataConverter):
    def __init__(self,
                 node_attr_names:list[str],
                 edge_attr_names:list[str],
                 edge_name_id_sender_receiver:list[str],
                 node_label_names:list[str]|None = None,
                 verbose:bool = True
                 ):
        self.node_attr_names = node_attr_names
        self.edge_attr_names = edge_attr_names
        self.edge_name_id_sender_receiver = edge_name_id_sender_receiver
        self.node_label_names = node_label_names
        self.verbose = verbose

    def transform(self, node_table:pd.DataFrame, edge_table:pd.DataFrame):
        node_attr = node_table[self.node_attr_names].values
        node_attr = torch.Tensor(node_attr)
        
        edge_index = edge_table[self.edge_name_id_sender_receiver]
        edge_index = torch.Tensor(edge_index.to_numpy()).to(torch.int64)
        edge_index = edge_index.T
        reversed_edge_index = edge_index[[1,0],:]
        edge_index = torch.concat([edge_index,reversed_edge_index],dim=1)

        edge_attr = edge_table[self.edge_attr_names].values # n edges, n edge attr 
        edge_attr = torch.Tensor(edge_attr)
        edge_attr = torch.concat([edge_attr,edge_attr],dim=0)

        node_train_mask = torch.ones(len(node_attr),dtype=torch.bool)

        node_labels = None
        if not(self.node_label_names is None):
            node_labels = node_table[self.node_label_names].values
            node_labels = torch.Tensor(node_labels)

        data_graph = torch_geometric.data.Data(
            x = node_attr, 
            x_names = self.node_attr_names,
            edge_index = edge_index,
            edge_attr = edge_attr,
            edge_attr_names = self.edge_attr_names,
            y = node_labels, 
            y_names = self.node_label_names,
            train_mask = torch.Tensor(node_train_mask), 
            val_mask = torch.Tensor(~node_train_mask)
            )

        
        if self.verbose:
            # check data
            print("Test function convert_table_to_graph")
            print(data_graph)
            print("validate:",data_graph.validate())
            print("is undirected:", data_graph.is_undirected())

            has_self_loop = min(abs(data_graph.edge_index[0]-data_graph.edge_index[1])) < 1
            print("has_self_loop:",has_self_loop)
            print("end Test function convert_table_to_graph")
        
        return data_graph
    

class ConverterTorchDataToDataFrame(DataConverter):
    def __init__(self,
                 verbose:bool = True):
        self.verbose = verbose

    def transform(self,data:torch_geometric.data.Data) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
        node_label_names = data.x_names
        node_table = pd.DataFrame(data.x,columns=data.x_names)
        if not (data.y_names is None):
            y = pd.DataFrame(data.y,columns=data.y_names)
            node_table = pd.concat([node_table,y],axis=1)
        node_label_names = node_table.columns.tolist()
        
        edge_name_id_sender_receiver = ["sender_id","receive_id"]
        edge_table = pd.DataFrame(data.edge_index.T,columns=edge_name_id_sender_receiver)
        
        return node_table, edge_table, edge_name_id_sender_receiver, node_label_names
