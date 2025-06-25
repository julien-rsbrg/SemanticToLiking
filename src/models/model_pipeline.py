import os
import numpy as np
import pandas as pd

from torch_geometric.data import Data

from src.processing.preprocessing import PreprocessingPipeline
from src.models.generic_model import GenericModel
from src.utils import recursive_mkdirs, read_yaml, save_yaml, turn_dict_values_iterable_to_list

class ModelPipeline():
    def __init__(self,
                 preprocessing_pipeline:PreprocessingPipeline,
                 model:GenericModel,
                 dst_folder_path:str):
        self.preprocessing_pipeline = preprocessing_pipeline
        self.model = model
        self.dst_folder_path = dst_folder_path

        recursive_mkdirs(self.dst_folder_path)


    def save_config(self,supplementary_config,file_name = "config.yml"):
        """
        Recommended to pass the kwargs parameters given to run_models inside the supplementary_config
        """
        config = {}
        config["preprocessing_pipeline"] = self.preprocessing_pipeline.get_config()
        config["model"] = self.model.get_config()
        config["other"] = supplementary_config
        config = turn_dict_values_iterable_to_list(config)
        save_yaml(config, dst_path=os.path.join(self.dst_folder_path,file_name))


    def run_preprocessing(self,graph:Data):
        graphs_dataset = self.preprocessing_pipeline.get_dataset_from_raw_data(
            edge_index = graph.edge_index.numpy(), 
            edge_attr = pd.DataFrame(graph.edge_attr, columns = graph.edge_attr_names),
            x = pd.DataFrame(graph.x, columns = graph.x_names),
            y = pd.DataFrame(graph.y, columns = graph.y_names)
        )
        return graphs_dataset
    

    def run_models(self, graphs_dataset, **kwargs):
        for graph_id, graph in enumerate(graphs_dataset):
            subfolder_graph_path = os.path.join(self.dst_folder_path,f"graph_{graph_id}")
            recursive_mkdirs(subfolder_graph_path)

            self.model.save(os.path.join(subfolder_graph_path,"model_init"))
            self.model.save_parameters(os.path.join(subfolder_graph_path,"model_params_init"))

            history = self.model.fit(dataset=[graph],**kwargs)
            history = pd.DataFrame(history)
            history.to_csv(os.path.join(subfolder_graph_path,"history.csv"))

            pred_values = self.model.predict(node_attr=graph.x, 
                                             edge_index=graph.edge_index, 
                                             edge_attr=graph.edge_attr,
                                             **kwargs)
            true_values = graph.y

            prediction_table = {"pred_values":pred_values.detach().cpu().numpy().flatten(),
                                "true_values":true_values.detach().cpu().numpy().flatten(),
                                "train_mask":graph.train_mask.detach().cpu().numpy().flatten(),
                                "val_mask":graph.val_mask.detach().cpu().numpy().flatten()}
            
            prediction_table = pd.DataFrame(prediction_table)
            prediction_table.to_csv(os.path.join(subfolder_graph_path,"prediction_table.csv"))
            
            self.model.save(os.path.join(subfolder_graph_path,"model_trained"))
            self.model.save_parameters(os.path.join(subfolder_graph_path,"model_params_trained"))
            
            self.model.reset_parameters()
    
    def predict(self,graph:Data, data_state:str = "raw"):
        """
        Parameters
        ----------
        data_state : (str)
        - keep in ["raw","preprocessed"]
        """
        assert data_state in ["raw","preprocessed"], data_state
        if data_state == "raw":
            graph = self.run_preprocessing(graph)[0]
        pred_values = self.model.predict(node_attr=graph.x, 
                                         edge_index=graph.edge_index, 
                                         edge_attr=graph.edge_attr)
        
        return pred_values