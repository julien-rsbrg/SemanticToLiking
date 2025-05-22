import os
import pandas as pd

from src.processing.preprocessing import PreprocessingPipeline
from src.models.generic_model import GenericModel
from src.utils import recursive_mkdirs, read_yaml, save_yaml

class ModelPipeline():
    def __init__(self,
                 preprocessing_pipeline:PreprocessingPipeline,
                 model:GenericModel,
                 dst_folder_path:str):
        self.preprocessing_pipeline = preprocessing_pipeline
        self.model = model
        self.dst_folder_path = dst_folder_path

        recursive_mkdirs(self.dst_folder_path)


    def save_config(self,supplementary_config):
        """
        Recommended to pass the kwargs parameters given to run_models inside the supplementary_config
        """
        config = {}
        config["preprocessing_pipeline"] = self.preprocessing_pipeline.get_config()
        config["model"] = self.model.get_config()
        config["other"] = supplementary_config
        save_yaml(config, dst_path=os.path.join(self.dst_folder_path,"config.yml"))


    def run_preprocessing(self):
        graphs_dataset = self.preprocessing_pipeline.get_dataset_from_raw_data()
        return graphs_dataset


    def run_models(self,graphs_dataset, **kwargs):
        for graph_id, graph in enumerate(graphs_dataset):
            subfolder_graph_path = os.path.join(self.dst_folder_path,f"graph_{graph_id}")
            recursive_mkdirs(subfolder_graph_path)

            self.model.save(os.path.join(subfolder_graph_path,"model_init"))

            history = self.model.fit(dataset=[graph],**kwargs)
            history = pd.DataFrame(history)
            history.to_csv(os.path.join(subfolder_graph_path,"history.csv"))

            pred_values = self.model.predict(node_attr=graph.x, 
                                             edge_index=graph.edge_index, 
                                             edge_attr=graph.edge_attr)
            true_values = graph.y

            prediction_table = {"pred_values":pred_values.cpu().numpy(),
                                "true_values":true_values.cpu().numpy(),
                                "train_mask":graph.train_mask.cpu().numpy(),
                                "val_mask":graph.val_mask.cpu().numpy()}
            prediction_table = pd.DataFrame(prediction_table)
            prediction_table.to_csv(os.path.join(subfolder_graph_path,"prediction_table.csv"))
            
            self.model.save(os.path.join(subfolder_graph_path,"model_trained"))
            self.model.reset_parameters()