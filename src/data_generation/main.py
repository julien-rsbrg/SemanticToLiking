import os
import numpy as np
import pandas as pd

import torch
from torch_geometric.data import Data

from src.processing.preprocessing import PreprocessingPipeline, KeepFeatureNamedSelector, KeepGroupSendersToGroupReceivers, MaskKeepQuantile, CrossValidationHandler, NoValidationHandler
from src.utils import recursive_mkdirs, read_yaml, save_yaml

from src.models.model_pipeline import ModelPipeline
from src.models.baseline_models import SimpleConvModel



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

        data_graph = Data(
            x = torch.randn(n_nodes,len(self.x_var_names)), 
            x_names = self.x_var_names,
            edge_index = torch.Tensor(edge_index).to(torch.int64),
            edge_attr = torch.randn(n_nodes**2,len(self.edge_attr_names)),
            edge_attr_names = self.edge_attr_names,
            y = torch.randn(n_nodes,len(self.y_var_names)), 
            y_names = self.y_var_names
            )

        return data_graph


    def include_prediction(self, graphs_dataset:list[Data]):
        new_graphs_dataset = []
        for graph in graphs_dataset:
            new_graph = graph.clone()
            pred_values = self.model_pipeline.predict(new_graph,data_state="preprocessed")
            new_graph.y = pred_values
            new_graphs_dataset.append(new_graph)

        return new_graphs_dataset


    def run(self):
        for i in range(self.n_graphs):
            print(f"\n-- Generate graph {i+1}/{self.n_graphs} --")
            graph = self.create_random_fully_connected_graph()
            graphs_dataset = self.model_pipeline.run_preprocessing(graph=graph)
            assert len(graphs_dataset) == 1, graphs_dataset
            
            graphs_dataset = self.include_prediction(graphs_dataset)

            torch.save(graphs_dataset[0], f = os.path.join(self.dst_folder_path,f"graph_{i}"))



def load_generated_dataset(src_folder_path):
    graphs_dataset = []
    for f in os.listdir(src_folder_path):
        if f.split("_")[0] == "graph":
            src_file_path = os.path.join(src_folder_path,f)
            graphs_dataset.append(torch.load(src_file_path,weights_only=False))
    return graphs_dataset



if __name__ == "__main__":
    dst_folder_path = "src/data_generation/generated/example"

    preprocessing_pipeline = PreprocessingPipeline(
            transformators=[
                KeepGroupSendersToGroupReceivers(
                    group_senders_mask_fn= lambda x,q: x["experience"] > q,
                    group_receivers_mask_fn= lambda x,q: x["experience"] <= q,
                    group_senders_thresholding=True,
                    group_receivers_thresholding=True,
                    group_senders_mask_threshold_fn=lambda x: float(torch.quantile(torch.Tensor(x["experience"]),0.33)),
                    group_receivers_mask_threshold_fn=lambda x: float(torch.quantile(torch.Tensor(x["experience"]),0.33))
                ),
                KeepFeatureNamedSelector(verbose=True,feature_names_kept=["liking"])
            ],
            complete_train_mask_selector=MaskKeepQuantile(feature_name="experience",q=0.33,mode="lower"),
            validation_handler=NoValidationHandler()
        )


    model = SimpleConvModel(aggr="min")

    model_pipeline = ModelPipeline(
        preprocessing_pipeline=preprocessing_pipeline,
        model=model,
        dst_folder_path=dst_folder_path
    )
    data_generator = DataGenerator(
        model_pipeline = model_pipeline,
        dst_folder_path = dst_folder_path,
        n_nodes = 60,
        n_graphs = 5,
        x_var_names = ["liking","experience"],
        y_var_names = ["liking"],
        edge_attr_names = ["similarity"]
    )
    data_generator.save_config(supplementary_config_generator={},supplementary_config_model={"model_name":"super_model"})
    data_generator.run()

    print("== example load ==")
    graphs_dataset = load_generated_dataset(dst_folder_path)
    print("graphs_dataset\n",graphs_dataset)

    print("graphs_dataset[0].x",graphs_dataset[0].x)
    print("graphs_dataset[0].y",graphs_dataset[0].y)
    print("ratio", (graphs_dataset[0].y != 0).sum()/len(graphs_dataset[0].y))
    print("graphs_dataset[0].edge_index",graphs_dataset[0].edge_index)
    
    """
    import src.visualization.display_graph as display_graph
    display_graph.draw_torch_graph(data=graphs_dataset[0],node_colors=graphs_dataset[0].x,node_color_label="x")
    """