import os

from src.processing.preprocessing import PreprocessingPipeline
from src.models.generic_model import GenericModel

class ModelPipeline():
    def __init__(self,
                 preprocessing_pipeline:PreprocessingPipeline,
                 model:GenericModel,
                 dst_folder_path:str):
        self.preprocessing_pipeline = preprocessing_pipeline
        self.model = model
        self.dst_folder_path = dst_folder_path

    def run_preprocessing(self):
        graphs_dataset = self.preprocessing_pipeline.get_dataset_from_raw_data()
        return graphs_dataset

    def run_models(self,graphs_dataset, **kwargs):
        predictions = {"pred_values":[],"true_values":[],"train_mask":[],"val_mask":[]}
        histories = []
        models = []
        for graph_id, graph in enumerate(graphs_dataset):
            new_dst_folder_path = os.path.join()

            histories.append(self.model.fit(dataset=[graph],**kwargs))

            pred_values = self.model.predict(node_attr=graph.x, 
                                             edge_index=graph.edge_index, 
                                             edge_attr=graph.edge_attr)
            true_values = graph.y
            predictions["pred_values"].append(pred_values.cpu().numpy())
            predictions["true_values"].append(true_values.cpu().numpy())
            predictions["train_mask"].append(graph.train_mask.cpu().numpy())
            predictions["val_mask"].append(graph.val_mask.cpu().numpy())
            
            # TODO: save model
            
            self.model.save()
            self.model.reset()

        return 


    def save(self):
        pass