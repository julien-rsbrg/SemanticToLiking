"""
This data is generated only for the post analysis of the models

"""


import os
import pandas as pd 
import numpy as np

from src.utils import recursive_mkdirs,save_yaml

def run():
    dst_folder_path = "src/data_generation/examples/raw/study_2"

    for participant_id in range(6):
        participant_folder_path = os.path.join(dst_folder_path,f"participant_{participant_id}")
        recursive_mkdirs(participant_folder_path)

        config = {
            "model_name":"my_model_2",
            "a":np.random.randn()*2 + 5,
            "b":np.random.randn()*1 + 7,
            "c":np.random.randint(0,10),
            "d":np.random.randint(-5,4)}
        save_yaml(config,dst_path=os.path.join(participant_folder_path,"config.yml"))

        for graph_id in range(4):
            graph_folder_path = os.path.join(participant_folder_path,f"graph_{graph_id}")
            recursive_mkdirs(graph_folder_path)

            history = {"epoch":np.arange(0,100),
                       "val_mae":np.random.randn(100),
                       "train_mae":np.random.randn(100)}
            pd.DataFrame(history).to_csv(os.path.join(graph_folder_path,"history.csv"))

            prediction_table = {
                "pred_values":np.random.randn(40),
                "true_values":np.random.randn(40),
                "train_mask":np.random.randint(2,size=40,dtype=bool)
            }
            prediction_table["val_mask"] = ~prediction_table["train_mask"]
            prediction_table = pd.DataFrame(prediction_table)
            prediction_table.to_csv(os.path.join(graph_folder_path,"prediction_table.csv"))

            model_params_init = {
                "param_0":[np.random.randn()],
                "param_1":[np.random.randn()*2+1]
            }
            model_params_init = pd.DataFrame(model_params_init)
            model_params_init.to_csv(os.path.join(graph_folder_path,"model_params_init.csv"))

            model_params_trained = model_params_init = {
                "param_0":[np.random.randn()*.1],
                "param_1":[np.random.randn()*.2+1]
            }
            model_params_trained = pd.DataFrame(model_params_trained)
            model_params_trained.to_csv(os.path.join(graph_folder_path,"model_params_trained.csv"))


if __name__ == "__main__":
    run()

