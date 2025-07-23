import time
import os

import pandas as pd
import torch

import src.data_handler as data_handler

from src.processing.preprocessing import PreprocessingPipeline, MaskThreshold, NoValidationHandler, KeepNodeFeaturesSelector
from src.models.model_pipeline import ModelPipeline

from src.models.kernel.diffusion_kernel import MaternKernelModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_or_test = "train" # could be a parsed argument in command line


if __name__ == "__main__":
    study_time_start = time.time()

    model_name = "3NN_3ExpNN_DiffusionKernel"
    study_name = pd.to_datetime("today").strftime("%Y-%m-%d_%H-%M_")+f"_{model_name}"
    study_folder_path = os.path.join(f"experiments_results/no_validation_07",train_or_test,study_name)
    study_raw_folder_path = os.path.join(study_folder_path,"raw")
    
    supplementary_config = {
        "model_name":model_name,
        "sim_used":"original",
        "model_fit_params":{}}
    
    participant_graph_names = os.listdir(os.path.join("data/temp_1_3NN_3ExpNN",train_or_test))
    for participant_i, participant_graph_name in enumerate(participant_graph_names):   
        participant_id = int(participant_graph_name.split("_")[-1])

        time_start_participant = time.time()
        print(f"start (participant_i+1)/(n_participants-1):{participant_i+1:d}/{len(participant_graph_names):d}")
        dst_folder_path = os.path.join(study_raw_folder_path,f"participant_{participant_id:d}")
        
        participant_graph = torch.load(os.path.join("data/temp_1_3NN_3ExpNN/",train_or_test,f"participant_graph_{participant_id}"),weights_only=False)

        if participant_graph is None:
            print(f"{participant_id} is None")
            continue
        

        validation_handler = NoValidationHandler()
        preprocessing_pipeline = PreprocessingPipeline(
            transformators=[
                KeepNodeFeaturesSelector(False,
                                         feature_names_kept=["liking"] # not used TODO change the MaternKernelModel for using x_base instead of y_base (careful homogeneous with GaussianProcess)
                                         )
            ],
            complete_train_mask_selector=MaskThreshold(feature_name="experience", threshold=0),
            validation_handler=validation_handler,
            base_mask_selector=MaskThreshold("experience", threshold=0, mode="strict_upper")
        )

        dim_in = 1

        complete_model = MaternKernelModel(
            dim_out=1,
            lengthscale=1.0, # will be fitted
            use_bias=False
        )

        # TODO this empty predict inside save config
        
        supplementary_config["model_fit_params"] = dict(
            complete_train_mask = participant_graph.complete_train_mask,
            edge_weight_name = "similarity",
            bounds = [(1e-3,1e2)],
            method = 'brute',
            Ns = 10000
        )

        model_pipeline = ModelPipeline(
            preprocessing_pipeline=preprocessing_pipeline,
            model=complete_model,
            dst_folder_path=dst_folder_path)

        model_pipeline.save_config(supplementary_config = {})
        graphs_dataset = model_pipeline.run_preprocessing(graph = participant_graph)
        print("graphs_dataset\n",graphs_dataset)
        model_pipeline.run_models(graphs_dataset=graphs_dataset, **supplementary_config["model_fit_params"])

        time_end_participant = time.time()

        print(f"end (participant_i+1)/n_participants-1:{participant_i+1:d}/{len(participant_graph_names):d}")
        participant_time_taken = time.time()-time_start_participant
        print("participant time taken: {}h - {}m - {}s".format(participant_time_taken//(3600),((participant_time_taken%3600)//60),((participant_time_taken%3600)%60)))
    

    study_time_taken = time.time()-study_time_start
    print("study time taken: {}h - {}m - {}s".format(study_time_taken//(3600),((study_time_taken%3600)//60),((study_time_taken%3600)%60)))
    print("data_handler.postprocess to ",os.path.join(study_folder_path,"processed"))
    data_handler.postprocess(study_raw_folder_path,os.path.join(study_folder_path,"processed"))