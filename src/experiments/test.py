import time
import os
import copy

import pandas as pd
import numpy as np
import torch

import src.data_handler as data_handler

from src.processing.raw_data_cleaning import prepare_graph_for_participant
from src.processing.preprocessing import PreprocessingPipeline, SeparatePositiveNegative, PolynomialFeatureGenerator, KeepNodeFeaturesSelector, FilterGroupSendersToGroupReceivers, KeepKNearestNeighbors, MaskThreshold, NoValidationHandler, CrossValidationHandler
from src.models.model_pipeline import ModelPipeline
from src.models.baseline_models import SimpleConvModel
from src.models.nn.gnn_layers import MyGATConv, MyGATConvNLeaps
from src.models.nn.ML_frameworks import GNNFramework

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    study_time_start = time.time()

    model_name = "test"
    study_name = pd.to_datetime("today").strftime("%Y-%m-%d_%H-%M_")+f"_{model_name}"
    study_folder_path = os.path.join("experiments_results/no_validation",study_name)
    study_raw_folder_path = os.path.join(study_folder_path,"raw")
    
    supplementary_config = {
        "model_name":model_name,
        "sim_used":"original",
        "model_fit_params":{}}

    participant_indices = np.arange(1,113)
    for participant_id in participant_indices:
        time_start_participant = time.time()
        print(f"start participant_id/(n_participants-1):{participant_id:d}/{len(participant_indices-1):d}")
        dst_folder_path = os.path.join(study_raw_folder_path,f"participant_{participant_id:d}")
        
        participant_graph = torch.load(f"data/temp_3NN_3ExpNN/participant_graph_{participant_id}",weights_only=False)

        if participant_graph is None:
            print(f"{participant_id} is None")
            continue

        validation_handler = NoValidationHandler()
        preprocessing_pipeline = PreprocessingPipeline(
            transformators=[],
            complete_train_mask_selector=MaskThreshold(feature_name="experience",threshold=0),
            validation_handler=validation_handler,
            base_mask_selector=None
        )

        dim_in = 1

        src_content_mask = torch.Tensor([True]*dim_in).to(torch.bool)
        src_edge_mask = torch.Tensor([True]*dim_in).to(torch.bool)
        dst_content_mask = torch.Tensor([False]*dim_in).to(torch.bool)
        dst_edge_mask = torch.Tensor([False]*dim_in).to(torch.bool)
        my_module = MyGATConvNLeaps(
            n_leaps = 2, # I know it's the max for this dataset
            in_channels=(dim_in,dim_in),
            out_channels=1,
            heads=1,
            negative_slope=1.0,
            add_self_loops=False,
            edge_dim=1,
            dropout=0.0,
            bias=True,
            src_content_mask=src_content_mask,
            src_edge_mask=src_edge_mask,
            dst_content_mask=dst_content_mask,
            dst_edge_mask=dst_edge_mask,
            src_content_require_grad=True,
            edge_require_grad=True,
            src_content_weight_initializer="glorot",
            edge_weight_initializer="glorot",
            src_content_edge_are_same = True,
            dst_content_edge_are_same = False
        )

        ## Training
        complete_model = GNNFramework(my_module,device)

        # TODO this empty predict inside save config

        complete_model.predict(torch.Tensor([[0]*dim_in,[1]*dim_in,[2]*dim_in]),
                                torch.Tensor([[0],
                                              [1]]).to(torch.int64),
                                torch.Tensor([[1]]),
                                complete_train_mask = [False,True,False])

        opt = complete_model.configure_optimizer(lr=1e-2)
        scheduler = complete_model.configure_scheduler(opt,0.1,0.1,10)
        weight_constrainer = None # complete_model.configure_weight_constrainer("clipper",0,100)
            

        supplementary_config["model_fit_params"] = dict(
            epochs = 10000,
            report_epoch_steps = 1,
            optimizer = opt,
            scheduler = scheduler,
            weight_constrainer = weight_constrainer,
            early_stopping_monitor="train_mae",
            patience=500,
            l2_reg=1e-2,
            complete_train_mask = participant_graph.complete_train_mask
        )


        model_pipeline = ModelPipeline(
            preprocessing_pipeline=preprocessing_pipeline,
            model=complete_model,
            dst_folder_path=dst_folder_path)
            
        _supplementary_config = copy.copy(supplementary_config) # for saving
        _supplementary_config["model_fit_params"] = dict(
            epochs = 10000,
            report_epoch_steps = 1,
            early_stopping_monitor="train_mae",
            patience = 500,
            l2_reg = 1e-2
        )

        model_pipeline.save_config(supplementary_config = _supplementary_config)

        complete_train_mask = participant_graph.complete_train_mask
        train_val_sets = validation_handler.apply(complete_train_mask)

        graphs_dataset = []
        for i in range(len(train_val_sets)):
            data_graph = participant_graph.clone()
            data_graph.train_mask = torch.Tensor(train_val_sets[i][0])
            data_graph.val_mask = torch.Tensor(train_val_sets[i][1])
            graphs_dataset.append(data_graph)

        model_pipeline.run_models(graphs_dataset=graphs_dataset, **supplementary_config["model_fit_params"])

        time_end_participant = time.time()

        print(f"end participant_id/n_participants-1:{participant_id:d}/{len(participant_indices-1):d}")
        participant_time_taken = time.time()-time_start_participant
        print("participant time taken: {}h - {}m - {}s".format(participant_time_taken//(3600),((participant_time_taken%3600)//60),((participant_time_taken%3600)%60)))
        break

    study_time_taken = time.time()-study_time_start
    print("study time taken: {}h - {}m - {}s".format(study_time_taken//(3600),((study_time_taken%3600)//60),((study_time_taken%3600)%60)))
    data_handler.postprocess(study_raw_folder_path,os.path.join(study_folder_path,"processed"))