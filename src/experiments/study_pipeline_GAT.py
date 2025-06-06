import time
import os
import copy

import pandas as pd
import torch

import src.data_handler as data_handler

from src.processing.raw_data_cleaning import prepare_graph_for_participant
from src.processing.preprocessing import PreprocessingPipeline, SeparatePositiveNegative, PolynomialFeatureGenerator, KeepFeatureNamedSelector, KeepGroupSendersToGroupReceivers, KeepKNearestNeighbors, MaskLowerThanSelector, CrossValidationHandler
from src.models.model_pipeline import ModelPipeline
from src.models.baseline_models import SimpleConvModel
from src.models.nn.gnn_layers import MyGATConv
from src.models.nn.ML_frameworks import GNNFramework

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    data = data_handler.load_data()

    study_folder_path = "experiments_results/test"
    study_raw_folder_path = os.path.join(study_folder_path,"raw")

    
    supplementary_config = {
        "model_name":"GAT_liking_3NN",
        "sim_used":"original",
        "model_fit_params":{}}

    participant_indices = data["participant"].unique()
    for participant_id in participant_indices:
        time_start_participant = time.time()
        print(f"start participant_id/(n_participants-1):{participant_id:d}/{len(participant_indices-1):d}")
        dst_folder_path = os.path.join(study_raw_folder_path,f"participant_{participant_id:d}")
        
        participant_graph, _ = prepare_graph_for_participant(data=data, 
                                                             participant_id=participant_id, 
                                                             sim_used=supplementary_config["sim_used"])

        preprocessing_pipeline = PreprocessingPipeline(
            transformators=[
                KeepGroupSendersToGroupReceivers(
                    group_senders_mask_fn= lambda x: x["experience"] > 0,
                    group_receivers_mask_fn= lambda x: x["experience"] <= 0,
                ),
                KeepKNearestNeighbors(3,["_sim"]),
                SeparatePositiveNegative(True,"liking"),
                PolynomialFeatureGenerator(verbose=True,feature_names_involved=["liking_pos"]),
                KeepFeatureNamedSelector(verbose=True,feature_names_kept=["liking_pos","liking_neg","liking_pos^2"])
            ],
            complete_train_mask_selector=MaskLowerThanSelector(feature_name="experience",threshold=0),
            validation_handler=CrossValidationHandler(n_partition=3)
        )
        dim_in = 3

        src_content_mask = torch.Tensor([True]*dim_in).to(torch.bool)
        src_edge_mask = torch.Tensor([False]*dim_in).to(torch.bool)
        dst_content_mask = torch.Tensor([False]*dim_in).to(torch.bool)
        dst_edge_mask = torch.Tensor([False]*dim_in).to(torch.bool)
        my_module = MyGATConv(
            in_channels=(dim_in,dim_in),
            out_channels=1,
            heads=1,
            negative_slope=0.0,
            add_self_loops=False,
            edge_dim=1,
            dropout=0.0,
            bias=True,
            src_content_mask=src_content_mask,
            src_edge_mask=src_edge_mask,
            dst_content_mask=dst_content_mask,
            dst_edge_mask=dst_edge_mask,
            src_content_require_grad=True,
            src_content_weight_initializer="glorot",
            edge_weight_initializer="ones")

        
        
        ## Training
        complete_model = GNNFramework(my_module,device)

        # TODO this empty predit inside save config

        complete_model.predict(torch.Tensor([[0]*dim_in,[1]*dim_in,[2]*dim_in]),
                               torch.Tensor([[0,0,1],
                                             [1,2,0]]).to(torch.int64),
                               torch.Tensor([[1],[1],[1]]))

        opt = complete_model.configure_optimizer(lr=10)
        scheduler = complete_model.configure_scheduler(opt,0.1,0.1,10)
        weight_constrainer = complete_model.configure_weight_constrainer("clipper",0,100)
        

        supplementary_config["model_fit_params"] = dict(
            epochs = 10,
            report_epoch_steps = 1,
            optimizer = opt,
            scheduler = scheduler,
            weight_constrainer=weight_constrainer,
            early_stopping_monitor="val_mae",
            patience=500,
            l2_reg=1e-2
        )


        model_pipeline = ModelPipeline(
            preprocessing_pipeline=preprocessing_pipeline,
            model=complete_model,
            dst_folder_path=dst_folder_path)
        
        _supplementary_config = copy.copy(supplementary_config) # for saving
        _supplementary_config["model_fit_params"] = dict(
            epochs = 10000,
            report_epoch_steps = 1,
            early_stopping_monitor="val_mae",
            patience=500,
            l2_reg=1e-2
        )

        model_pipeline.save_config(supplementary_config = _supplementary_config)
        graphs_dataset = model_pipeline.run_preprocessing(graph = participant_graph)
        model_pipeline.run_models(graphs_dataset=graphs_dataset, **supplementary_config["model_fit_params"])

        time_end_participant = time.time()

        print(f"end participant_id/n_participants-1:{participant_id:d}/{len(participant_indices-1):d}")
        time_take_participant = (time.time()-time_start_participant)
        print("time taken: {}h - {}m - {}s".format(time_take_participant//(3600),((time_take_participant%3600)//60),((time_take_participant%3600)%60)))
    
    data_handler.postprocess(study_raw_folder_path,os.path.join(study_folder_path,"processed"))