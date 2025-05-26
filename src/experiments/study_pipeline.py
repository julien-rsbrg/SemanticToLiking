import time
import os

import pandas as pd

import src.data_handler as data_handler

from src.processing.raw_data_cleaning import prepare_graph_for_participant
from src.processing.preprocessing import PreprocessingPipeline, KeepFeatureNamedSelector, KeepGroupSendersToGroupReceivers, MaskLowerThanSelector, CrossValidationHandler
from src.models.model_pipeline import ModelPipeline
from src.models.baseline_models import SimpleConvModel

if __name__ == "__main__":
    data = data_handler.load_data()

    study_folder_path = "experiments/study_2025-05-16"
    study_raw_folder_path = os.path.join(study_folder_path,"raw")
    supplementary_config = {
        "model_name":"my_super_model",
        "sim_used":"original",
        "model_fit_params":{}}

    participant_indices = data["participant"].unique()
    for participant_id in participant_indices:
        time_start_participant = time.time()
        print(f"start participant_id/n_participants-1:{participant_id:d}/{len(participant_indices-1):d}")
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
                KeepFeatureNamedSelector(verbose=True,feature_names_kept=["liking"])
            ],
            complete_train_mask_selector=MaskLowerThanSelector(feature_name="experience",threshold=0),
            validation_handler=CrossValidationHandler(n_partition=10)
        )

        model = SimpleConvModel(aggr="mean")
        model_pipeline = ModelPipeline(
            preprocessing_pipeline=preprocessing_pipeline,
            model=model,
            dst_folder_path=dst_folder_path)
        
        model_pipeline.save_config(supplementary_config = supplementary_config)
        graphs_dataset = model_pipeline.run_preprocessing(graph = participant_graph)
        model_pipeline.run_models(graphs_dataset=graphs_dataset, **supplementary_config["model_fit_params"])

        time_end_participant = time.time()

        print(f"end participant_id/n_participants-1:{participant_id:d}/{len(participant_indices-1):d}")
        time_take_participant = (time.time()-time_start_participant)
        print("time taken: {}h - {}m - {}s".format(time_take_participant//(3600),((time_take_participant%3600)//60),((time_take_participant%3600)%60)))
    
    data_handler.postprocess(study_raw_folder_path,os.path.join(study_folder_path,"processed"))