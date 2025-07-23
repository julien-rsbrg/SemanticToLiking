import time
import os
import copy
import itertools

import pandas as pd
import torch


import src.data_handler as data_handler

from src.processing.preprocessing import PreprocessingPipeline, MaskThreshold, KeepMonotonousNodeAttr, KeepNodeFeaturesSelector, NoValidationHandler
from src.models.model_pipeline import ModelPipeline
from src.models.nn.gnn_layers import MyGATConvNLeaps
from src.models.nn.ML_frameworks import GNNFramework

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_or_test = 'train'

"""
factor_to_levels = {
    "bias":[False,True],
    # check whether the intercept is necessary in the model
    "attention_liking":[False,True],
    # check whether the individual has biased attention based on the liking of the activity
    "amplification_liking":[False,True],
    "edge":[False,True],
    "combine_att_amp_liking":[False]
}

poss_combinations = list(itertools.product(*factor_to_levels.values()))
"""

factor_to_levels = {
    "bias":[False,True],
    # check whether the intercept is necessary in the model
    "attention_liking":[True],
    # check whether the individual has biased attention based on the liking of the activity
    "amplification_liking":[True],
    "edge":[False,True],
    "combine_att_amp_liking":[True]
}

poss_combinations = list(itertools.product(*factor_to_levels.values()))

# if some combinations are already carried: 
poss_combinations = list(set(poss_combinations) - 
                         {(False, False, False, False, False),
                          (False, False, False, True, False),
                          (False, False, True, False, False),
                          (False, False, True, True, False),
                          (False, True, False, False, False),
                          (False, True, False, True, False),
                          (False, True, True, False, False),
                          (False, True, True, True, False),
                          (True, False, False, False, False),
                          (True, False, False, True, False)})



if __name__ == "__main__":
    all_model_names = []
    all_study_time_taken = []
    for use_bias, att_liking, amp_liking, edge, combine_att_amp in poss_combinations:
        study_time_start = time.time()
        model_name = f"GAT_liking_sim_amp_3NN_3ExpNN_no_val_bias-{use_bias}_att-liking-{att_liking}_amp-liking-{amp_liking}_edge-{edge}_comb-att-amp-{combine_att_amp}"
        all_model_names.append(model_name)

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
            
            participant_graph = torch.load(f"data/temp_1_3NN_3ExpNN/{train_or_test}/participant_graph_{participant_id}",weights_only=False)

            if participant_graph is None:
                print(f"{participant_id} is None")
                continue

            validation_handler = NoValidationHandler()
            preprocessing_pipeline = PreprocessingPipeline(
                transformators=[
                    KeepMonotonousNodeAttr(
                        node_attr_name_used = "leaps_from_cluster",
                        ascending = True,
                        strict = True
                    ), # remove the undirectedness of the graph
                    KeepNodeFeaturesSelector(True,
                                             feature_names_kept=["liking"])
                ],
                complete_train_mask_selector=MaskThreshold(feature_name="experience",threshold=0),
                validation_handler=validation_handler,
                base_mask_selector=None
            )

            dim_in = 1

            src_content_mask = torch.Tensor([True]*dim_in).to(torch.bool)
            src_edge_mask = torch.Tensor([att_liking]*dim_in).to(torch.bool)
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
                bias=use_bias,
                src_content_mask=src_content_mask,
                src_edge_mask=src_edge_mask,
                dst_content_mask=dst_content_mask,
                dst_edge_mask=dst_edge_mask,
                src_content_weight_initializer="glorot" if amp_liking else "ones",
                edge_weight_initializer="glorot" if edge else "ones",
                src_content_require_grad=amp_liking,
                edge_require_grad=edge,
                src_content_edge_are_same = combine_att_amp)


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
                dst_folder_path=dst_folder_path
            )
                
            _supplementary_config = copy.copy(supplementary_config) # for saving
            _supplementary_config["model_fit_params"] = dict(
                epochs = 10000,
                report_epoch_steps = 1,
                early_stopping_monitor="train_mae",
                patience = 500,
                l2_reg = 1e-2
            )

            model_pipeline.save_config(supplementary_config = _supplementary_config)
            graphs_dataset = model_pipeline.run_preprocessing(graph = participant_graph)
            print("graphs_dataset\n",graphs_dataset)
            model_pipeline.run_models(graphs_dataset=graphs_dataset, **supplementary_config["model_fit_params"])

            time_end_participant = time.time()

            print(f"end (participant_i+1)/n_participants-1:{participant_i+1:d}/{len(participant_graph_names):d}")
            participant_time_taken = time.time()-time_start_participant
            print("participant time taken: {}h - {}m - {:.4f}s".format(participant_time_taken//(3600),((participant_time_taken%3600)//60),((participant_time_taken%3600)%60)))

        study_time_taken = time.time()-study_time_start
        all_study_time_taken.append(study_time_taken)
        print("study time taken: {}h - {}m - {:.4f}s".format(study_time_taken//(3600),((study_time_taken%3600)//60),((study_time_taken%3600)%60)))
        data_handler.postprocess(study_raw_folder_path,os.path.join(study_folder_path,"processed"))

    print("\n== Report time durations ==")
    for i in range(len(all_model_names)):
        print(f"model {all_model_names[i]}" + " time taken: {}h - {}m - {:.4f}s".format(all_study_time_taken[i]//(3600),((all_study_time_taken[i]%3600)//60),((all_study_time_taken[i]%3600)%60)))