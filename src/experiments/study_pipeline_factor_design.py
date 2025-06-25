import time
import os
import copy

import pandas as pd
import itertools
import torch


import src.data_handler as data_handler

from src.processing.raw_data_cleaning import prepare_graph_for_participant
from src.processing.preprocessing import PreprocessingPipeline, KeepFeaturesSelector, FilterGroupSendersToGroupReceivers, KeepKNearestNeighbors, PolynomialFeatureGenerator, MaskLowerThanSelector, CrossValidationHandler,HoldPOutValidationHandler, SeparatePositiveNegative
from src.models.model_pipeline import ModelPipeline
from src.models.baseline_models import SimpleConvModel
from src.models.nn.gnn_layers import MyGATConv
from src.models.nn.ML_frameworks import GNNFramework

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
factor_to_levels = {
    "leakyReLu":[1.0], 
    # The goal of allowing a complete slope (1.0) in the negative domain is for the attention to spread over all values. and have more variance
    "bias":[False,True],
    # check whether the intercept is necessary in the model
    "attention_liking":[False,True],
    # check whether the individual has biased attention based on the liking of the activity
    "amplification_liking":[False,True],
    # check whether the individual amplifies the liking when generalizing (indipendently from attention)
    "dissociate_liking_pos_neg":[False,True],
    # check whether the individual has a different generalization process depending on the liking of the activity
    "2d_polynomial_liking":[False]
    # allows nonlinear processes (degree 2 only for the liking parameters)
}

poss_combinations = list(itertools.product(*factor_to_levels.values()))
poss_combinations = list(set(poss_combinations) - {(1.0, False, False, False, False, False),(1.0, False, True, False, False, False)})

if __name__ == "__main__":
    data = data_handler.load_data()

    for leakyReLu, use_bias, att_liking, amp_liking, diss_liking, liking2d in poss_combinations:
        model_name = f"GAT_LR-{leakyReLu}_bias-{use_bias}_att-liking-{att_liking}_amp-liking-{amp_liking}_diss-liking-{diss_liking}_liking2d-{liking2d}"
        study_name = pd.to_datetime("today").strftime("%Y-%m-%d_%H-%M_")+f"_{model_name}"
        study_folder_path = os.path.join("experiments_results/3-fold_cross_validation",study_name)
        study_raw_folder_path = os.path.join(study_folder_path,"raw")
        print()
        print("model_name:",model_name)
        print()
        print()
        
        supplementary_config = {
            "model_name":model_name,
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

            transformators = [FilterGroupSendersToGroupReceivers(
                group_senders_mask_fn= lambda x: x["experience"] > 0,
                group_receivers_mask_fn= lambda x: x["experience"] <= 0,
                )]
            if diss_liking:
                transformators += [SeparatePositiveNegative(True,"liking")]
                if liking2d:
                    transformators += [PolynomialFeatureGenerator(verbose=True,feature_names_involved=["liking_pos"]),
                                       KeepFeaturesSelector(verbose=True,feature_names_kept=["liking_pos","liking_neg","liking_pos^2"])]
                    dim_in = 3
                else:
                    transformators += [KeepFeaturesSelector(verbose=True,feature_names_kept=["liking_pos","liking_neg"])]
                    dim_in = 2
            else:
                if liking2d:
                    transformators += [PolynomialFeatureGenerator(verbose=True,feature_names_involved=["liking"]),
                                       KeepFeaturesSelector(verbose=True,feature_names_kept=["liking","liking^2"])]
                    dim_in = 2
                else:
                    transformators += [KeepFeaturesSelector(verbose=True,feature_names_kept=["liking"])]  
                    dim_in = 1
            
            preprocessing_pipeline = PreprocessingPipeline(
                transformators=transformators,
                complete_train_mask_selector=MaskLowerThanSelector(feature_name="experience",threshold=0), # WARNING keep the same threshold as in FilterGroupSendersToGroupReceivers
                validation_handler=CrossValidationHandler(n_partition=3)
            )

            src_content_mask = torch.Tensor([True]*dim_in).to(torch.bool)
            src_edge_mask = torch.Tensor([att_liking]*dim_in).to(torch.bool)
            dst_content_mask = torch.Tensor([False]*dim_in).to(torch.bool)
            dst_edge_mask = torch.Tensor([False]*dim_in).to(torch.bool)
            my_module = MyGATConv(
                in_channels=(dim_in,dim_in),
                out_channels=1,
                heads=1,
                negative_slope=leakyReLu,
                add_self_loops=False,
                edge_dim=1,
                dropout=0.0,
                bias=use_bias,
                src_content_mask=src_content_mask,
                src_edge_mask=src_edge_mask,
                dst_content_mask=dst_content_mask,
                dst_edge_mask=dst_edge_mask,
                src_content_require_grad=amp_liking,
                src_content_weight_initializer="ones" if not(amp_liking) else "glorot",
                edge_weight_initializer="ones")

            
            
            ## Training
            complete_model = GNNFramework(my_module,device)

            # TODO this empty predict inside save config

            complete_model.predict(torch.Tensor([[0]*dim_in,[1]*dim_in,[2]*dim_in]),
                                torch.Tensor([[0],
                                              [1]]).to(torch.int64),
                                torch.Tensor([[1]]))

            opt = complete_model.configure_optimizer(lr=1e-2)
            scheduler = complete_model.configure_scheduler(opt,0.1,0.1,10)
            weight_constrainer = None # complete_model.configure_weight_constrainer("clipper",0,100)
            

            supplementary_config["model_fit_params"] = dict(
                epochs = 10000,
                report_epoch_steps = 1,
                optimizer = opt,
                scheduler = scheduler,
                weight_constrainer = weight_constrainer,
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
                patience = 500,
                l2_reg = 1e-2
            )

            model_pipeline.save_config(supplementary_config = _supplementary_config)
            graphs_dataset = model_pipeline.run_preprocessing(graph = participant_graph)
            model_pipeline.run_models(graphs_dataset=graphs_dataset, **supplementary_config["model_fit_params"])

            time_end_participant = time.time()

            print(f"end participant_id/n_participants-1:{participant_id:d}/{len(participant_indices-1):d}")
            time_take_participant = (time.time()-time_start_participant)
            print("time taken: {}h - {}m - {}s".format(time_take_participant//(3600),((time_take_participant%3600)//60),((time_take_participant%3600)%60)))
        
        data_handler.postprocess(study_raw_folder_path,os.path.join(study_folder_path,"processed"))
        
        # add note on the model
        dst_note_path = os.path.join(study_folder_path,"processed","note.txt")
        math_formula = """$$L'_i = \beta_0 + \displaystyle\sum_{j\in \mathcal{N}(i)} \alpha_{j,i} \beta_L L_j$$

        $$\alpha_{j,i} = \frac{\mbox{exp}\left(\mbox{LeakyReLu}_{\lambda}(a_LL_j + a_ss_{j,i})\right)}{\displaystyle\sum_{k\in\mathcal{N}(i)}\mbox{exp}\left(\mbox{LeakyReLu}_{\lambda}(a_L L_k + a_s s_{k,i})\right)}$$

        with 

        - $L_i$ the liking of activity $i$, $L_i'$ the predicted liking of i.

        - $s_{i,j}$ the semantic (cosine) similarity from MPNet from activity $i$ to $j$,

        - $\mathcal{N}(i)$ the neighbors of $i$,

        - $\beta_0$ the bias or intercept

        - $\beta_L$ the weight amplifying the liking

        - $a_L$ the attentional parameter to the liking 

        - $a_s$ the attentional parameter to the similarity

        - $\lambda$ the negative slope of the LeakyReLu

        - $\alpha_{j,i}$ the final weight for considering $j$ to compute the liking of $i$

        Careful: these names currently don't match the ones used in the model itself. See documentation for that. 
        """

        hypotheses = [
            "- LeakyReLu can be cancelled when $\lambda = 1.0$. If better with $\lambda = 1.0$ than $\lambda = 0.0$, then the only way for the model to have equivalent attention to all neighboring activities is to put attentional parameters ($a_L$ and $a_s$) to 0.0. This parameter is meant to compare with previous results... I expect the model to be better when $\lambda = 1.0$, which allows more expressivity coming from the attentional parameters.",
            "- Having a bias/intercept means the participant is giving a liking by default to activities. It is expected to be significantly different from 0.0 in depressed participants and negative, not necessarily for the ones in the control group.",
            "- $a_L$ is expected to play a role... It means that the participant focuses more on some activities than others. For depressed participants, $a_L$ is expected negative, meaning that they focus on negative activities to generalize. For the control group, it is expected to be null.",
            "- $\beta_L$ expresses the amplification applied to an activity once it is focused upon. It is expected to be positive and close to 1. It is expected to be close to 0.0 when the participant is not considering the experienced activities at all, which could be the case in depressed participants.",
            "- Any liking activity could be separated in positive and negative and permit a different set of parameters for each ($\beta_{L^+}$ vs $\beta_{L^-}$, $a_{L^+}$ vs $a_{L^-}$). It is expected that (when applicable) $\beta_{L^+} \neq \beta_{L^-}$ and $a_{L^+} \neq a_{L^-}$ in depressed individuals only."
            "- The liking could be raised to the power 2. This complexify the model and should allow a better fit. It is meant to raise the possibility of more nonlinearities." 
        ]

        method_to_answer_hypotheses = [
            "- A factorial design should have been carried out. Compare models one on one with everything constant except the parameter of interest.",
            "- For comparison, you may consider the performance and the relation of their trained parameters to depression. Later on, pivotal testing (not done yet) or the bayesian approach (not done yet) should more rigorously (not unvalidate or) refute the hypotheses",
            "- Validation is better than training to see how the model perform on previously unseen data (which is the case when using the model on real patients)"
        ]

        note = f"""Model is for ...
        This model is meant for blabla...
        *Model mathematical formula:*\n\n
        {math_formula}\n\n
        """+"*Hypotheses:*\n\n"+'\n\n'.join(hypotheses)+"\n\n*To answer those hypotheses:*\n\n"+"\n\n".join(method_to_answer_hypotheses)

        note += "\n\nFor this model, $\beta_0$ learnt = {beta_0:d}; $\beta_L$ learnt = {beta_L:d}; $a_L$ learnt = {a_l:d}; $\lambda = {_lambda:.1f}$".format(beta_0=use_bias, beta_L=amp_liking, a_l=att_liking,_lambda=leakyReLu)
       
        file = open(dst_note_path, "w") 
        file.write(note) 
        file.close() 