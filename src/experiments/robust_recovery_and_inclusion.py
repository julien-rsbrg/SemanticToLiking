import os
import copy

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.utils.undirected import is_undirected
from torch_geometric.utils.isolated import contains_isolated_nodes

import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist

import networkx as nx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device = ",device)

# perso modules
from src.visualization.display_graph import convert_torch_to_networkx_graph
import src.graph_analysis.shortest_paths as gsp


from src.processing.preprocessing import PreprocessingPipeline, MaskThreshold, KeepMonotonousNodeAttr, KeepNodeFeaturesSelector, NoValidationHandler, TurnUndirected, KeepKNearestNeighbors, LockKNearestGroupSendersToGroupReceivers, FillFeature, AddL2DistEdge, AddNLeapsFromClusterNode, KeepEdgeFeaturesSelector, RemoveIsolatedNodes

from src.models.model_pipeline import ModelPipeline

from src.models.nn.gnn_layers import MyGATConvNLeaps
from src.models.nn.ML_frameworks import GNNFramework
from src.models.kernel.gaussian_process import MyRBFGaussianProcessRegressor
from src.models.kernel.diffusion_kernel import MaternKernelModel

from src.utils import recursive_mkdirs

# preprocessing pipelines

dk_preprocessing_pipeline = PreprocessingPipeline(
    transformators=[
        KeepNodeFeaturesSelector(True,
                                 feature_names_kept=["liking"]), # not really necessary
        KeepEdgeFeaturesSelector(False,
                                 feature_names_kept=["similarity"])
        ],
        complete_train_mask_selector=MaskThreshold(feature_name="experience",threshold=0),
        validation_handler=NoValidationHandler(),
        base_mask_selector=MaskThreshold("experience", threshold=0, mode="strict_upper")
)

gp_preprocessing_pipeline = PreprocessingPipeline(
    transformators=[
        KeepNodeFeaturesSelector(False,
                                 feature_names_kept=['embedding_0',"embedding_1"]), # prior knowledge about the graphs
        KeepEdgeFeaturesSelector(False,
                                 feature_names_kept=["similarity"])
        ],
        complete_train_mask_selector=MaskThreshold(feature_name="experience",threshold=0),
        validation_handler=NoValidationHandler(),
        base_mask_selector=MaskThreshold("experience", threshold=0, mode="strict_upper")
)

gat_preprocessing_pipeline = PreprocessingPipeline(
    transformators=[
        KeepMonotonousNodeAttr(
            node_attr_name_used = "leaps_from_cluster",
            ascending = True,
            strict = True
        ), # remove the undirectedness of the graph
        KeepNodeFeaturesSelector(True,
                                 feature_names_kept=["liking"]), # not really necessary
        KeepEdgeFeaturesSelector(False,
                                 feature_names_kept=["similarity"])
        ],
        complete_train_mask_selector=MaskThreshold(feature_name="experience",threshold=0),
        validation_handler=NoValidationHandler(),
        base_mask_selector=MaskThreshold("experience", threshold=0, mode="strict_upper")
)



# main code

dst_main_folder = "experiments_results/recovery_and_inclusion"

poss_models_fitted = ["DiffusionKernel","GaussianProcess","GAT"]



if __name__ == "__main__":
    for model_to_fit_folder in os.listdir("data/generated/predictions"):
        model_to_fit = model_to_fit_folder.split("_")[-1]

        for param_version_folder in os.listdir(os.path.join("data/generated/predictions",model_to_fit_folder)):
            for graph_folder in os.listdir(os.path.join("data/generated/predictions",model_to_fit_folder,param_version_folder)):
                if os.path.isdir(os.path.join("data/generated/predictions",model_to_fit_folder,param_version_folder,graph_folder)):
                    for subgraph_file in os.listdir(os.path.join("data/generated/predictions",model_to_fit_folder,param_version_folder,graph_folder)):
                        subgraph_path = os.path.join("data/generated/predictions",model_to_fit_folder,param_version_folder,graph_folder,subgraph_file)
                        graph = torch.load(subgraph_path,weights_only=False)

                        for model_fitted in poss_models_fitted:
                            dst_folder = os.path.join(dst_main_folder,model_to_fit_folder,param_version_folder,graph_folder,subgraph_file,model_fitted)

                            supplementary_config = {
                                "model_name":model_fitted,
                                "current_date":pd.to_datetime("today").strftime("%Y-%m-%d_%H-%M_"),
                                "model_fit_params":{}
                            }
                            
                            if model_fitted == "DiffusionKernel":
                                dk_model = MaternKernelModel(dim_out=1, nu = np.inf)

                                full_dk_model = ModelPipeline(
                                    preprocessing_pipeline = dk_preprocessing_pipeline,
                                    model = dk_model,
                                    dst_folder_path = dst_folder
                                )

                                supplementary_config["model_fit_params"] = dict(
                                    complete_train_mask = graph.complete_train_mask,
                                    edge_weight_name = "similarity",
                                    bounds = [(1e-3,1e2)], # should be consistent across all scripts
                                    method="brute",
                                    Ns = 2
                                )
                                full_dk_model.save_config(supplementary_config = {})
                                graphs_dataset = full_dk_model.run_preprocessing(graph = graph)
                                full_dk_model.run_models(graphs_dataset=graphs_dataset, **supplementary_config["model_fit_params"])

                            elif model_fitted == "GaussianProcess":
                                gp_model = MyRBFGaussianProcessRegressor(dim_out=1,lengthscale=0.5,length_scale_bounds="fixed",dist_metric="cosine")

                                full_gp_model = ModelPipeline(
                                    preprocessing_pipeline = gp_preprocessing_pipeline,
                                    model = gp_model,
                                    dst_folder_path = dst_folder
                                )
                                
                                supplementary_config["model_fit_params"] = dict(
                                    complete_train_mask = graph.complete_train_mask,
                                    edge_weight_name = "similarity",
                                    bounds = [(1e-3,1e2)], # should be consistent across all scripts
                                    method="brute",
                                    Ns = 2
                                )

                                full_gp_model.save_config(supplementary_config = {})
                                graphs_dataset = full_gp_model.run_preprocessing(graph = graph)
                                full_gp_model.run_models(graphs_dataset=graphs_dataset, **supplementary_config["model_fit_params"])

                            elif model_fitted == "GAT":
                                dim_in = 1
                                use_bias, att_liking, amp_liking, edge, combine_att_amp = True, True, True, True, False

                                src_content_mask = torch.Tensor([True]*dim_in).to(torch.bool)
                                src_edge_mask = torch.Tensor([att_liking]*dim_in).to(torch.bool)
                                dst_content_mask = torch.Tensor([False]*dim_in).to(torch.bool)
                                dst_edge_mask = torch.Tensor([False]*dim_in).to(torch.bool)
                                my_module = MyGATConvNLeaps(
                                    n_leaps = 6, # I checked before there won't be more needed from the dataset
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
                                    src_content_edge_are_same=combine_att_amp)

                                gat_model = GNNFramework(my_module,device)

                                # for completing initialization of params
                                gat_model.predict(torch.Tensor([[0]*dim_in,[1]*dim_in,[2]*dim_in]),
                                    torch.Tensor([[0],
                                                [1]]).to(torch.int64),
                                    torch.Tensor([[1]]),
                                    complete_train_mask = [False,True,False])
                                
                                full_gat_model = ModelPipeline(
                                    preprocessing_pipeline = gat_preprocessing_pipeline,
                                    model = gat_model,
                                    dst_folder_path = dst_folder
                                )
                                
                                opt = gat_model.configure_optimizer(lr=1e-2)
                                scheduler = gat_model.configure_scheduler(opt,0.1,0.1,10)
                                weight_constrainer = None # complete_model.configure_weight_constrainer("clipper",0,100)

                                supplementary_config["model_fit_params"] = dict(
                                    epochs = 2,
                                    report_epoch_steps = 1,
                                    optimizer = opt,
                                    scheduler = scheduler,
                                    weight_constrainer = weight_constrainer,
                                    early_stopping_monitor="train_mae",
                                    patience=500,
                                    l2_reg=1e-2,
                                    complete_train_mask = graph.complete_train_mask
                                )

                                _supplementary_config = copy.copy(supplementary_config) # for saving, and removing the elements that can't be saved
                                _supplementary_config["model_fit_params"] = dict(
                                    epochs = 10000,
                                    report_epoch_steps = 1,
                                    early_stopping_monitor="train_mae",
                                    patience = 500,
                                    l2_reg = 1e-2
                                )

                                full_gat_model.save_config(supplementary_config = _supplementary_config)

                                full_gat_model.save_config(supplementary_config = {})
                                graphs_dataset = full_gat_model.run_preprocessing(graph = graph)
                                full_gat_model.run_models(graphs_dataset=graphs_dataset, **supplementary_config["model_fit_params"])

                            else:
                                raise NotImplementedError(f"Unknown model_fitted: {model_fitted}")
