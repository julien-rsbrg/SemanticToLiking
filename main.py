from sklearn.preprocessing import StandardScaler

import torch
import torch_geometric

from src.models.nn_layers import MLPModel
from src.models.gnn_layers import myGATConv
from src.models.frameworks import GNN_naive_framework
import src.loading as loading
import src.processing as processing

from visualization.analyse_model import plot_errors_labels_comparison

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Data


data = loading.load_data()

def nor_function(a,b):
    return (a or b) and not(a and b)

data["NoExp_Exp"] = data.apply(lambda row: nor_function(row["word1_experience"]>50,row["word2_experience"]>50),axis=1)

scaler = StandardScaler()
data.loc[:,["word1_sc_liking","word2_sc_liking","sc_senenceBERT_mpnet_similarity","sc_depressionCont","sc_NoExp_Exp"]] = scaler.fit_transform(data.loc[:,["word1_liking","word2_liking","senenceBERT_mpnet_similarity","depressionCont","NoExp_Exp"]])
data.loc[:,["word1_sc_liking","word2_sc_liking","sc_senenceBERT_mpnet_similarity","sc_depressionCont","sc_NoExp_Exp"]]

subdata = data[data["participant"] == 1]

participant_graph = processing.convert_table_to_graph(
    complete_data_table=subdata,
    node_attr_names=["sc_liking","experience"],
    node_label_names=["sc_liking"],
    edge_attr_names=["senenceBERT_mpnet_similarity"])

print("participant_graph:",participant_graph)
print("participant_graph.x:",participant_graph.x)
print("participant_graph.edge_attr:",participant_graph.edge_attr)
print("participant_graph.y:",participant_graph.y)

## Model

#my_module = MLPModel(c_in=1, c_hidden=5, c_out=1,num_layers=2,dp_rate=0.0)

src_mask = torch.Tensor([False,True]).to(torch.bool)
dst_mask = torch.Tensor([True,True]).to(torch.bool)
my_module = myGATConv(
    in_channels=(2,2),
    out_channels=1,
    heads=1,
    negative_slope=0.0,
    add_self_loops=False,
    edge_dim=1,
    dropout=0.0,
    src_mask=src_mask,
    dst_mask=dst_mask)

print(my_module.lin_src)

"""
my_module = torch_geometric.nn.GATConv(
    in_channels=(1,1),
    out_channels=1,
    heads=1,
    negative_slope=0.0,
    add_self_loops=False,
    edge_dim=1,
    dropout=0.0)


my_module = torch_geometric.nn.GCNConv(
    in_channels=1, 
    out_channels=1
    )
"""

# my_module(x=participant_graph.x,edge_index=participant_graph.edge_index,edge_attr=participant_graph.edge_attr)
print(my_module)
print([param for param in my_module.parameters()])

## Training
complete_model = GNN_naive_framework(my_module,device)
opt = complete_model.configure_optimizer(lr=1)
scheduler = complete_model.configure_scheduler(opt,1,1,10)

history = complete_model.train([participant_graph],10000,1,opt,scheduler,"train_loss",200)

## Visualization
plot_errors_labels_comparison(complete_model,participant_graph,False)