from typing import Tuple

import pandas as pd
import torch
from torch_geometric.data import Data

def build_node_table(complete_data_table:pd.DataFrame,
                     feature_names:list[str]) -> pd.DataFrame:
    
    word1_feature_names = ["word1_%s"%fname for fname in feature_names]
    extracted_features_word1 = complete_data_table[["word1"]+word1_feature_names]
    col_renaming = {"word1":"word"}
    col_renaming.update({word1_feature_names[i]:feature_names[i] for i in range(len(feature_names))})
    extracted_features_word1.rename(columns=col_renaming,inplace=True)

    # in case the complete_data_table doesn't contain an undirected graph
    word2_feature_names = ["word2_%s"%fname for fname in feature_names]
    extracted_features_word2 = complete_data_table[["word2"]+word2_feature_names]
    col_renaming = {"word2":"word"}
    col_renaming.update({word2_feature_names[i]:feature_names[i] for i in range(len(feature_names))})
    extracted_features_word2.rename(columns=col_renaming,inplace=True)

    node_data_table = pd.concat([extracted_features_word1,extracted_features_word2],axis=0)

    node_data_table.drop_duplicates(subset=["word"],keep="first",inplace=True)
    node_data_table.reset_index(inplace=True,drop=True)
    return node_data_table



def get_translater_index_to_var(node_table:pd.DataFrame,var_name:str) -> dict:
    translater_index_to_var = {}
    for v, i in enumerate(node_table[var_name]):
        translater_index_to_var[i] = v
    return translater_index_to_var



def build_edge_table(complete_data_table:pd.DataFrame,
                     translater_word_to_index:dict,
                     edge_attr_names:list[str]) -> Tuple[pd.DataFrame,dict]:
    
    complete_data_table["word1_index"] = complete_data_table["word1"].apply(lambda single_word: translater_word_to_index[single_word])
    complete_data_table["word2_index"] = complete_data_table["word2"].apply(lambda single_word: translater_word_to_index[single_word])

    edge_table = complete_data_table[["word1_index","word2_index"] + edge_attr_names]

    return edge_table



def convert_table_to_graph(
        complete_data_table:pd.DataFrame,
        node_attr_names:list[str],
        edge_attr_names:list[str],
        node_label_names:list[str] = None,
        return_word_to_index:bool = False) -> Data:
    
    node_table = build_node_table(complete_data_table=complete_data_table,
                                  feature_names=node_attr_names)
    node_attr = node_table[node_attr_names].values
    node_attr = torch.Tensor(node_attr)

    translater_word_to_index = get_translater_index_to_var(node_table=node_table,var_name="word")
    edge_table = build_edge_table(complete_data_table = complete_data_table,
                                  translater_word_to_index = translater_word_to_index,
                                  edge_attr_names=edge_attr_names)
    
    edge_index = edge_table[["word1_index","word2_index"]]
    edge_index = torch.Tensor(edge_index.to_numpy()).to(torch.int64)
    edge_index = edge_index.T
    reversed_edge_index = edge_index[[1,0],:]
    edge_index = torch.concat([edge_index,reversed_edge_index],dim=1)

    edge_attr = edge_table[edge_attr_names].values # n edges, n edge attr 
    edge_attr = torch.Tensor(edge_attr)
    edge_attr = torch.concat([edge_attr,edge_attr],dim=0)

    node_train_mask = torch.ones(len(node_attr),dtype=torch.bool)

    node_labels = None
    if not(node_label_names is None):
        node_labels = node_table[node_label_names].values
        node_labels = torch.Tensor(node_labels)

    data_graph = Data(
        x = node_attr, 
        edge_index = edge_index,
        edge_attr = edge_attr,
        y = node_labels, 
        train_mask = torch.Tensor(node_train_mask), 
        val_mask = torch.Tensor(~node_train_mask)
        )

    def test_function(data_graph:Data):
        print("Test function convert_table_to_graph")
        print(data_graph)
        print("validate:",data_graph.validate())
        print("is undirected:", data_graph.is_undirected())

        has_self_loop = min(abs(data_graph.edge_index[0]-data_graph.edge_index[1])) < 1
        print("has_self_loop:",has_self_loop)
        print("end Test function convert_table_to_graph")
    
    test_function(data_graph)
    if return_word_to_index:
        return data_graph, translater_word_to_index
    else:
        return data_graph