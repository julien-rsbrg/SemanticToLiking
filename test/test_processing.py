import numpy as np
import pandas as pd

import src.processing.preprocessing

def create_playground_graph():
    x = pd.DataFrame({"x":[False,True,True,True,False,True]})
    y = pd.DataFrame()
    edge_attr = pd.DataFrame({"dist":[1,1,2,1,2,3]})
    edge_index = np.array([[0,1,2,3,4,5],
                           [1,0,0,2,2,2]])
    return x,y,edge_attr,edge_index

def test_keep_k_nearest_neighbors():
    print("\n- Launch test_keep_k_nearest_neighbors -\n")
    # k = 1
    x,y,edge_attr,edge_index = create_playground_graph()
    kNNEdgeSelector = src.processing.preprocessing.KeepKNearestNeighbors(k=1,
                                                                         edge_attr_names_used=["dist"])
    edge_index,edge_attr,_x,_y = kNNEdgeSelector.fit_transform(edge_index=edge_index,
                                                             edge_attr=edge_attr,
                                                             x=x,
                                                             y=y)
    
    assert np.allclose(edge_index,np.array([[0,1,3],
                                            [1,0,2]])),("edge_index=",edge_index)
    assert np.allclose(edge_attr.to_numpy(),np.array([[1],[1],[1]]))
    assert np.allclose(_x.to_numpy(),x.to_numpy())
    assert np.allclose(_y.to_numpy(),y.to_numpy())

    # k = 2
    x,y,edge_attr,edge_index = create_playground_graph()
    kNNEdgeSelector = src.processing.preprocessing.KeepKNearestNeighbors(k=2,
                                                                         edge_attr_names_used=["dist"])
    edge_index,edge_attr,x,y = kNNEdgeSelector.fit_transform(edge_index=edge_index,
                                                             edge_attr=edge_attr,
                                                             x=x,
                                                             y=y)
    assert np.allclose(edge_index,np.array([[0,1,2,3,4],
                                            [1,0,0,2,2]])),("edge_index=",edge_index)
    assert np.allclose(edge_attr.to_numpy(),np.array([[1],[1],[2],[1],[2]]))
    assert np.allclose(_x.to_numpy(),x.to_numpy())
    assert np.allclose(_y.to_numpy(),y.to_numpy())
    print("\n- Validated test_cut_group -\n")


def test_cut_group():
    print("\n- Launch test_cut_group -\n")
    x,y,edge_attr,edge_index = create_playground_graph()
    cut_group_true_to_false = src.processing.preprocessing.CutGroupSendersToGroupReceivers(
        group_senders_mask_fn=lambda x: x["x"],
        group_receivers_mask_fn=lambda x: ~x["x"],
    )
    _edge_index, _edge_attr, _x, _y = cut_group_true_to_false.fit_transform(
        edge_index=edge_index,
        edge_attr=edge_attr,
        x=x,
        y=y
    )
    assert np.allclose(_edge_index,np.array([[0,3,4,5],
                                             [1,2,2,2]])),("edge_index=",edge_index)
    assert np.allclose(_edge_attr.to_numpy(),np.array([[1],[1],[2],[3]]))
    assert np.allclose(_x.to_numpy(),x.to_numpy())
    assert np.allclose(_y.to_numpy(),y.to_numpy())
    print("\n- Validated test_cut_group -\n")



if __name__ == "__main__":
    test_keep_k_nearest_neighbors()
    test_cut_group()