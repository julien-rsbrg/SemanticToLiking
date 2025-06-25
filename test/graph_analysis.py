
import numpy as np

from src.graph_analysis.shortest_paths import shortest_path_djikstra

def test_shortest_path_djikstra():
    edge_index = np.array([[0,0,0,0,1,2,2,3,4],
                        [1,2,3,4,6,3,5,5,6]])
    edge_weight = np.array([2,2,4,1,2,1,5,1,2])
    true_values = np.array([0,2,2,3,1,4,3])

    assert np.allclose(true_values,shortest_path_djikstra(0,edge_index,edge_weight)[0])