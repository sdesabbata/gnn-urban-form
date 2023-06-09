# Base libraries
import networkx as nx
import osmnx as ox
from torch_geometric.utils import from_networkx
# For testing
import unittest
import random
# OS environment setup
from local_directories import *

def norm_min_max(val, val_min, val_max, min_max_limit):
    val_norm = float(
        (val - val_min) /
        (val_max - val_min)
    )
    if min_max_limit and val_norm > 1.0:
        val_norm = 1.0
    if min_max_limit and val_norm < 0.0:
        val_norm = 0.0
    return val_norm


def to_pyg_graph(
        ego_node_id,
        ego_node_graph,
        node_attr_names=["street_count"],
        node_attr_min_max={"street_count": (0, 4)},
        node_attr_min_max_limit=True,
        edge_attr_names=["length"],
        edge_attr_min_max={"length": (0, 1000)},
        edge_attr_min_max_limit=True,
        # DO NOT use the next argument
        # this is for testing only
        test_node_ids=False
    ):
    ego_node_graph_copy = ego_node_graph.copy()
    pyg_node_index = None

    # Remove all node attributes but street_count (node degree)
    # which becomes x
    for node_index, node_object in enumerate(ego_node_graph_copy.nodes(data=True)):
        # Extract node key and attr
        node_key = node_object[0]
        node_attr = node_object[1]
        # Check if ego node and save index to return
        if node_key == ego_node_id:
            pyg_node_index = node_index
        # Save node attributes in order
        pyg_node_attrs = []
        if test_node_ids:
            pyg_node_attrs.append(int(node_key))
        else:
            for node_attr_name in node_attr_names:
                pyg_node_attrs.append(
                    norm_min_max(
                        node_attr[node_attr_name],
                        node_attr_min_max[node_attr_name][0],
                        node_attr_min_max[node_attr_name][1],
                        node_attr_min_max_limit
                    )
                )
        # Remove all node attributes
        for key in list(node_attr):
            node_attr.pop(key, None)
        # Add single node attribute as a list of floats
        node_attr["x"] = pyg_node_attrs

    # Remove all edge attributes but length
    # which becomes edge_weight
    for _, _, edge_attr in ego_node_graph_copy.edges(data=True):
        # Save edge attributes in order
        pyg_edge_attrs = []
        for edge_attr_name in edge_attr_names:
            pyg_edge_attrs.append(
                norm_min_max(
                    edge_attr[edge_attr_name],
                    edge_attr_min_max[edge_attr_name][0],
                    edge_attr_min_max[edge_attr_name][1],
                    edge_attr_min_max_limit
                )
            )
        for key in list(edge_attr):
            edge_attr.pop(key, None)
        edge_attr["edge_weight"] = pyg_edge_attrs

    # Create Pytorch Geometric graph
    pyg_graph = from_networkx(ego_node_graph_copy)

    # Remove additional graph attributes
    del pyg_graph.created_date
    del pyg_graph.created_with
    del pyg_graph.crs
    del pyg_graph.simplified
    del ego_node_graph_copy

    # Return Pytorch Geometric graph
    return pyg_graph, pyg_node_index


# --- TESTING --- #

class test_utils(unittest.TestCase):

    def test_norm_min_max(self):
        # Test with limits
        self.assertEqual(
            norm_min_max(-0.1,0,1,True),
            0
        )
        self.assertEqual(
            norm_min_max(0,0,1,True),
            0
        )
        self.assertEqual(
            norm_min_max(0.5,0,1,True),
            0.5
        )
        self.assertEqual(
            norm_min_max(1,0,1,True),
            1
        )
        self.assertEqual(
            norm_min_max(1.1,0,1,True),
            1
        )
        # Test with limits and normalisation
        self.assertEqual(
            norm_min_max(-1,0,10,True),
            0
        )
        self.assertEqual(
            norm_min_max(0,0,10,True),
            0
        )
        self.assertEqual(
            norm_min_max(5,0,10,True),
            0.5
        )
        self.assertEqual(
            norm_min_max(10,0,10,True),
            1
        )
        self.assertEqual(
            norm_min_max(11,0,10,True),
            1
        )
        # Test without limits
        self.assertEqual(
            norm_min_max(-0.1,0,1,False),
            -0.1
        )
        self.assertEqual(
            norm_min_max(0,0,1,False),
            0
        )
        self.assertEqual(
            norm_min_max(0.5,0,1,False),
            0.5
        )
        self.assertEqual(
            norm_min_max(1,0,1,False),
            1
        )
        self.assertEqual(
            norm_min_max(1.1,0,1,False),
            1.1
        )
        # Test with limits and normalisation
        self.assertEqual(
            norm_min_max(-1,0,10,False),
            -0.1
        )
        self.assertEqual(
            norm_min_max(0,0,10,False),
            0
        )
        self.assertEqual(
            norm_min_max(5,0,10,False),
            0.5
        )
        self.assertEqual(
            norm_min_max(10,0,10,False),
            1
        )
        self.assertEqual(
            norm_min_max(11,0,10,False),
            1.1
        )

    def test_to_pyg_graph(self):
        max_distance = 500
        leicester = ox.io.load_graphml(bulk_storage_directory + "/osmnx/raw_excluded/leicester-1864.graphml")
        for node in random.sample(list(leicester.nodes), 100):
            # Extract ego graph
            node_ego_graph = nx.generators.ego_graph(
                leicester, node,
                radius=max_distance,
                undirected=True,
                distance="length"
            )
            # Convert to PyG graph
            pyg_ego_graph, pyg_node_index = to_pyg_graph(
                node, node_ego_graph,
                test_node_ids=True
            )
            # Test node ID matches
            #print(f"{node=} {node_ego_graph.nodes=} {pyg_node_index=}")
            self.assertEqual(
                int(pyg_ego_graph.x[pyg_node_index][0]),
                node
            )

if __name__ == '__main__':
    unittest.main()

# June 9th, 2023
# Ran 2 tests in 89.173s
# OK
# Process finished with exit code 0