# Base libraries
import random
import numpy as np
import pandas as pd
# NetworkX
import networkx as nx
import osmnx as ox
# Torch
import torch
from torch_geometric.nn import GAE
from torch_geometric.utils import from_networkx
# GNN models
from gnnuf_models import *
# OS environment setup
from local_directories import *

# Convert linegraph to Pytorch Geometric linegraph
def to_pyg_linegraph(ego_graph, max_distance):
    # Extract lengths and slopes
    seg_edges_length = nx.get_edge_attributes(ego_graph, "length")
    seg_edges_grade_abs = nx.get_edge_attributes(ego_graph, "grade_abs")

    # Create line graph
    seg_linegraph = nx.line_graph(ego_graph)
    # Add street lenghts and slopes as attribute x
    for seglg_node in seg_linegraph.nodes():
        seg_edge_length = seg_edges_length[(seglg_node[0], seglg_node[1], 0)]
        seg_edge_grade_abs = seg_edges_grade_abs[(seglg_node[0], seglg_node[1], 0)]
        seg_linegraph.nodes[seglg_node]["x"] = [
            # Normalisation
            (seg_edge_length / max_distance),
            ((seg_edge_grade_abs / 0.05) if seg_edge_grade_abs < 0.05 else 1.0)
        ]
    del seglg_node

    # Return Pytorch Geometric graph
    return from_networkx(seg_linegraph)

def to_pyg_graph(ego_graph):

    # Remove all node attributes but street_count (node degree)
    # which becomes x
    for _, node_attr in ego_graph.nodes(data=True):
        street_count = node_attr["street_count"]
        for key in list(node_attr):
            node_attr.pop(key, None)
        node_attr["x"] = [float(street_count)]

    # Remove all edge attributes but length
    # which becomes edge_weight
    for _, _, edge_attr in ego_graph.edges(data=True):
        length = edge_attr["length"]
        for key in list(edge_attr):
            edge_attr.pop(key, None)
        edge_attr["edge_weight"] = [float(length)]

    # Create Pytorch Geometric graph
    pyg_graph = from_networkx(ego_graph)

    # Normalise x and edge_weight between 0 and 1
    # pyg_graph.x.max(dim=0)
    # pyg_graph.edge_weight.max(dim=0)
    pyg_graph.x = (pyg_graph.x / pyg_graph.x.max(dim=0).values)
    pyg_graph.edge_weight = 1 - (pyg_graph.edge_weight / pyg_graph.edge_weight.max(dim=0).values)

    # Remove additional graph attributes
    del pyg_graph.created_date
    del pyg_graph.created_with
    del pyg_graph.crs
    del pyg_graph.simplified

    # Return Pytorch Geometric graph
    return pyg_graph


# Reset random seeds
random_seed = 2674
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Run on CUDA")
else:
    print("Run on CPU")

# Load model
model_name = "gnnuf_model_v0-2"
model = GAE(VanillaGCNEncoder(1, 128, 64))
model.load_state_dict(torch.load(this_repo_directory + "/models/" + model_name + ".pt", map_location=device))
model = model.to(device)
model.eval()

# Load Leciester's graph
leicester = ox.io.load_graphml(bulk_storage_directory + "/osmnx/raw_excluded/leicester-1864.graphml")

leicester_embs = {}
neighbourhood_min_nodes = 8
max_distance = 500
count = 0

for node in leicester.nodes:

    # Create the corresponding ego graph
    node_ego_graph = nx.generators.ego_graph(leicester, node, radius=max_distance, undirected=True, distance="length")

    # Only keep the sampled area if it has a minimum number of nodes
    if len(node_ego_graph.nodes()) > neighbourhood_min_nodes:

        # Convert linegraph to Pytorch Geometric linegraph
        #node_pyg = to_pyg_linegraph(node_ego_graph, max_distance)
        node_pyg = to_pyg_graph(node_ego_graph)
        if node_pyg is not None:
            node_pyg = node_pyg.to(device)

            # Encode
            node_pyg_emb = model.encode(node_pyg.x, node_pyg.edge_index, node_pyg.edge_weight)
            node_pyg_emb_gmp = global_mean_pool(node_pyg_emb, None)
            leicester_embs[node] = np.squeeze(node_pyg_emb_gmp.cpu().detach().numpy())

        else:
            print("PyG graph is None")

# Save
leicester_embs_df = pd.DataFrame.from_dict(leicester_embs, orient="index", columns=[f"EMB{i:03d}" for i in range(64)])
leicester_embs_df = leicester_embs_df.reset_index()
leicester_embs_df = leicester_embs_df.rename(columns={"index": "osmnx_node_id"})
leicester_embs_df.to_csv(this_repo_directory + "/data/leicester-1864_emb_" + model_name + ".csv", index=False)
