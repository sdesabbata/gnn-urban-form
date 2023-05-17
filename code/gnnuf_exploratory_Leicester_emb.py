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
model_name = "gnnuf_model_v0-1"
model = GAE(VanillaGCNEncoder(2, 128, 64))
model.load_state_dict(torch.load(this_repo_directory + "/models/" + model_name + ".pt", map_location=device))
model.eval()

# Load Leciester's graph
leicester = ox.io.load_graphml(bulk_storage_directory + "/osmnx/raw/leicester-1864.graphml")

leicester_embs = {}
leicester_basestats_df = None
neighbourhood_min_nodes = 8
max_distance = 1000
count = 0

for node in leicester.nodes:

    # Create the corresponding ego graph
    node_ego_graph = nx.generators.ego_graph(leicester, node, radius=max_distance, undirected=True, distance="length")

    # Only keep the sampled area if it has a minimum number of nodes
    if len(node_ego_graph.nodes()) > neighbourhood_min_nodes:

        # Convert linegraph to Pytorch Geometric linegraph
        node_pyglg = to_pyg_linegraph(node_ego_graph)
        node_pyglg = node_pyglg.to(device)

        # Encode
        node_pyglg_emb = model.encode(node_pyglg.x,node_pyglg.edge_index)
        node_pyglg_emb_gmp = global_mean_pool(node_pyglg_emb, None)
        leicester_embs[node] = np.squeeze(node_pyglg_emb_gmp.cpu().detach().numpy())

        # Calculate base stats
        basestats = ox.stats.basic_stats(node_ego_graph)
        if leicester_basestats_df is None:
            leicester_basestats_df = pd.DataFrame.from_dict(basestats)
        else:
            leicester_basestats_df = pd.concat([leicester_basestats_df, pd.DataFrame.from_dict(basestats)])

# Save
leicester_embs_df = pd.DataFrame.from_dict(leicester_embs, orient="index", columns=[f"EMB{i:03d}" for i in range(64)])
leicester_embs_df = leicester_embs_df.reset_index()
leicester_embs_df = leicester_embs_df.rename(columns={"index": "osmnx_node_id"})
leicester_embs_df.to_csv(this_repo_directory + "/data/leicester-1864_emb_" + model_name + ".csv", index=False)

leicester_basestats_df.to_csv(this_repo_directory + "/data/leicester-1864_basestats.csv", index=False)