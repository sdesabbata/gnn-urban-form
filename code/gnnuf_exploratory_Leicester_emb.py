# Base libraries
import os
from os.path import isfile, join
import math
import random
import numpy as np
import pandas as pd
from umap import UMAP
# NetworkX
import networkx as nx
import osmnx as ox
# Torch
import torch
from torch_geometric.nn import GAE
from torch_geometric.utils import from_networkx, train_test_split_edges
from torch_geometric.loader import GraphSAINTNodeSampler
# GNN models
from gnnuf_models import *
# OS environment setup
from local_directories import *

# Convert linegraph to Pytorch Geometric linegraph
def to_pyg_linegraph(ego_graph):

        # Extract lengths and slopes
        seg_edges_length = nx.get_edge_attributes(ego_graph, "length")
        seg_edges_grade_abs = nx.get_edge_attributes(ego_graph, "grade_abs")

        # Create line graph
        seg_linegraph = nx.line_graph(ego_graph)
        # Add street lenghts and slopes as attribute x
        for seglg_node in seg_linegraph.nodes():
            seg_linegraph.nodes[seglg_node]["x"] = [
                seg_edges_length[(seglg_node[0], seglg_node[1], 0)],
                seg_edges_grade_abs[(seglg_node[0], seglg_node[1], 0)]
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

# Load Leciester's graph
leicester = ox.io.load_graphml(bulk_storage_directory + "/osmnx/raw/leicester-1864.graphml")

model = GAE(VanillaGCNEncoder(2, 128, 64))
model.load_state_dict(torch.load(bulk_storage_directory + "/models/osmnx_model_v1.pt", map_location=torch.device('cpu')))
model.eval()

leicester_embs = {}

neighbourhood_min_nodes = 16
max_distance = 2000

# For 10% of the street nodes in Leicester
leicester_sample_nodes = random.sample(list(leicester.nodes), math.ceil(len(leicester.nodes) * 0.1))

count = 0
#for node in leicester.nodes:
for node in leicester_sample_nodes:
    if count % 10 == 0:
        print(f"{count} of {len(leicester_sample_nodes)} - {(count/len(leicester_sample_nodes))*100:.2f}%")
    count += 1

    # Create the corresponding ego graph
    node_ego_graph = nx.generators.ego_graph(leicester, node, radius=max_distance, undirected=True, distance="length")

    # Only keep the sampled area if it has a minimum number of nodes
    if len(node_ego_graph.nodes()) > neighbourhood_min_nodes:

        # Convert linegraph to Pytorch Geometric linegraph
        node_pyglg = to_pyg_linegraph(node_ego_graph)

        # Encode
        node_pyglg_emb = model.encode(node_pyglg.x,node_pyglg.edge_index)
        node_pyglg_emb_gmp = global_mean_pool(node_pyglg_emb, None)
        leicester_embs[node] = np.squeeze(node_pyglg_emb_gmp.cpu().detach().numpy())


leicester_embs_df = pd.DataFrame.from_dict(leicester_embs, orient="index", columns=[f"EMB{i:03d}" for i in range(64)])
leicester_embs_df = leicester_embs_df.reset_index()
leicester_embs_df = leicester_embs_df.rename(columns={"index": "osmnx_node_id"})
leicester_embs_df.to_csv(bulk_storage_directory + "/osmnx/embedded/leicester-1864-gnnuf.csv", index=False)
