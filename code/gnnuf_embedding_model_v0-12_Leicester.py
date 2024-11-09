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
# GNN models and utils
from gnnuf_models_pl import *
from gnnuf_utils import *
# OS environment setup
from local_directories import *



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
model_name = "gnnuf_model_v0-12"
gaemodel_in_channels = 1
gaemodel_edge_dim = 1
gaemodel_out_channels = 2
model = GAEModel(
    in_channels=gaemodel_in_channels,
    edge_dim=gaemodel_edge_dim,
    out_channels=gaemodel_out_channels
)
out_channels_for_df = gaemodel_out_channels
model.load_state_dict(torch.load(this_repo_directory + "/models/" + model_name + ".pt", map_location=device))
model = model.to(device)
model.eval()

model_info_str = str(model)
print(model_name)
print(model_info_str)

# Load Leciester's graph
leicester = ox.io.load_graphml(bulk_storage_directory + "/osmnx/raw/leicester-1864.graphml")

leicester_embs = {}
neighbourhood_min_nodes = 8
max_distance = 500
count = 0

print("Embedding nodes...")
for node in leicester.nodes:
    print(f"\t{node}")

    # Create the corresponding ego graph
    node_ego_graph = nx.generators.ego_graph(leicester, node, radius=max_distance, undirected=True, distance="length")

    # Only keep the sampled area if it has a minimum number of nodes
    if len(node_ego_graph.nodes()) > neighbourhood_min_nodes:

        # Convert linegraph to Pytorch Geometric linegraph
        #node_pyg = to_pyg_linegraph(node_ego_graph, max_distance)
        node_pyg, node_index = to_pyg_graph(
            node,
            node_ego_graph,
            node_attr_names=["street_count"],
            node_attr_min_max={"street_count": (1, 4)},
            node_attr_min_max_limit=True,
            edge_attr_names=["length"],
            edge_attr_min_max={"length": (50, 500)},
            edge_attr_min_max_limit=True,
        )
        if node_pyg is not None:
            node_pyg = node_pyg.to(device)

            # Encode
            node_pyg_emb = model.encode(node_pyg.x, node_pyg.edge_index, node_pyg.edge_weight)
            #node_pyg_emb_gmp = global_mean_pool(node_pyg_emb, None)
            #leicester_embs[node] = np.squeeze(node_pyg_emb_gmp.cpu().detach().numpy())
            leicester_embs[node] = node_pyg_emb.cpu().detach().numpy()[node_index, ]

        else:
            print("PyG graph is None")

# Save
leicester_embs_df = pd.DataFrame.from_dict(leicester_embs, orient="index", columns=[f"EMB{i:03d}" for i in range(out_channels_for_df)])
leicester_embs_df = leicester_embs_df.reset_index()
leicester_embs_df = leicester_embs_df.rename(columns={"index": "osmnx_node_id"})
leicester_embs_df.to_csv(this_repo_directory + "/data/leicester-1864_emb_" + model_name + ".csv", index=False)
