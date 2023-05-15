# Base libraries
import os
from os.path import isfile, join
import math
import numpy as np
import random
# NetworkX
import networkx as nx
import osmnx as ox
# Torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import GAE
from torch_geometric.utils import from_networkx
# GNN models
from gnnuf_models import *
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


class OSMxDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # Just check if Leicester's graph is there
        return ["leicester-1864.graphml"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        print("Haven't got time to implement this. Please download the data manually -.- ")

    def process(self):
        graphml_file_names = [
            join(osmnx_dir + "/raw", f)
            for f in os.listdir(osmnx_dir + "/raw")
            if f[-8:] == ".graphml"
            if isfile(join(osmnx_dir + "/raw", f))]

        neighbourhoods_list = []

        for graphml_file_name in graphml_file_names:

            print(f"Processing: {graphml_file_name}")

            G = ox.io.load_graphml(graphml_file_name)

            # Sample 1% of nodes
            sample_nodes = random.sample(list(G.nodes), math.ceil(len(G.nodes) * 0.01))
            # sample_nodes = random.sample(list(G.nodes), math.ceil(10))

            for sampled_node in sample_nodes:

                sampled_ego_graph = nx.generators.ego_graph(
                    G, sampled_node,
                    radius=2000, undirected=True, distance="length"
                )

                if len(sampled_ego_graph.nodes) > 32:

                    seg_edges_length = nx.get_edge_attributes(sampled_ego_graph, "length")
                    seg_edges_edges_grade_abs = nx.get_edge_attributes(sampled_ego_graph, "grade_abs")

                    seg_linegraph = nx.line_graph(sampled_ego_graph)
                    for seglg_node in seg_linegraph.nodes():
                        seg_linegraph.nodes[seglg_node]["x"] = [
                            seg_edges_length[(seglg_node[0], seglg_node[1], 0)],
                            seg_edges_edges_grade_abs[(seglg_node[0], seglg_node[1], 0)]
                        ]
                    del seglg_node

                    pyg_graph = from_networkx(seg_linegraph)
                    neighbourhoods_list.append(pyg_graph)

        self.data, self.slices = self.collate(neighbourhoods_list)
        torch.save((self.data, self.slices), self.processed_paths[0])


osmnx_dir = bulk_storage_directory + "/osmnx"
osmnx_dataset = OSMxDataset(osmnx_dir)


from gnnuf_models import VanillaGCNEncoder

model = GAE(VanillaGCNEncoder(2, 128, 64))
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

print(model)

from torch_geometric.loader import DataLoader
osmnx_loader = DataLoader(osmnx_dataset, batch_size=32, shuffle=True)

epochs = 100
for epoch in range(1, epochs + 1):
    print(f"{epoch=}")

    model.train()

    for osmnx_linegraph in osmnx_loader:
        optimizer.zero_grad()
        osmnx_linegraph = osmnx_linegraph.to(device)
        z = model.encode(osmnx_linegraph.x, osmnx_linegraph.edge_index)
        loss = model.recon_loss(z, osmnx_linegraph.edge_index)
        osmnx_linegraph_loss = float(loss)
        print(f"{osmnx_linegraph_loss=}")
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), bulk_storage_directory + "/models/osmnx_model_v1.pt")
