# Based on
# https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/autoencoder.py [MIT license]
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/autoencoder.py [MIT license]
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/argva_node_clustering.py [MIT license]

from typing import Optional, Tuple
import warnings

import torch
from torch_geometric.nn import GCNConv, GATConv, GINEConv


# ------------------------ #
#   Vanilla GCN encoder    #
# ------------------------ #

# Based on
# https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial6/Tutorial6.ipynb

class VanillaGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, gcn_channels, out_channels):
        super(VanillaGCNEncoder, self).__init__()
        self.en_conv1 = GCNConv(in_channels, gcn_channels)
        self.en_conv2 = GCNConv(gcn_channels, gcn_channels)
        self.en_linear_out = torch.nn.Linear(gcn_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.en_conv1(x, edge_index, edge_weight).relu()
        x = self.en_conv2(x, edge_index, edge_weight).relu()
        x = self.en_linear_out(x)
        return torch.tanh(x)


# ----------------- #
#   GINE encoder    #
# ----------------- #

# Based on
# https://github.com/pyg-team/pytorch_geometric/blob/11513fdde087d001e15c2eda5ff3c07c2240e1c0/examples/graph_gps.py

class GINEEncoder(torch.nn.Module):
    def __init__(self, in_channels, edge_dim, gine_mlp_channels, out_channels):
        super(GINEEncoder, self).__init__()

        self.gine_convs = torch.nn.ModuleList()
        #for layers_features in gin_mlp_channels:
        #    self.gine_convs.append(GINEConv(
        #        torch.nn.Sequential(
        #            torch.nn.Linear(layers_features[0], layers_features[1]),
        #            torch.nn.ReLU(),
        #            torch.nn.Linear(layers_features[1], layers_features[2]),
        #            torch.nn.ReLU()
        #        ),
        #        edge_dim=edge_dim
        #    ))
        for mlp_channels in gine_mlp_channels:
            mlp_layers = []
            for i in range(len(mlp_channels) - 1):
                mlp_layers += [
                    torch.nn.Linear(mlp_channels[i], mlp_channels[i + 1]),
                    torch.nn.ReLU(),
                    #torch.nn.BatchNorm2d(channel_list[i + 1])
                ]
            self.gine_convs.append(
                GINEConv(
                    torch.nn.Sequential(*mlp_layers),
                    edge_dim=edge_dim
                )
            )

        self.en_linear_post = torch.nn.Linear(gine_mlp_channels[-1][-1], out_channels)

    def forward(self, x, edge_index, edge_weight):
        for conv in self.gine_convs:
            x = conv(x, edge_index, edge_weight)
        x = self.en_linear_post(x)
        return torch.tanh(x)

# ---------------------- #
#   GAT-based encoder    #
# ---------------------- #

class GATEncoder(torch.nn.Module):
    def __init__(self,
            layers_features_in, layers_features_prep, layers_features_gatc, layers_features_post,
            num_gat_heads, negative_slope, dropout
        ):
        super(GATEncoder, self).__init__()

        self.negative_slope = negative_slope

        # --- Prep layers ---
        if len(layers_features_prep) == 0:
            self.sequential_prep = None
        else:
            self.sequential_prep = torch.nn.Sequential()
            self.sequential_prep.append(torch.nn.Linear(layers_features_in, layers_features_prep[0]))
            self.sequential_prep.append(torch.nn.LeakyReLU(negative_slope=negative_slope))
            for i in range(len(layers_features_prep) - 1):
                self.sequential_prep.append(torch.nn.Linear(layers_features_prep[i], layers_features_prep[i + 1]))
                self.sequential_prep.append(torch.nn.LeakyReLU(negative_slope=negative_slope))

        # --- GAT layers ---
        self.gatconvs = torch.nn.ModuleList()
        # GAT layers except last
        for layers_features_gatc_h in layers_features_gatc[:-1]:
            self.gatconvs.append(GATConv(
                -1, layers_features_gatc_h,
                heads=num_gat_heads, negative_slope=negative_slope, dropout=dropout
            ))
        # Last GAT layer, averaged instead of concatenated
        self.gatconvs.append(GATConv(
            -1, layers_features_gatc[-1],
            heads=num_gat_heads, concat=False, negative_slope=negative_slope, dropout=dropout
        ))

        # --- Final layers ---
        self.sequential_post = torch.nn.Sequential()
        self.sequential_post.append(torch.nn.Linear(layers_features_gatc[-1], layers_features_post[0]))
        self.sequential_post.append(torch.nn.LeakyReLU(negative_slope=negative_slope))
        for i in range(len(layers_features_post) - 1):
            self.sequential_post.append(torch.nn.Linear(layers_features_post[i], layers_features_post[i + 1]))

        self.final_batch_norm_1d = torch.nn.BatchNorm1d(layers_features_post[-1])

    def forward(self, attributes, edge_index):
        if self.sequential_prep is not None:
            attributes = self.sequential_prep(attributes)
        for a_gatconv in self.gatconvs:
            attributes = a_gatconv(attributes, edge_index)
        attributes = self.sequential_post(attributes)
        attributes = self.final_batch_norm_1d(attributes)
        return attributes
