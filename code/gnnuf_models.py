# Based on
# https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/autoencoder.py [MIT license]
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/autoencoder.py [MIT license]
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/argva_node_clustering.py [MIT license]

from typing import Optional, Tuple
import warnings

import torch
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, InnerProductDecoder


# ------------------------ #
#   Vanilla GCN encoder    #
# ------------------------ #

# Based on
# https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial6/Tutorial6.ipynb

class VanillaGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, gcn_channels, out_channels):
        super(VanillaGCNEncoder, self).__init__()
        self.en_linear1 = torch.nn.Linear(in_channels, in_channels)
        self.en_conv1 = GCNConv(in_channels, gcn_channels)
        self.en_conv2 = GCNConv(gcn_channels, gcn_channels)
        self.en_linear2 = torch.nn.Linear(gcn_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        # x = self.en_linear1(x).relu()
        x = self.en_conv1(x, edge_index, edge_weight).relu()
        x = self.en_conv2(x, edge_index, edge_weight).relu()
        x = self.en_linear2(x)
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
