# Based on
# https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/autoencoder.py [MIT license]
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/autoencoder.py [MIT license]
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/argva_node_clustering.py [MIT license]

from typing import Optional, Tuple
import warnings
import math

import torch
import torch.nn as nn
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
        self.en_linear_out = nn.Linear(gcn_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.en_conv1(x, edge_index, edge_weight).relu()
        x = self.en_conv2(x, edge_index, edge_weight).relu()
        x = self.en_linear_out(x)
        return torch.tanh(x)



# ------------------------------- #
#   Simple Sparse GINE encoder    #
# ------------------------------- #

# ----------------------------------------------------------------------

# TopK activation function
# by Gao et al (2024)
# https://arxiv.org/abs/2406.04093
#
# Based on
# https://github.com/openai/sparse_autoencoder
# MIT license

class TopK(nn.Module):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k
        self.postact_fn = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        # make all other values 0
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result

def normalized_mean_squared_error(
    reconstruction: torch.Tensor,
    original_input: torch.Tensor,
) -> torch.Tensor:
    """
    :param reconstruction: output of Autoencoder.decode (shape: [batch, n_inputs])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :return: normalized mean squared error (shape: [1])
    """
    return (
        ((reconstruction - original_input) ** 2).mean(dim=1) / (original_input**2).mean(dim=1)
    ).mean()

def normalized_L1_loss(
    latent_activations: torch.Tensor,
    original_input: torch.Tensor,
) -> torch.Tensor:
    """
    :param latent_activations: output of Autoencoder.encode (shape: [batch, n_latents])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :return: normalized L1 loss (shape: [1])
    """
    return (latent_activations.abs().sum(dim=1) / original_input.norm(dim=1)).mean()

# ----------------------------------------------------------------------

class SimpleSparseGINEEncoder(torch.nn.Module):
    def __init__(self, in_channels, edge_dim, out_channels):
        super(SimpleSparseGINEEncoder, self).__init__()

        self.hid_channels = 128
        self.emb_channels = 64
        self.emb_topk = 32

        self.onehot_enc = nn.Sequential(
            nn.Linear(in_channels, self.hid_channels),
            nn.LeakyReLU(),
            nn.Linear(self.hid_channels, self.emb_channels),
            TopK(self.emb_topk)
        )

        self.gineconv_1_lin = nn.Sequential(
            nn.Linear(self.emb_channels, self.hid_channels),
            nn.LeakyReLU(),
            nn.Linear(self.hid_channels, self.emb_channels),
            TopK(self.emb_topk)
        )
        self.gineconv_1 = GINEConv(
            self.gineconv_1_lin,
            edge_dim=edge_dim
        )

        self.gineconv_2_lin = nn.Sequential(
            nn.Linear(self.emb_channels, self.hid_channels),
            nn.LeakyReLU(),
            nn.Linear(self.hid_channels, self.emb_channels),
            TopK(self.emb_topk)
        )
        self.gineconv_2 = GINEConv(
            self.gineconv_2_lin,
            edge_dim=edge_dim
        )

        self.post_ltopk = nn.Sequential(
            nn.Linear(self.emb_channels, out_channels),
            TopK(math.floor(out_channels/2))
        )

    def forward(self, x, edge_index, edge_weight):
        x = self.onehot_enc(x)
        x = x + self.gineconv_1(x, edge_index, edge_weight)
        x = x + self.gineconv_2(x, edge_index, edge_weight)
        return self.post_ltopk(x)



# # ---------------------- #
# #   GAT-based encoder    #
# # ---------------------- #
#
# class GATEncoder(torch.nn.Module):
#     def __init__(self,
#             layers_features_in, layers_features_prep, layers_features_gatc, layers_features_post,
#             num_gat_heads, negative_slope, dropout
#         ):
#         super(GATEncoder, self).__init__()
#
#         self.negative_slope = negative_slope
#
#         # --- Prep layers ---
#         if len(layers_features_prep) == 0:
#             self.sequential_prep = None
#         else:
#             self.sequential_prep =nn.Sequential()
#             self.sequential_prep.append(torch.nn.Linear(layers_features_in, layers_features_prep[0]))
#             self.sequential_prep.append(torch.nn.LeakyReLU(negative_slope=negative_slope))
#             for i in range(len(layers_features_prep) - 1):
#                 self.sequential_prep.append(torch.nn.Linear(layers_features_prep[i], layers_features_prep[i + 1]))
#                 self.sequential_prep.append(torch.nn.LeakyReLU(negative_slope=negative_slope))
#
#         # --- GAT layers ---
#         self.gatconvs =nn.ModuleList()
#         # GAT layers except last
#         for layers_features_gatc_h in layers_features_gatc[:-1]:
#             self.gatconvs.append(GATConv(
#                 -1, layers_features_gatc_h,
#                 heads=num_gat_heads, negative_slope=negative_slope, dropout=dropout
#             ))
#         # Last GAT layer, averaged instead of concatenated
#         self.gatconvs.append(GATConv(
#             -1, layers_features_gatc[-1],
#             heads=num_gat_heads, concat=False, negative_slope=negative_slope, dropout=dropout
#         ))
#
#         # --- Final layers ---
#         self.sequential_post =nn.Sequential()
#         self.sequential_post.append(torch.nn.Linear(layers_features_gatc[-1], layers_features_post[0]))
#         self.sequential_post.append(torch.nn.LeakyReLU(negative_slope=negative_slope))
#         for i in range(len(layers_features_post) - 1):
#             self.sequential_post.append(torch.nn.Linear(layers_features_post[i], layers_features_post[i + 1]))
#
#         self.final_batch_norm_1d =nn.BatchNorm1d(layers_features_post[-1])
#
#     def forward(self, attributes, edge_index):
#         if self.sequential_prep is not None:
#             attributes = self.sequential_prep(attributes)
#         for a_gatconv in self.gatconvs:
#             attributes = a_gatconv(attributes, edge_index)
#         attributes = self.sequential_post(attributes)
#         attributes = self.final_batch_norm_1d(attributes)
#         return attributes
