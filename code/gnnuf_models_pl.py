# Based on
# https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/autoencoder.py [MIT license]
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/autoencoder.py [MIT license]
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/argva_node_clustering.py [MIT license]

# from typing import Optional, Tuple
# import warnings
import math

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.nn import GAE, GCNConv, GATConv, GINEConv


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

class SimpleSparseGINEEncoder(pl.LightningModule):
    def __init__(
        self, 
        in_channels: int = 1,
        edge_dim: int = 1,
        out_channels: int = 2
        ):
        super(SimpleSparseGINEEncoder, self).__init__()

        self.in_channels = in_channels
        self.edge_dim = edge_dim
        self.hot_channels = 4
        self.mid_channels = 32
        self.emb_channels = 256
        self.out_channels = out_channels

        self.onehot_enc = nn.Sequential(
            nn.Linear(self.in_channels, self.hot_channels),
            nn.LeakyReLU(),
            nn.Linear(self.hot_channels, self.mid_channels),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.mid_channels, self.emb_channels)
        )

        self.gineconv_1_lin = nn.Sequential(
            nn.Linear(self.emb_channels, self.emb_channels),
            nn.LeakyReLU(),
            nn.Linear(self.emb_channels, self.emb_channels)
        )
        self.gineconv_1 = GINEConv(
            self.gineconv_1_lin,
            edge_dim=self.edge_dim
        )

        self.gineconv_2_lin = nn.Sequential(
            nn.Linear(self.emb_channels, self.emb_channels),
            nn.LeakyReLU(),
            nn.Linear(self.emb_channels, self.emb_channels)
        )
        self.gineconv_2 = GINEConv(
            self.gineconv_2_lin,
            edge_dim=self.edge_dim
        )

        self.gineconv_3_lin = nn.Sequential(
            nn.Linear(self.emb_channels, self.emb_channels),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.emb_channels, self.emb_channels)
        )
        self.gineconv_3 = GINEConv(
            self.gineconv_3_lin,
            edge_dim=self.edge_dim
        )
        
        self.post_lin = nn.Sequential(
            nn.Linear(self.emb_channels, self.mid_channels),
            nn.LeakyReLU(),
            nn.Linear(self.mid_channels, self.out_channels)
        )

    def forward(self, x, edge_index, edge_weight):
        x = self.onehot_enc(x)
        x = self.gineconv_1(x, edge_index, edge_weight)
        x = self.gineconv_2(x, edge_index, edge_weight)
        x = self.gineconv_3(x, edge_index, edge_weight)
        return self.post_lin(x)


# Define the LightningModule
class GAEModel(pl.LightningModule):
    def __init__(self, 
            in_channels: int = 1,
            edge_dim: int = 1,
            out_channels: int = 2,
            learning_rate: float = 1e-3
        ):
        super().__init__()
        # Initialize the GAE model
        self.in_channels = in_channels
        self.edge_dim = edge_dim
        self.out_channels = out_channels
        self.gae = GAE(
            SimpleSparseGINEEncoder(
                in_channels=self.in_channels,
                edge_dim=self.edge_dim,
                out_channels=self.out_channels
            )
        )
        self.learning_rate = learning_rate
        # self.best_loss = math.inf

    def forward(self, x, edge_index, edge_weight):
        return self.gae.encode(x, edge_index, edge_weight)

    def encode(self, x, edge_index, edge_weight):
        return self.gae.encode(x, edge_index, edge_weight)

    def training_step(self, batch, batch_idx):
        # batch = batch.to(self.device)
        z = self.gae.encode(batch.x, batch.edge_index, batch.edge_weight)
        loss = self.gae.recon_loss(z, batch.edge_index)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        # batch = batch.to(self.device)
        z = self.gae.encode(batch.x, batch.edge_index, batch.edge_weight)
        loss = self.gae.recon_loss(z, batch.edge_index)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.size(0))
        return loss

    def test_step(self, batch, batch_idx):
        # batch = batch.to(self.device)
        z = self.gae.encode(batch.x, batch.edge_index, batch.edge_weight)
        loss = self.gae.recon_loss(z, batch.edge_index)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.size(0))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3, threshold=1e-3, min_lr=1e-7)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'train_loss'}

    def on_save_checkpoint(self, checkpoint):
        # Save any additional information if needed
        checkpoint['model_state'] = self.gae.state_dict()



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
