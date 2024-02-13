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

class GINEEncoderPP(torch.nn.Module):
    def __init__(self,
            in_channels,
            edge_dim,
            layers_features_prep,
            gine_mlp_channels,
            layers_features_post,
            dropout,
            negative_slope
        ):
        super(GINEEncoderPP, self).__init__()

        # --- Prep layers ---
        self.sequential_prep = torch.nn.Sequential()
        # If not prep layers, then None
        if len(layers_features_prep) == 0:
            self.sequential_prep = None
        # Otherwise
        else:
            # Add layer from input channel size to first hidden channels
            self.sequential_prep.append(torch.nn.Linear(in_channels, layers_features_prep[0]))
            self.sequential_prep.append(torch.nn.LeakyReLU(negative_slope=negative_slope))
            if dropout > 0:
                self.sequential_prep.append(torch.nn.Dropout(dropout))
            # Add hidden layers
            for i in range(len(layers_features_prep) - 1):
                self.sequential_prep.append(torch.nn.Linear(layers_features_prep[i], layers_features_prep[i + 1]))
                # If it is the last layer
                if i == (len(layers_features_prep) - 2):
                    self.sequential_prep.append(torch.nn.Tanh())
                # Otherwise add leakyReLU and dropout:
                else:
                    self.sequential_prep.append(torch.nn.LeakyReLU(negative_slope=negative_slope))
                    if dropout > 0:
                        self.sequential_prep.append(torch.nn.Dropout(dropout))

        # --- GINE layers ---
        self.gine_convs = torch.nn.ModuleList()
        for i in range(len(gine_mlp_channels)):
            mlp_layers = []
            # Setup first MLP layer for first GINE layer
            if i == 0:
                # If there are no prep layers
                if len(layers_features_prep) == 0:
                    # Add first MLP layer of first GINE layer from input channels
                    mlp_layers += [
                        torch.nn.Linear(in_channels, gine_mlp_channels[i][0]),
                        torch.nn.LeakyReLU(negative_slope=negative_slope)
                    ]
                    if dropout > 0:
                        mlp_layers += [torch.nn.Dropout(dropout)]
                # Otherwise, if there are prep layers
                else:
                    # Add first MLP layer of first GINE layer from last prep layer channels
                    mlp_layers += [
                        torch.nn.Linear(layers_features_prep[-1], gine_mlp_channels[i][0]),
                        torch.nn.LeakyReLU(negative_slope=negative_slope)
                    ]
                    if dropout > 0:
                        mlp_layers += [torch.nn.Dropout(dropout)]
            # Setup first MLP layer for subsequent GINE layers
            else:
                # First MLP layer of subsequent GINE layers
                mlp_layers += [
                    torch.nn.Linear(gine_mlp_channels[i-1][-1], gine_mlp_channels[i][0]),
                    torch.nn.LeakyReLU(negative_slope=negative_slope)
                ]
                if dropout > 0:
                        mlp_layers += [torch.nn.Dropout(dropout)]
            # Setup subsequent MLP layers for all GINE layers
            for j in range(len(gine_mlp_channels[i]) - 1):
                mlp_layers += [
                    torch.nn.Linear(gine_mlp_channels[i][j], gine_mlp_channels[i][j + 1])
                ]
                # If it is the last MLP layer, add tanh activation
                if j == (len(gine_mlp_channels[i]) - 2):
                    mlp_layers += [torch.nn.Tanh()]
                # otherwise, for hidden layers, add leakyReLU and dropout
                else:
                    mlp_layers += [torch.nn.LeakyReLU(negative_slope=negative_slope)]
                    if dropout > 0:
                        mlp_layers += [torch.nn.Dropout(dropout)]
            # Add GINE layer
            self.gine_convs.append(
                GINEConv(torch.nn.Sequential(*mlp_layers), edge_dim=edge_dim)
            )

        # --- Final layers ---
        self.sequential_post = torch.nn.Sequential()
        # If not prep layers, then None
        if len(layers_features_post) == 0:
            self.sequential_post = None
        # Otherwise
        else:
            # Add layer from last GINE to first post
            self.sequential_post.append(torch.nn.Linear(gine_mlp_channels[-1][-1], layers_features_post[0]))
            # If there are more layers afterward, add leakyReLU and dropout
            if len(layers_features_post) > 1:
                self.sequential_post.append(torch.nn.LeakyReLU(negative_slope=negative_slope))
                if dropout > 0:
                    self.sequential_post.append(torch.nn.Dropout(dropout))
            # Add subsequent layers, if any
            for i in range(len(layers_features_post) - 1):
                self.sequential_post.append(torch.nn.Linear(layers_features_post[i], layers_features_post[i+1]))
                # If there are more layers afterward, add leakyReLU and dropout
                if len(layers_features_post) > 2 and i < (len(layers_features_post) - 2):
                    self.sequential_post.append(torch.nn.LeakyReLU(negative_slope=negative_slope))
                    if dropout > 0:
                        self.sequential_post.append(torch.nn.Dropout(dropout))
            # Add final tanh activation
            self.sequential_post.append(torch.nn.Tanh())

    def forward(self, x, edge_index, edge_weight):
        if self.sequential_prep is not None:
            x = self.sequential_prep(x)
        for conv in self.gine_convs:
            x = conv(x, edge_index, edge_weight)
        if self.sequential_post is not None:
            x = self.sequential_post(x)
        return x


class GINEEncoderPPwSkip(torch.nn.Module):
    def __init__(self,
            in_channels,
            edge_dim,
            layers_features_prep,
            gine_mlp_channels,
            layers_features_post,
            dropout,
            negative_slope
        ):
        super(GINEEncoderPPwSkip, self).__init__()

        # --- Prep layers ---
        self.sequential_prep = torch.nn.Sequential()
        # If not prep layers, then None
        if len(layers_features_prep) == 0:
            self.sequential_prep = None
        # Otherwise
        else:
            # Add layer from input channel size to first hidden channels
            self.sequential_prep.append(torch.nn.Linear(in_channels, layers_features_prep[0]))
            self.sequential_prep.append(torch.nn.LeakyReLU(negative_slope=negative_slope))
            if dropout > 0:
                self.sequential_prep.append(torch.nn.Dropout(dropout))
            # Add hidden layers
            for i in range(len(layers_features_prep) - 1):
                self.sequential_prep.append(torch.nn.Linear(layers_features_prep[i], layers_features_prep[i + 1]))
                # If it is the last layer
                if i == (len(layers_features_prep) - 2):
                    self.sequential_prep.append(torch.nn.Tanh())
                # Otherwise add leakyReLU and dropout:
                else:
                    self.sequential_prep.append(torch.nn.LeakyReLU(negative_slope=negative_slope))
                    if dropout > 0:
                        self.sequential_prep.append(torch.nn.Dropout(dropout))

        # --- GINE layers ---
        self.gine_convs = torch.nn.ModuleList()
        self.skips = torch.nn.ModuleList()
        for i in range(len(gine_mlp_channels)):
            mlp_layers = []
            # Setup first MLP layer for first GINE layer
            if i == 0:
                # If there are no prep layers
                if len(layers_features_prep) == 0:
                    # Add first MLP layer of first GINE layer from input channels
                    mlp_layers += [
                        torch.nn.Linear(in_channels, gine_mlp_channels[i][0]),
                        torch.nn.LeakyReLU(negative_slope=negative_slope)
                    ]
                    if dropout > 0:
                        mlp_layers += [torch.nn.Dropout(dropout)]
                    self.skips.append(torch.nn.Sequential(
                        torch.nn.Linear(in_channels, gine_mlp_channels[i][-1]),
                        torch.nn.Tanh()))
                # Otherwise, if there are prep layers
                else:
                    # Add first MLP layer of first GINE layer from last prep layer channels
                    mlp_layers += [
                        torch.nn.Linear(layers_features_prep[-1], gine_mlp_channels[i][0]),
                        torch.nn.LeakyReLU(negative_slope=negative_slope)
                    ]
                    if dropout > 0:
                        mlp_layers += [torch.nn.Dropout(dropout)]
                    self.skips.append(torch.nn.Sequential(
                        torch.nn.Linear(layers_features_prep[-1], gine_mlp_channels[i][-1]),
                        torch.nn.Tanh()))
            # Setup first MLP layer for subsequent GINE layers
            else:
                # First MLP layer of subsequent GINE layers
                mlp_layers += [
                    torch.nn.Linear(gine_mlp_channels[i-1][-1], gine_mlp_channels[i][0]),
                    torch.nn.LeakyReLU(negative_slope=negative_slope)
                ]
                if dropout > 0:
                        mlp_layers += [torch.nn.Dropout(dropout)]
                self.skips.append(torch.nn.Sequential(
                    torch.nn.Linear(gine_mlp_channels[i-1][-1], gine_mlp_channels[i][-1]),
                    torch.nn.Tanh()))
            # Setup first MLP layer for subsequent GINE layers
            # Setup subsequent MLP layers for all GINE layers
            for j in range(len(gine_mlp_channels[i]) - 1):
                mlp_layers += [
                    torch.nn.Linear(gine_mlp_channels[i][j], gine_mlp_channels[i][j + 1])
                ]
                # If it is the last MLP layer, add tanh activation
                if j == (len(gine_mlp_channels[i]) - 2):
                    mlp_layers += [torch.nn.Tanh()]
                # otherwise, for hidden layers, add leakyReLU and dropout
                else:
                    mlp_layers += [torch.nn.LeakyReLU(negative_slope=negative_slope)]
                    if dropout > 0:
                        mlp_layers += [torch.nn.Dropout(dropout)]
            # Add GINE layer
            self.gine_convs.append(
                GINEConv(torch.nn.Sequential(*mlp_layers), edge_dim=edge_dim)
            )

        # --- Final layers ---
        self.sequential_post = torch.nn.Sequential()
        # If not prep layers, then None
        if len(layers_features_post) == 0:
            self.sequential_post = None
        # Otherwise
        else:
            # Add layer from last GINE to first post
            self.sequential_post.append(torch.nn.Linear(gine_mlp_channels[-1][-1], layers_features_post[0]))
            # If there are more layers afterward, add leakyReLU and dropout
            if len(layers_features_post) > 1:
                self.sequential_post.append(torch.nn.LeakyReLU(negative_slope=negative_slope))
                if dropout > 0:
                    self.sequential_post.append(torch.nn.Dropout(dropout))
            # Add subsequent layers, if any
            for i in range(len(layers_features_post) - 1):
                self.sequential_post.append(torch.nn.Linear(layers_features_post[i], layers_features_post[i+1]))
                # If there are more layers afterward, add leakyReLU and dropout
                if len(layers_features_post) > 2 and i < (len(layers_features_post) - 2):
                    self.sequential_post.append(torch.nn.LeakyReLU(negative_slope=negative_slope))
                    if dropout > 0:
                        self.sequential_post.append(torch.nn.Dropout(dropout))
            # Add final tanh activation
            self.sequential_post.append(torch.nn.Tanh())

    def forward(self, x, edge_index, edge_weight):
        if self.sequential_prep is not None:
            x = self.sequential_prep(x)
        for i in range(len(self.gine_convs)):
            r = x.clone()
            x = self.gine_convs[i](x, edge_index, edge_weight) + self.skips[i](r)
        if self.sequential_post is not None:
            x = self.sequential_post(x)
        return x


class GINEEncoderPPwSkipCat(torch.nn.Module):
    def __init__(self,
            in_channels,
            edge_dim,
            layers_features_prep,
            gine_mlp_channels,
            layers_features_post,
            dropout,
            negative_slope
        ):
        super(GINEEncoderPPwSkipCat, self).__init__()

        # # Add room for skip connections
        # for j in range(len(gine_mlp_channels)):
        #     if j < (len(gine_mlp_channels) - 1):
        #         gine_mlp_channels[j+1][0] = gine_mlp_channels[j][0] + gine_mlp_channels[j][-1]
        # layers_features_post[0] = gine_mlp_channels[-1][0] + gine_mlp_channels[-1][-1]

        # --- Prep layers ---
        self.sequential_prep = torch.nn.Sequential()
        # If not prep layers, then None
        if len(layers_features_prep) == 0:
            self.sequential_prep = None
        # Otherwise
        else:
            # Add layer from input channel size to first hidden channels
            self.sequential_prep.append(torch.nn.Linear(in_channels, layers_features_prep[0]))
            self.sequential_prep.append(torch.nn.LeakyReLU(negative_slope=negative_slope))
            if dropout > 0:
                self.sequential_prep.append(torch.nn.Dropout(dropout))
            # Add hidden layers
            for i in range(len(layers_features_prep) - 1):
                self.sequential_prep.append(torch.nn.Linear(layers_features_prep[i], layers_features_prep[i + 1]))
                # If it is the last layer
                if i == (len(layers_features_prep) - 2):
                    self.sequential_prep.append(torch.nn.Tanh())
                # Otherwise add leakyReLU and dropout:
                else:
                    self.sequential_prep.append(torch.nn.LeakyReLU(negative_slope=negative_slope))
                    if dropout > 0:
                        self.sequential_prep.append(torch.nn.Dropout(dropout))

        # --- GINE layers ---
        self.gine_convs = torch.nn.ModuleList()
        for i in range(len(gine_mlp_channels)):
            mlp_layers = []
            # Setup first MLP layer for first GINE layer
            if i == 0:
                # If there are no prep layers
                if len(layers_features_prep) == 0:
                    # Add first MLP layer of first GINE layer from input channels
                    mlp_layers += [
                        torch.nn.Linear(in_channels, gine_mlp_channels[i][0]),
                        torch.nn.LeakyReLU(negative_slope=negative_slope)
                    ]
                    if dropout > 0:
                        mlp_layers += [torch.nn.Dropout(dropout)]
                # Otherwise, if there are prep layers
                else:
                    # Add first MLP layer of first GINE layer from last prep layer channels
                    mlp_layers += [
                        torch.nn.Linear(layers_features_prep[-1], gine_mlp_channels[i][0]),
                        torch.nn.LeakyReLU(negative_slope=negative_slope)
                    ]
                    if dropout > 0:
                        mlp_layers += [torch.nn.Dropout(dropout)]
            # Setup first MLP layer for subsequent GINE layers
            # with room for skip connections (the first to receive skip connections)
            else:
                # First MLP layer of subsequent GINE layers
                # If this is the second layer
                if i == 1:
                    # If there are no prep layers, add the input channel size
                    if len(layers_features_prep) == 0:
                        mlp_layers += [torch.nn.Linear(gine_mlp_channels[i - 1][-1] + in_channels, gine_mlp_channels[i][0])]
                    # Otherwise add the last prep layer size
                    else:
                        mlp_layers += [torch.nn.Linear(gine_mlp_channels[i - 1][-1] + layers_features_prep[-1], gine_mlp_channels[i][0])]
                # Otherwise add the output size of the layer before the previous one
                else:
                    mlp_layers += [
                        torch.nn.Linear(gine_mlp_channels[i-1][-1] + 
                        gine_mlp_channels[i-2][-1], 
                        gine_mlp_channels[i][0])]
                # Add leakeyReLU and Dropput
                mlp_layers += [torch.nn.LeakyReLU(negative_slope=negative_slope)]
                if dropout > 0:
                        mlp_layers += [torch.nn.Dropout(dropout)]
            # Setup subsequent MLP layers for all GINE layers
            for j in range(len(gine_mlp_channels[i]) - 1):
                mlp_layers += [
                    torch.nn.Linear(gine_mlp_channels[i][j], gine_mlp_channels[i][j + 1])
                ]
                # If it is the last MLP layer, add tanh activation
                if j == (len(gine_mlp_channels[i]) - 2):
                    mlp_layers += [torch.nn.Tanh()]
                # otherwise, for hidden layers, add leakyReLU and dropout
                else:
                    mlp_layers += [torch.nn.LeakyReLU(negative_slope=negative_slope)]
                    if dropout > 0:
                        mlp_layers += [torch.nn.Dropout(dropout)]
            # Add GINE layer
            self.gine_convs.append(
                GINEConv(torch.nn.Sequential(*mlp_layers), edge_dim=edge_dim)
            )

        # --- Final layers ---
        self.sequential_post = torch.nn.Sequential()
        # If not prep layers, add just one to collect the skip connections
        if len(layers_features_post) == 0:
            # If there is only one GINE layer
            if len(gine_mlp_channels) == 1:
                # If there are no prep layers, add the input channel size
                if len(layers_features_prep) == 0:
                    self.sequential_post.append(
                        torch.nn.Linear(gine_mlp_channels[-1][-1] + in_channels, gine_mlp_channels[-1][-1]))
                # Otherwise add the last prep layer size
                else:
                    self.sequential_post.append(
                        torch.nn.Linear(gine_mlp_channels[-1][-1] + layers_features_prep[-1], gine_mlp_channels[-1][-1]))
            # Otherwise add the output size of the layer before the last one
            else:
                self.sequential_post.append(
                    torch.nn.Linear(gine_mlp_channels[-1][-1] + gine_mlp_channels[-2][-1], gine_mlp_channels[-1][-1]))
            # Add final tanh activation
            self.sequential_post.append(torch.nn.Tanh())
        # Otherwise
        else:
            # Add layer from last GINE to first post, with skip connections
            # If there is only one GINE layer
            if len(gine_mlp_channels) == 1:
                # If there are no prep layers, add the input channel size
                if len(layers_features_prep) == 0:
                    self.sequential_post.append(
                        torch.nn.Linear(gine_mlp_channels[-1][-1] + in_channels, layers_features_post[0]))
                # Otherwise add the last prep layer size
                else:
                    self.sequential_post.append(
                        torch.nn.Linear(gine_mlp_channels[-1][-1] + layers_features_prep[-1], layers_features_post[0]))
            # Otherwise add the output size of the layer before the last one
            else:
                self.sequential_post.append(
                    torch.nn.Linear(gine_mlp_channels[-1][-1] + gine_mlp_channels[-2][-1], layers_features_post[0]))
            # If there are more layers afterward, add leakyReLU and dropout
            if len(layers_features_post) > 1:
                self.sequential_post.append(torch.nn.LeakyReLU(negative_slope=negative_slope))
                if dropout > 0:
                    self.sequential_post.append(torch.nn.Dropout(dropout))
            # Add subsequent layers, if any
            for i in range(len(layers_features_post) - 1):
                self.sequential_post.append(torch.nn.Linear(layers_features_post[i], layers_features_post[i+1]))
                # If there are more layers afterward, add leakyReLU and dropout
                if len(layers_features_post) > 2 and i < (len(layers_features_post) - 2):
                    self.sequential_post.append(torch.nn.LeakyReLU(negative_slope=negative_slope))
                    if dropout > 0:
                        self.sequential_post.append(torch.nn.Dropout(dropout))
            # Add final tanh activation
            self.sequential_post.append(torch.nn.Tanh())

    def forward(self, x, edge_index, edge_weight):
        if self.sequential_prep is not None:
            x = self.sequential_prep(x)
        for conv in self.gine_convs:
            r = x.clone()
            x = conv(x, edge_index, edge_weight)
            #print("...adding skip connections")
            #print(x.size())
            x = torch.cat((x, r), dim=1)
            #print(x.size())
            #print("\n")
        return self.sequential_post(x)


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
#             self.sequential_prep = torch.nn.Sequential()
#             self.sequential_prep.append(torch.nn.Linear(layers_features_in, layers_features_prep[0]))
#             self.sequential_prep.append(torch.nn.LeakyReLU(negative_slope=negative_slope))
#             for i in range(len(layers_features_prep) - 1):
#                 self.sequential_prep.append(torch.nn.Linear(layers_features_prep[i], layers_features_prep[i + 1]))
#                 self.sequential_prep.append(torch.nn.LeakyReLU(negative_slope=negative_slope))
#
#         # --- GAT layers ---
#         self.gatconvs = torch.nn.ModuleList()
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
#         self.sequential_post = torch.nn.Sequential()
#         self.sequential_post.append(torch.nn.Linear(layers_features_gatc[-1], layers_features_post[0]))
#         self.sequential_post.append(torch.nn.LeakyReLU(negative_slope=negative_slope))
#         for i in range(len(layers_features_post) - 1):
#             self.sequential_post.append(torch.nn.Linear(layers_features_post[i], layers_features_post[i + 1]))
#
#         self.final_batch_norm_1d = torch.nn.BatchNorm1d(layers_features_post[-1])
#
#     def forward(self, attributes, edge_index):
#         if self.sequential_prep is not None:
#             attributes = self.sequential_prep(attributes)
#         for a_gatconv in self.gatconvs:
#             attributes = a_gatconv(attributes, edge_index)
#         attributes = self.sequential_post(attributes)
#         attributes = self.final_batch_norm_1d(attributes)
#         return attributes
