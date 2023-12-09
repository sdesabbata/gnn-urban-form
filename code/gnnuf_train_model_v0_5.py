# Base libraries
import numpy as np
import copy
# Torch
from torch_geometric.nn import GAE
from torch_geometric.loader import DataLoader
# GNN models
from gnnuf_models_v0_5 import *
# OSMNx dataset
from osmnx_dataset import *

# OS environment setup
from local_directories import *

# Reset random seeds
random_seed = 2674
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Run on CUDA")
else:
    print("Run on CPU")

dataset_neighbourhood_sample = 0.01
dataset_neighbourhood_min_nodes = 8
dataset_max_distance = 500
dataset_info_str = f"""{dataset_neighbourhood_sample=}
{dataset_neighbourhood_min_nodes=}
{dataset_max_distance=}"""

# Load the data
osmnx_dataset = OSMnxDataset(
    bulk_storage_directory + "/osmnx",
    neighbourhood_sample=dataset_neighbourhood_sample,
    neighbourhood_min_nodes=dataset_neighbourhood_min_nodes,
    max_distance=dataset_max_distance
)
osmnx_dataset_train, osmnx_dataset_test = torch.utils.data.random_split(osmnx_dataset, [0.8, 0.2])
osmnx_loader_train = DataLoader(osmnx_dataset_train, batch_size=32, shuffle=True)
osmnx_loader_test = DataLoader(osmnx_dataset_test, batch_size=32, shuffle=True)

# Define the model
#model_name = "gnnuf_model_v0-4"
#model = GAE(VanillaGCNEncoder(1, 64, 2))
model_name = "gnnuf_model_v0-5"
model = GAE(GINEEncoder(1, 1, 64, 2))
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

model_info_str = str(model)
print(model_name)
print(model_info_str)

# Train
training_log_str = ""
epochs = 1000
best_model = None
best_loss = math.inf
for epoch in range(1, epochs + 1):
    epoch_loss = 0

    model.train()
    for osmnx_graph in osmnx_loader_train:
        # Zero gradients
        optimizer.zero_grad()
        # Encode
        osmnx_graph = osmnx_graph.to(device)
        z = model.encode(osmnx_graph.x, osmnx_graph.edge_index, osmnx_graph.edge_weight)
        # Decode and calculate loss
        loss = model.recon_loss(z, osmnx_graph.edge_index)
        epoch_loss += float(loss)
        # Backpropagate
        loss.backward()
        # Optimization step
        optimizer.step()

    epoch_loss /= len(osmnx_loader_train)
    epoch_loss_str = f"Epoch {epoch}, average loss over {len(osmnx_loader_train)} batches: {epoch_loss}"
    training_log_str += epoch_loss_str + "\n"
    print(epoch_loss_str)
    if epoch_loss < best_loss:
        best_model = copy.deepcopy(model)

# Test
best_model.eval()
test_loss = 0
for osmnx_graph in osmnx_loader_test:
    # Encode
    osmnx_graph = osmnx_graph.to(device)
    z = best_model.encode(osmnx_graph.x, osmnx_graph.edge_index, osmnx_graph.edge_weight)
    # Decode and calculate loss
    loss = best_model.recon_loss(z, osmnx_graph.edge_index)
    test_loss += float(loss)
test_loss /= len(osmnx_loader_test)
test_loss_str = f"Average test loss over {len(osmnx_loader_test)} batches: {test_loss}"
training_log_str += "\n" + test_loss_str + "\n"
print(test_loss_str)

# Save model
torch.save(best_model.state_dict(), this_repo_directory + "/models/" + model_name + ".pt")
with open(this_repo_directory + "/models/" + model_name + "__info.txt", 'wt', encoding='utf-8') as file_info:
    file_info.write("--- Dataset ---\n\n")
    file_info.write(dataset_info_str+"\n\n")
    file_info.write("--- Model ---\n\n")
    file_info.write(model_name+"\n\n")
    file_info.write(model_info_str+"\n\n")
    file_info.write("--- Training ---\n\n")
    file_info.write(training_log_str+"\n\n")
