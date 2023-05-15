# Base libraries
import numpy as np
# Torch
from torch_geometric.nn import GAE
from torch_geometric.loader import DataLoader
# GNN models
from gnnuf_models import *
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

# Load the data
osmnx_dataset = OSMxDataset(
    bulk_storage_directory + "/osmnx",
    neighbourhood_sample=0.05,
    neighbourhood_min_nodes=16,
    max_distance=2000,
)
osmnx_dataset_train, osmnx_dataset_test = torch.utils.data.random_split(osmnx_dataset, [0.8, 0.2])
osmnx_loader_train = DataLoader(osmnx_dataset_train, batch_size=32, shuffle=True)
osmnx_loader_test = DataLoader(osmnx_dataset_test, batch_size=32, shuffle=True)

# Define the model
model = GAE(VanillaGCNEncoder(2, 128, 64))
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
print(model)

# Train
epochs = 1000
best_model = None
best_loss = math.Inf
for epoch in range(1, epochs + 1):
    print(f"{epoch=}")
    epoch_loss = 0

    model.train()
    for osmnx_linegraph in osmnx_loader_train:
        # Zero gradients
        optimizer.zero_grad()
        # Encode
        osmnx_linegraph = osmnx_linegraph.to(device)
        z = model.encode(osmnx_linegraph.x, osmnx_linegraph.edge_index)
        # Decode and calculate loss
        loss = model.recon_loss(z, osmnx_linegraph.edge_index)
        epoch_loss += float(loss)
        # Backpropagate
        loss.backward()
        # Optimization step
        optimizer.step()

    epoch_loss /= len(osmnx_loader_train)
    print(f"Average epoch loss over {len(osmnx_loader_train)} batches: {epoch_loss}")
    if epoch_loss < best_loss:
        best_model = model.deepcopy()

# Test
best_model.eval()
test_loss = 0
for osmnx_linegraph in osmnx_loader_test:
    # Encode
    osmnx_linegraph = osmnx_linegraph.to(device)
    z = best_model.encode(osmnx_linegraph.x, osmnx_linegraph.edge_index)
    # Decode and calculate loss
    loss = best_model.recon_loss(z, osmnx_linegraph.edge_index)
    test_loss += float(loss)
test_loss /= len(osmnx_loader_test)
print(f"Average test loss over {len(osmnx_loader_test)} batches: {test_loss}")

# Save model
torch.save(best_model.state_dict(), bulk_storage_directory + "/models/gnnuf_model_v0-1.pt")
