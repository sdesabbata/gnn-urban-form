# Base libraries
import numpy as np
import copy
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

dataset_neighbourhood_sample = 0.05
dataset_neighbourhood_min_nodes = 8
dataset_max_distance = 500
dataset_info_str = f"""{dataset_neighbourhood_sample=}
{dataset_neighbourhood_min_nodes=}
{dataset_max_distance=}"""

# Load the data
osmnx_dataset = OSMnxDataset(
    bulk_storage_directory + "/osmnx_005",
    neighbourhood_sample=dataset_neighbourhood_sample,
    neighbourhood_min_nodes=dataset_neighbourhood_min_nodes,
    max_distance=dataset_max_distance
)
osmnx_dataset_train, osmnx_dataset_test = torch.utils.data.random_split(osmnx_dataset, [0.8, 0.2])
osmnx_loader_train = DataLoader(osmnx_dataset_train, batch_size=32, shuffle=True)
osmnx_loader_test = DataLoader(osmnx_dataset_test, batch_size=32, shuffle=True)


