# Base libraries
import numpy as np
import copy
# Torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

# GNN models
from gnnuf_models_pl import *
# OSMNx dataset
from osmnx_dataset import *

# OS environment setup
from local_directories import *

# Set random seeds
# random_seed = 689028967 used for model v1.0 emb2 and emb8, diverging for emb32
# random_seed = 711102427 used for model v1.0 emb32, diverging for emb128
random_seed = 182542180
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
pl.seed_everything(random_seed, workers=True)

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
osmnx_dataset_train, osmnx_dataset_val, osmnx_dataset_test = random_split(osmnx_dataset, [0.7, 0.1, 0.2])
osmnx_loader_train = DataLoader(osmnx_dataset_train, batch_size=32, shuffle=True, num_workers=11)
osmnx_dataset_val = DataLoader(osmnx_dataset_val, batch_size=32, num_workers=11)
osmnx_loader_test = DataLoader(osmnx_dataset_test, batch_size=32, num_workers=11)

# Define the model
gaemodel_in_channels   = 1
gaemodel_edge_dim      = 1
gaemodel_out_channels  = 128
gaemodel_name          = f"gnnuf_model_v1_0_emb{gaemodel_out_channels}"
gaemodel = GAEModel(
    in_channels=gaemodel_in_channels,
    edge_dim=gaemodel_edge_dim,
    out_channels=gaemodel_out_channels
)
print(gaemodel)

# Loggers
logger_folder = bulk_storage_directory + "/lightning_logs"
logger_tb = TensorBoardLogger(logger_folder, name=gaemodel_name)
logger_csv = CSVLogger(logger_folder, name=gaemodel_name)

# Trainer
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=os.path.join(this_repo_directory, "models"),
    filename=gaemodel_name + '_{epoch:02d}',
    save_top_k=1,
    mode='min',
)
lr_monitor = LearningRateMonitor(
    logging_interval='epoch'
)
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=100
)

trainer = pl.Trainer(
    devices=1, 
    accelerator='gpu',
    logger=[logger_tb, logger_csv], 
    max_epochs=1000,
    callbacks=[
        checkpoint_callback, 
        lr_monitor,
        early_stop_callback
    ]
)

tuner = Tuner(trainer)
lr_finder = tuner.lr_find(
    gaemodel,
    train_dataloaders=osmnx_loader_train, 
    val_dataloaders=osmnx_dataset_val
)
new_lr = lr_finder.suggestion()
print(f'{new_lr=}')
gaemodel.hparams.learning_rate = new_lr

# Train the model
trainer.fit(
    gaemodel, 
    train_dataloaders=osmnx_loader_train, 
    val_dataloaders=osmnx_dataset_val
)

# Test the model
trainer.test(
    gaemodel, 
    dataloaders=osmnx_loader_test
)

# Save best model as pt
best_model_path = checkpoint_callback.best_model_path
print(f'{best_model_path}')
best_model = GAEModel.load_from_checkpoint(
    best_model_path,
    in_channels=gaemodel_in_channels,
    edge_dim=gaemodel_edge_dim,
    out_channels=gaemodel_out_channels
)
torch.save(
    best_model.state_dict(), 
    os.path.join(this_repo_directory, f'models/{gaemodel_name}.pt')
)
