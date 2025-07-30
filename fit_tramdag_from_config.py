import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from sklearn.model_selection import train_test_split
import torch
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt

# Own utils
from utils.graph import *
from utils.tram_models import *
from utils.tram_model_helpers import *
from utils.tram_data import *
from utils.loss_continous import *
from utils.sampling_tram_data import *

experiment_name = "tramdag_paper_exp_6_2_mixed"  # <-- Set experiment name
seed = 42
np.random.seed(seed)

LOG_DIR = "/home/bule/TramDag/dev_experiment_logs"
EXPERIMENT_DIR = os.path.join(LOG_DIR, experiment_name)
DATA_PATH = EXPERIMENT_DIR
CONF_DICT_PATH = os.path.join(EXPERIMENT_DIR, f"configuration.json")
EXP_DATA_PATH = os.path.join(DATA_PATH, f"{experiment_name}.csv")

try:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Train with GPU support.")
    else:
        device = torch.device('cpu')
        print("No GPU found, train with CPU support.")
except Exception as e:
    print("Error setting up device:", e)
    raise

################# DATA LOADING #################
try:
    print("Loading data...")
    df = pd.read_csv(EXP_DATA_PATH)
    print(f"Data shape: {df.shape}")
except Exception as e:
    print("Failed to load CSV data:", e)
    raise

try:
    print("Splitting data into train/val/test...")
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
except Exception as e:
    print("Error during data splitting:", e)
    raise

try:
    print("Computing quantiles...")
    quantiles = train_df.quantile([0.05, 0.95])
    min_vals = quantiles.loc[0.05]
    max_vals = quantiles.loc[0.95]
except Exception as e:
    print("Error computing quantiles:", e)
    raise

################# TRAINING PARAMS #################
train_list = ['x1', 'x2', 'x3']
batch_size = 512
epochs = 2000
learning_rate = 0.01
use_scheduler = False

############### LOAD CONFIGURATION ###############
try:
    print("Loading configuration...")
    configuration_dict = load_configuration_dict(CONF_DICT_PATH)
    target_nodes = configuration_dict['nodes']
except Exception as e:
    print("Error loading configuration:", e)
    raise

############### TRAINING LOOP ####################
for node in target_nodes:
    try:
        print(f'\n--- Node: {node} ---')

        if node not in train_list:
            print(f"Skipping node {node} (not in train list)")
            continue

        if (target_nodes[node]['node_type'] == 'source') and (target_nodes[node]['data_type'] == 'other'):# Skip unsupported types
            print(f"Node type : other , is not supported yet")
            continue

        ###################### SETUP PATHS ######################
        NODE_DIR = os.path.join(EXPERIMENT_DIR, f'{node}')
        os.makedirs(NODE_DIR, exist_ok=True)

        if not check_if_training_complete(node, NODE_DIR, epochs):
            print(f"Training already complete for node {node}. Skipping...")
            continue

        ###################### CREATE MODEL #####################
        print("Creating model...")
        tram_model = get_fully_specified_tram_model_v5(node, target_nodes, verbose=True)
        tram_model.to(device)

        ###################### DATALOADERS ######################
        print("Setting up dataloaders...")
        train_loader, val_loader = get_dataloader_v5(
            node, target_nodes, train_df, val_df, batch_size=batch_size,
            return_intercept_shift=True, verbose=False
        )

        ###################### OPTIMIZER ########################
        print("Initializing optimizer...")
        optimizer = torch.optim.Adam(tram_model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2) if use_scheduler else None

        ###################### TRAINING #########################
        print("Starting training loop...")
        train_val_loop_v6(
            node,
            target_nodes,
            NODE_DIR,
            tram_model,
            train_loader,
            val_loader,
            epochs,
            optimizer,
            use_scheduler,
            scheduler,
            save_linear_shifts=False,
            verbose=1,
            device=device
        )

    except Exception as e:
        print(f"Training failed for node {node}: {e}")
        import traceback
        traceback.print_exc()
