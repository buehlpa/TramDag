import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import logistic
from scipy.special import logit

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from torch.cuda.amp import autocast, GradScaler

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Train with GPU support.")
else:
    device = torch.device('cpu')
print("No GPU found, train with CPU support.")

import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt


# own utils
from utils.graph import *
from utils.tram_models import *
from utils.tram_model_helpers import *
from utils.tram_data import *
from utils.loss_continous import *
from utils.sampling_tram_data import *


experiment_name = "multiprocessing"   ## <--- set experiment name
seed=42
np.random.seed(seed)

DATA_PATH = "/home/bule/TramDag/data"
LOG_DIR="/home/bule/TramDag/dev_experiment_logs"
EXPERIMENT_DIR = os.path.join(LOG_DIR, experiment_name)
os.makedirs(EXPERIMENT_DIR,exist_ok=True)


from scipy.special import logit
from mpl_toolkits.mplot3d import Axes3D


# Define the functions used in the DGP
def f1(x1, x2):
    return np.sin(np.pi * x1) * np.cos(np.pi * x2)

def f2(x2, x3):
    return np.exp(-((x2 - 1)**2 + (x3 - 1)**2))

def f3(x4, x2):
    return (x4 * x2) / (1 + x4**2 + x2**2)

def dgp_continuous_interactions(n_obs=100, seed=42):
    np.random.seed(seed)

    # Independent variables
    x1 = np.random.uniform(0, 2, size=n_obs)
    x2 = np.random.uniform(0, 2, size=n_obs)
    x3 = np.random.uniform(0, 2, size=n_obs)
    x4 = np.random.uniform(0, 2, size=n_obs)
    x5 = np.random.normal(0, 1, size=n_obs)
    x6 = np.random.uniform(0, 2, size=n_obs)
    x7 = np.random.normal(0, 1, size=n_obs)

    # Response variable with interactions
    y = f1(x1, x2) + f2(x3, x4) + f3(x5, x6) + 1.5 * x7

    df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5, 'x6': x6, 'x7': x7, 'x8': y})
    return df

# Generate data
df = dgp_continuous_interactions()


# 1. Split the data
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# 2. Compute quantiles from training data
quantiles = train_df.quantile([0.05, 0.95])
min_vals = quantiles.loc[0.05]
max_vals = quantiles.loc[0.95]

# 3. Normalize all sets using training quantiles
def normalize_with_quantiles(df, min_vals, max_vals):
    return (df - min_vals) / (max_vals - min_vals)

# train_df = normalize_with_quantiles(train_df, min_vals, max_vals)
# val_df = normalize_with_quantiles(val_df, min_vals, max_vals)
# test_df = normalize_with_quantiles(test_df, min_vals, max_vals)

print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")


# --- Editable Parameters ---
variable_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']
data_type={'x1':'cont','x2':'cont','x3':'cont','x4':'cont','x5':'cont','x6':'cont','x7':'cont','x8':'cont'}  # continous , images , ordinal
adj_matrix = np.load(os.path.join(EXPERIMENT_DIR, "adj_matrix.npy"),allow_pickle=True)
nn_names_matrix= create_nn_model_names(adj_matrix,data_type)


conf_dict=get_configuration_dict(adj_matrix,nn_names_matrix, data_type)
# write min max to conf dict
for i,node in enumerate(data_type.keys()):
    conf_dict[node]['min']=min_vals[i].tolist()   # <---- TODO add quanitle marker
    conf_dict[node]['max']=max_vals[i].tolist()
    

# write to file
CONF_DICT_PATH = os.path.join(EXPERIMENT_DIR, f"{experiment_name}_conf.json")
with open(CONF_DICT_PATH, 'w') as f:
    json.dump(conf_dict, f, indent=4)
    
print(f"Configuration saved to {CONF_DICT_PATH}")


DEV_TRAINING=True
train_list=['x1','x2','x3','x4','x5','x6','x7','x8']#['x2']#'x1','x2']#,'x3']#['x1']#['x1','x2','x3']#,#,['x1','x2','x3'] # <-  set the nodes which have to be trained , useful if further training is required else lsit all vars

batch_size = 512#4112
epochs = 100# <- if you want a higher numbe rof epochs, set the number higher and it loads the old model and starts from there
learning_rate=0.1
use_scheduler =  False

import torch.multiprocessing as mp


# Map each node to a GPU, round-robin
def assign_gpus(nodes, num_gpus):
    mapping = {}
    for i, node in enumerate(nodes):
        mapping[node] = i % num_gpus
    return mapping

def train_one_node(node, gpu_id):
    # Force this process to see only the assigned GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)  # inside the process, GPU 0 is the one we've exposed

    # === Replicate your existing per-node code ===
    if node not in train_list:
        print(f"Skipping node {node} (not in train_list).")
        return

    if conf_dict[node]['node_type'] in ('source', 'other'):
        print(f"Skipping node {node} of type {conf_dict[node]['node_type']}.")
        return

    # Paths
    NODE_DIR = os.path.join(EXPERIMENT_DIR, node)
    os.makedirs(NODE_DIR, exist_ok=True)
    MODEL_PATH, LAST_MODEL_PATH, TRAIN_HIST_PATH, VAL_HIST_PATH = model_train_val_paths(NODE_DIR)

    # Build model
    tram_model = get_fully_specified_tram_model(node, conf_dict, verbose=True).to('cuda')
    _, ordered_terms_in_h, _ = ordered_parents(node, conf_dict)

    # Data
    train_loader, val_loader = get_dataloader(
        node, conf_dict, train_df, val_df, batch_size=batch_size, verbose=True
    )

    # Load history if exists
    if os.path.exists(MODEL_PATH) and os.path.exists(TRAIN_HIST_PATH) and os.path.exists(VAL_HIST_PATH):
        print(f"[{node}] Loading existing model & history …")
        tram_model.load_state_dict(torch.load(MODEL_PATH))
        with open(TRAIN_HIST_PATH) as f:
            train_loss_hist = json.load(f)
        with open(VAL_HIST_PATH) as f:
            val_loss_hist = json.load(f)
        start_epoch = len(train_loss_hist)
        best_val_loss = min(val_loss_hist)
    else:
        print(f"[{node}] No checkpoint found. Starting fresh …")
        train_loss_hist, val_loss_hist = [], []
        start_epoch = 0
        best_val_loss = float('inf')

    if start_epoch >= epochs:
        print(f"[{node}] Already trained for {epochs} epochs, skipping.")
        return

    # Optimizer & scheduler
    optimizer = torch.optim.Adam(tram_model.parameters(), lr=learning_rate)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        if use_scheduler else None
    )

    # Min/max tensors
    min_vals = torch.tensor(conf_dict[node]['min'], dtype=torch.float32).to('cuda')
    max_vals = torch.tensor(conf_dict[node]['max'], dtype=torch.float32).to('cuda')
    min_max = torch.stack([min_vals, max_vals], dim=0)

    # Run training
    train_val_loop(
        start_epoch,
        epochs,
        tram_model,
        train_loader,
        val_loader,
        train_loss_hist,
        val_loss_hist,
        best_val_loss,
        'cuda',
        optimizer,
        use_scheduler,
        scheduler,
        min_max,
        NODE_DIR,
        ordered_terms_in_h,
        save_linear_shifts=True
    )

def main():
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs found – can’t parallelize!")

    # Only nodes you actually want to train
    nodes_to_train = [n for n in conf_dict if n in train_list
                      and conf_dict[n]['node_type'] not in ('source','other')]

    gpu_map = assign_gpus(nodes_to_train, num_gpus)

    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)
    processes = []
    for node, gpu_id in gpu_map.items():
        p = mp.Process(target=train_one_node, args=(node, gpu_id))
        p.start()
        processes.append(p)

    # Wait
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()