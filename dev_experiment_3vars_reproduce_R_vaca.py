import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import logistic

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
from utils.continous import *
from utils.sampling_tram_data import *

# 1. Experiments and Paths

experiment_name = "reproduce_vaca_CI"   ## <--- set experiment name
seed=42
np.random.seed(seed)

DATA_PATH = "/home/bule/TramDag/data"
LOG_DIR="/home/bule/TramDag/dev_experiment_logs"
EXPERIMENT_DIR = os.path.join(LOG_DIR, experiment_name)
os.makedirs(EXPERIMENT_DIR,exist_ok=True)


# # # 2.  Data

# - same experiment as in the R CODE 
# - provide data in the form of a pandas dataframe, if there are iamges add the paths to each image to the df


def generate_data_dgp_vaca(n_samples: int = 10000):
    # Generate X1 using a bimodal distribution
    X1 = np.where(
        np.random.rand(n_samples) < 0.5,
        np.random.normal(-2, np.sqrt(1.5), n_samples),
        np.random.normal(1.5, 1, n_samples)
    )
    # Generate X2 using the relationship X2 = -X1 + N(0, 1)
    X2 = -X1 + np.random.normal(0, 1, n_samples)
    # Generate X3 using the relationship X3 = X1 + 0.25 * X2 + N(0, 1)
    X3 = X1 + 0.25 * X2 + np.random.normal(0, 1, n_samples)
    
    data = pd.DataFrame({
        "x1": X1,
        "x2": X2,
        "x3": X3
    })
    
    return data


# ----- RUN DGP -----
EXP_DATA_PATH=os.path.join(EXPERIMENT_DIR, f"{experiment_name}.csv")


if not os.path.exists(EXP_DATA_PATH):
    df = generate_data_dgp_vaca(n_samples=10_000)

    print(df.head())
    df.to_csv(EXP_DATA_PATH, index=False)

else:
    df = pd.read_csv(EXP_DATA_PATH)
    print(f"Loaded data from {EXP_DATA_PATH}")
    


### Standardize
# scaler = StandardScaler()
# df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Get min_vals and max_vals as torch tensors
quantiles = df.quantile([0.025, 0.975])
min_vals = quantiles.loc[0.025].values.astype(np.float32)
max_vals = quantiles.loc[0.975].values.astype(np.float32)

# sns.pairplot(df)
# plt.suptitle("", y=1.02)
# plt.tight_layout()
# plt.show()


## 2.1 train test split


# train
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
# Validation and test
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")


# describe if data is continous or  ordinal  ['cont', 'ord','other']
# note that other data than tabular can only be used as source 

# TODO support for interactions in CI model eg CI_1

# Example 1 dgp tramdag paper  3x3: 
data_type={'x1':'cont','x2':'cont','x3':'cont'}  # continous , images , ordinal
adj_matrix = np.array([
    [ "0", "ci", "ci"],  # A -> B (cs), A -> C (ls)
    [ "0", "0", "cs"],  # B -> D (ls)
    [ "0", "0", "0"],  # C -> D (cs)
], object)


# its also possible to have ci11 and ci12 etc to inlcude multiple varibales for a network for the intercept. aswell as for cs name it with ci11 ci12

plot_seed=5
# plot_dag(adj_matrix,data_type, seed=plot_seed)


# 4. Configuration for the Models

# - all SI and LS model are generated outmatically since these are shallow NN's
# - CI and CS have to be defined by the User and can be Passed for each model, -> generate default networks which are generated automaitcally


# check if there are Ci or Compelx shifts in the models. If yes define the modelnames
nn_names_matrix= create_nn_model_names(adj_matrix,data_type)
# plot_nn_names_matrix(nn_names_matrix,data_type)

# TODO: fucniton to automate
# if different models should be used, defin model in utils.models 
# e.g ComplexInterceptCUSTOMImage # any possible eg VITS

# # rename the modelnames in the nn_names_matrix
# nn_names_matrix[0,2]='ComplexInterceptCustomTabular'  
# nn_names_matrix[1,2]='ComplexShiftCustomTabular'  
# plot_nn_names_matrix(nn_names_matrix,data_type)

#TODO : OPTION write config to a argparser to  and args object to pass datatypes

conf_dict=get_configuration_dict(adj_matrix,nn_names_matrix, data_type)
# write min max to conf dict
for i,node in enumerate(data_type.keys()):
    conf_dict[node]['min']=min_vals[i].tolist()
    conf_dict[node]['max']=max_vals[i].tolist()
    

# write to file
CONF_DICT_PATH = os.path.join(EXPERIMENT_DIR, f"{experiment_name}_conf.json")
with open(CONF_DICT_PATH, 'w') as f:
    json.dump(conf_dict, f, indent=4)
    
print(f"Configuration saved to {CONF_DICT_PATH}")

# conf_dict

# # 5. Fit models

# - each model independently fitting


DEV_TRAINING=True
train_list=['x1','x2','x3']#['x1']#['x1','x2','x3']#,#,['x1','x2','x3'] # <-  set the nodes which have to be trained , useful if further training is required else lsit all vars

batch_size = 4112
epochs = 4500   # <- if you want a higher numbe rof epochs, set the number higher and it loads the old model and starts from there
use_scheduler = True


# For each NODE 
for node in conf_dict:
    
    print(f'\n----*----------*-------------*--------------- Node: {node} ------------*-----------------*-------------------*--')
    
    ########################## 0. Skip nodes ###############################
    if node not in train_list:# Skip if node is not in train_list
        print(f"Skipping node {node} as it's not in the training list.")
        continue
    if (conf_dict[node]['node_type'] == 'source') and (conf_dict[node]['node_type'] == 'other'):# Skip unsupported types
        print(f"Node type : other , is not supported yet")
        continue

    ########################## 1. Setup Paths ###############################
    NODE_DIR = os.path.join(EXPERIMENT_DIR, f'{node}')
    os.makedirs(NODE_DIR, exist_ok=True)
    MODEL_PATH, LAST_MODEL_PATH, TRAIN_HIST_PATH, VAL_HIST_PATH = model_train_val_paths(NODE_DIR)

    ########################## 2. Create Model ##############################
    tram_model = get_fully_specified_tram_model(node, conf_dict, verbose=True).to(device)

    ########################## 3. Create Dataloaders ########################
    train_loader, val_loader = get_dataloader(node, conf_dict, train_df, val_df, batch_size=batch_size, verbose=True)

    ########################## 4. Load Model & History ######################
    if os.path.exists(MODEL_PATH) and os.path.exists(TRAIN_HIST_PATH) and os.path.exists(VAL_HIST_PATH):
        print("Existing model found. Loading weights and history...")
        tram_model.load_state_dict(torch.load(MODEL_PATH))

        with open(TRAIN_HIST_PATH, 'r') as f:
            train_loss_hist = json.load(f)
        with open(VAL_HIST_PATH, 'r') as f:
            val_loss_hist = json.load(f)

        start_epoch = len(train_loss_hist)
        best_val_loss = min(val_loss_hist)
        print(f"Continuing training from epoch {start_epoch}...")
    else:
        print("No existing model found. Starting fresh...")
        train_loss_hist, val_loss_hist = [], []
        start_epoch = 0
        best_val_loss = float('inf')

    # Skip if already trained
    if start_epoch >= epochs:
        print(f"Node {node} already trained for {epochs} epochs. Skipping.")
        continue

    ########################## 5. Optimizer & Scheduler ######################
    optimizer = torch.optim.AdamW(tram_model.parameters(), lr=0.1, eps=1e-8, weight_decay=1e-2)
    
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    else:
        scheduler = None

    ########################## 6. Min/Max Tensor #############################
    min_vals = torch.tensor(conf_dict[node]['min'], dtype=torch.float32).to(device)
    max_vals = torch.tensor(conf_dict[node]['max'], dtype=torch.float32).to(device)
    min_max = torch.stack([min_vals, max_vals], dim=0)

    ########################## 7. Training Loop ##############################

    if DEV_TRAINING:
        train_val_loop(
            start_epoch,
            epochs,
            tram_model,
            train_loader,
            val_loader,
            train_loss_hist,
            val_loss_hist,
            best_val_loss,
            device,
            optimizer,
            use_scheduler,
            scheduler,
            min_max,
            NODE_DIR
        )
        
        
conf_dict
EXPERIMENT_DIR
train_df
val_df
device

verbose=False


batch_size = 4112
for node in conf_dict:
    print(f'\n----*----------*-------------*--------check CS of {node} ------------*-----------------*-------------------*--')
    if node != 'x3':
        print('not x3')
        continue
    #### 0. Paths
    NODE_DIR = os.path.join(EXPERIMENT_DIR, f'{node}')
    
    ##### 1. Load model 
    model_path = os.path.join(NODE_DIR, "best_model.pt")
    tram_model = get_fully_specified_tram_model(node, conf_dict, verbose=verbose)
    tram_model = tram_model.to(device)
    tram_model.load_state_dict(torch.load(model_path))
    tram_model.eval()

    ##### 2. Dataloader
    train_loader, val_loader = get_dataloader(node, conf_dict, train_df, val_df, batch_size=batch_size, verbose=verbose)
    
    #### 3. Forward Pass
    min_vals = torch.tensor(conf_dict[node]['min'], dtype=torch.float32).to(device)
    max_vals = torch.tensor(conf_dict[node]['max'], dtype=torch.float32).to(device)
    min_max = torch.stack([min_vals, max_vals], dim=0)

    h_train_list, h_val_list = [], []
    y_pred_list=[]
    with torch.no_grad():
        print("\nProcessing training data...")
        for x, y in tqdm(train_loader, desc=f"Train loader ({node})", total=len(train_loader)):
            y = y.to(device)
            int_input, shift_list = preprocess_inputs(x, device=device)
            
            print(shift_list)
            y_pred = tram_model(int_input=int_input, shift_input=shift_list)
            y_pred_list.append(y_pred)

            
            
            h_train, _ = contram_nll(y_pred, y, min_max=min_max, return_h=True)
            h_train_list.extend(h_train.cpu().numpy())

        # print("\nProcessing validation data...")
        # for x, y in tqdm(val_loader, desc=f"Val loader ({node})", total=len(val_loader)):
        #     y = y.to(device)
        #     int_input, shift_list = preprocess_inputs(x, device=device)
        #     y_pred = tram_model(int_input=int_input, shift_input=shift_list)
        #     h_val, _ = contram_nll(y_pred, y, min_max=min_max, return_h=True)
        #     h_val_list.extend(h_val.cpu().numpy())

# y_pred_list


# 7. Sample from Graph

n = 10_000  # Desired number of latent samples 
batch_size = 1
verbose=True
delete_all_previously_sampled=True


def sample_full_dag_chandru(conf_dict,
                            EXPERIMENT_DIR,
                            device,
                            n= 10_000,
                            batch_size = 32,
                            delete_all_previously_sampled=True,
                            verbose=True):
    

    if delete_all_previously_sampled:
        delete_all_samplings(conf_dict, EXPERIMENT_DIR)
        
    for node in conf_dict:
        print(f'\n----*----------*-------------*--------Sample Node: {node} ------------*-----------------*-------------------*--')
        NODE_DIR = os.path.join(EXPERIMENT_DIR, f'{node}')
        SAMPLING_DIR = os.path.join(NODE_DIR, 'sampling')
        os.makedirs(SAMPLING_DIR, exist_ok=True)
        
        if check_roots_and_latents(NODE_DIR, rootfinder='chandrupatla', verbose=verbose):
            continue
        
        skipping_node = False
        if conf_dict[node]['node_type'] != 'source':
            for parent in conf_dict[node]['parents']:
                if not check_roots_and_latents(os.path.join(EXPERIMENT_DIR, parent), rootfinder='chandrupatla', verbose=verbose):
                    skipping_node = True
                    break
                
        if skipping_node:
            print(f"Skipping {node} as parent {parent} is not sampled yet.")
            continue
        
        min_vals = torch.tensor(conf_dict[node]['min'], dtype=torch.float32).to(device)
        max_vals = torch.tensor(conf_dict[node]['max'], dtype=torch.float32).to(device)
        min_max = torch.stack([min_vals, max_vals], dim=0)
        
        
        latent_sample = torch.tensor(logistic.rvs(size=n), dtype=torch.float32).to(device)
        #latent_sample = truncated_logistic_sample(n=n, low=0, high=1, device=device)
        
        if verbose:
            print("-- sampled latents")
            
        model_path = os.path.join(NODE_DIR, "best_model.pt")
        tram_model = get_fully_specified_tram_model(node, conf_dict, verbose=verbose).to(device)
        tram_model.load_state_dict(torch.load(model_path))
        

            
        if verbose:
            print("-- loaded modelweights")
            
        dataset = SamplingDataset(node=node, EXPERIMENT_DIR=EXPERIMENT_DIR, rootfinder='chandrupatla', number_of_samples=n, conf_dict=conf_dict, transform=None)
        sample_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        output_list = []
        with torch.no_grad():
            for x in tqdm(sample_loader, desc=f"h() for samples in  {node}"):
                x = [xi.to(device) for xi in x]
                int_input, shift_list = preprocess_inputs(x, device=device)
                model_outputs = tram_model(int_input=int_input, shift_input=shift_list)
                output_list.append(model_outputs)
                
        if conf_dict[node]['node_type'] == 'source':
            if verbose:
                print("source node, Defaults to SI and 1 as inputs")
            theta_single = output_list[0]['int_out'][0]
            theta_single = transform_intercepts_continous(theta_single)
            thetas_expanded = theta_single.repeat(n, 1)
            shifts = torch.zeros(n, device=device)
        else:
            if verbose:
                print("node has parents, previously sampled data is loaded for each pa(node)")
            y_pred = merge_outputs(output_list, skip_nan=True)
            shifts = y_pred['shift_out']
            if shifts is None:
                print("shift_out was None; defaulting to zeros.")
                shifts = torch.zeros(n, device=device)
            thetas = y_pred['int_out']
            thetas_expanded = transform_intercepts_continous(thetas).squeeze()
            shifts = shifts.squeeze()
            
        low = torch.full((n,), -1e5, device=device)
        high = torch.full((n,), 1e5, device=device)
        
        ## Root finder using Chandrupatla's method
        def f_vectorized(targets):
            return vectorized_object_function(
                thetas_expanded,
                targets,
                shifts,
                latent_sample,
                k_min=min_max[0],
                k_max=min_max[1]
            )
            
        root = chandrupatla_root_finder(
            f_vectorized,
            low,
            high,
            max_iter=10_000,
            tol=1e-9
        )
        
        ## Saving
        root_path = os.path.join(SAMPLING_DIR, "roots_chandrupatla.pt")
        latents_path = os.path.join(SAMPLING_DIR, "latents.pt")
        
        if torch.isnan(root).any():
            print(f'Caution! Sampling for {node} consists of NaNs')
            
        torch.save(root, root_path)
        torch.save(latent_sample, latents_path)
        
        
sample_full_dag_chandru(conf_dict,
                            EXPERIMENT_DIR,
                            device,
                            n= 10_000,
                            batch_size = 1,
                            delete_all_previously_sampled=False,
                            verbose=True)        