"""
Copyright 2025 Zurich University of Applied Sciences (ZHAW)
Pascal Buehler, Beate Sick, Oliver Duerr

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
from torch.utils.data import  DataLoader
import numpy as np
import warnings
import os
import shutil
from statsmodels.graphics.gofplots import qqplot_2samples
from scipy.stats import logistic, probplot
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from .model_helpers import   get_fully_specified_tram_model     
from .continous import   transform_intercepts_continous,h_extrapolated, vectorized_object_function ,chandrupatla_root_finder,bisection_root_finder
from .data import  GenericDataset, get_dataloader
from .ordinal import transform_intercepts_ordinal
from .continous import contram_nll

# This file contains helper functions for sampling full dag and to help with  tram data


#


def merge_outputs(dict_list, skip_nan=True):
    import warnings

    int_outs = []
    shift_outs = []
    skipped_count = 0

    for d in dict_list:
        int_tensor = d['int_out']

        # Combine shift outputs correctly
        if isinstance(d['shift_out'], list):
            # Option 1: additive contribution from multiple components
            shift_tensor = torch.stack(d['shift_out'], dim=0).sum(dim=0)
            # Option 2 (alternative): concatenate if each shift corresponds to a theta dimension
            # shift_tensor = torch.cat(d['shift_out'], dim=1)
        else:
            shift_tensor = d['shift_out']

        # Optionally skip entries with all NaNs in int_out
        if skip_nan and torch.isnan(int_tensor).all():
            skipped_count += 1
            continue

        int_outs.append(int_tensor)
        if shift_tensor is not None:
            shift_outs.append(shift_tensor)

    if skipped_count > 0:
        warnings.warn(f"{skipped_count} entries with all-NaN 'int_out' were skipped.")

    merged = {
        'int_out': torch.cat(int_outs, dim=0) if int_outs else None,
        'shift_out': torch.cat(shift_outs, dim=0) if shift_outs else None
    }

    return merged

def is_outcome_modelled_continous(node,target_nodes_dict):
    if 'yc'in target_nodes_dict[node]['data_type'].lower() or 'continous' in target_nodes_dict[node]['data_type'].lower():
        return True
    else:
        return False

def is_outcome_modelled_ordinal(node,target_nodes_dict):
    if 'yo'in target_nodes_dict[node]['data_type'].lower() and 'ordinal' in target_nodes_dict[node]['data_type'].lower():
        return True
    else:
        return False  


################################### Helpers for Sampling ###################################


def delete_all_samplings(conf_dict,EXPERIMENT_DIR):
    for node in conf_dict:
        NODE_DIR = os.path.join(EXPERIMENT_DIR, f'{node}')
        SAMPLING_DIR = os.path.join(NODE_DIR, 'sampling')
        # Delete the 'sampling' folder and its contents if it exists
        if os.path.exists(SAMPLING_DIR):
            shutil.rmtree(SAMPLING_DIR)
            print(f'Deleted directory: {SAMPLING_DIR}')
        else:
            print(f'Directory does not exist: {SAMPLING_DIR}')

def check_sampled_and_latents(NODE_DIR, debug=True):
    sampling_dir = os.path.join(NODE_DIR, 'sampling')
    root_path = os.path.join(sampling_dir, 'sampled.pt')
    latents_path = os.path.join(sampling_dir, 'latents.pt')

    if not os.path.exists(root_path):
        raise FileNotFoundError(f"'sampled.pt' not found in {sampling_dir}")
    if not os.path.exists(latents_path):
        raise FileNotFoundError(f"'latents.pt' not found in {sampling_dir}")

    if debug:
        print(f"[DEBUG] check_sampled_and_latents: Found 'sampled.pt' in {sampling_dir}")
        print(f"[DEBUG] check_sampled_and_latents: Found 'latents.pt' in {sampling_dir}")

    return True


################################### LATENTS U and U_low / U_upper ###################################

def provide_latents_for_continous(
    node,
    configuration_dict,
    EXPERIMENT_DIR,
    data_loader,
    base_df,
    verbose=False,
    min_max=None
):
    """
    Compute latent representations for each observation in base_df
    and return a DataFrame with columns [node, "_U"] aligned with base_df.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_nodes = configuration_dict["nodes"]
    if is_outcome_modelled_ordinal(node, target_nodes):
        raise ValueError("Not yet defined for ordinal target variables")

    #### 0. Paths
    NODE_DIR = os.path.join(EXPERIMENT_DIR, f"{node}")

    ##### 1. Load model 
    model_path = os.path.join(NODE_DIR, "best_model.pt")
    if not os.path.exists(model_path):
        print("[Warning] best_model.pt not found, falling back to initial_model.pt")
        model_path = os.path.join(NODE_DIR, "initial_model.pt")
    
    
    tram_model = get_fully_specified_tram_model(
        node, configuration_dict, debug=verbose, set_initial_weights=False
    )
    tram_model = tram_model.to(device)
    tram_model.load_state_dict(torch.load(model_path, map_location=device))
    tram_model.eval()

    #### 2. Forward Pass
    if min_max is None:
        min_vals = torch.tensor(target_nodes[node]['min'], dtype=torch.float32, device=device)
        max_vals = torch.tensor(target_nodes[node]['max'], dtype=torch.float32, device=device)
        min_max = torch.stack([min_vals, max_vals], dim=0)

    latents_list = []
    with torch.no_grad():
        for (int_input, shift_list), y in data_loader:
            # Move everything to device
            int_input = int_input.to(device)
            shift_list = [s.to(device) for s in shift_list]
            y = y.to(device)  # targets to device
            y_pred = tram_model(int_input=int_input, shift_input=shift_list)

            latents, _ = contram_nll(y_pred, y, min_max=min_max, return_h=True)
            latents_list.extend(latents.cpu().numpy())

    # Turn into DataFrame aligned with base_df
    latents_df = pd.DataFrame({
                node: base_df[node].values,
                f"{node}_U": latents_list
            }, index=base_df.index)

    return latents_df

def provide_latents_for_ordinal(
    node,
    configuration_dict,
    EXPERIMENT_DIR,
    data_loader,
    base_df,
    verbose=False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_nodes = configuration_dict["nodes"]
    if not is_outcome_modelled_ordinal(node, target_nodes):
        raise ValueError("Only for ordinal targets")

    NODE_DIR = os.path.join(EXPERIMENT_DIR, f"{node}")
    model_path = os.path.join(NODE_DIR, "best_model.pt")
    if not os.path.exists(model_path):
        print("[Warning] best_model.pt not found, using initial_model.pt")
        model_path = os.path.join(NODE_DIR, "initial_model.pt")

    tram_model = get_fully_specified_tram_model(
        node, configuration_dict, debug=verbose, set_initial_weights=False
    ).to(device)
    tram_model.load_state_dict(torch.load(model_path, map_location=device))
    tram_model.eval()

    # collect all cutpoints
    all_cutpoints = []
    with torch.no_grad():
        for (int_input, shift_list), _ in data_loader:
            int_input = int_input.to(device)
            shift_list = [s.to(device) for s in shift_list]
            out = tram_model(int_input=int_input, shift_input=shift_list)
            int_in = out["int_out"]
            shift_in = out["shift_out"]
            int_trans = transform_intercepts_ordinal(int_in)
            if shift_in is not None:
                shift = torch.stack(shift_in, dim=1).sum(dim=1).view(-1)
                int_trans = int_trans - shift.unsqueeze(1)
            all_cutpoints.append(int_trans.cpu().numpy())

    # concatenate to single array [N, K+1]
    all_cutpoints = np.concatenate(all_cutpoints, axis=0)

    # observed categories (assumed 0-based)
    k_obs = base_df[node].to_numpy().astype(int)

    # lower = θ_k, upper = θ_{k+1}
    u_low = all_cutpoints[np.arange(len(k_obs)), k_obs]
    u_upper = all_cutpoints[np.arange(len(k_obs)), k_obs + 1]

    latents_df = pd.DataFrame({
        node: base_df[node].values,
        f"{node}_U_lower": u_low,
        f"{node}_U_upper": u_upper
    }, index=base_df.index)
    return latents_df

def create_latent_df_for_full_dag(configuration_dict, EXPERIMENT_DIR, df, verbose=False, min_max_dict=None):

    all_latents_dfs = []

    for node in configuration_dict["nodes"]:
        
        
        if verbose:
                print(f"[INFO] Processing node '{node}'")
        
            # Build dataset and dataloader for this node

        node_dataset = GenericDataset(
            df,
            target_col=node,
            target_nodes=configuration_dict["nodes"],
            skip_checks=True
        )
        node_loader = DataLoader(
            node_dataset,
            batch_size=4096,   # you can tune this
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
            
        # Skip ordinal outcomes if not supported
        if is_outcome_modelled_ordinal(node, configuration_dict["nodes"]):
            
            node_latents_df =provide_latents_for_ordinal(
                node,
                configuration_dict,
                EXPERIMENT_DIR,
                data_loader=node_loader,
                base_df=df,
                verbose=False
            )
            all_latents_dfs.append(node_latents_df)
            
        
        elif is_outcome_modelled_continous(node, configuration_dict["nodes"]):
            
            if min_max_dict is not None and node in min_max_dict:
                min_vals = torch.tensor(min_max_dict[node][0], dtype=torch.float32)
                max_vals = torch.tensor(min_max_dict[node][1], dtype=torch.float32)
                min_max = torch.stack([min_vals, max_vals], dim=0)
            else:
                min_max=None

            # Compute latents
            node_latents_df = provide_latents_for_continous(
                                                            node=node,
                                                            configuration_dict=configuration_dict,
                                                            EXPERIMENT_DIR=EXPERIMENT_DIR,
                                                            data_loader=node_loader,
                                                            base_df=df,
                                                            verbose=verbose,
                                                            min_max=min_max
                                                            )

            all_latents_dfs.append(node_latents_df)

    # Concatenate all node-specific latent dataframes
    all_latents_df = pd.concat(all_latents_dfs, axis=1)

    # Remove duplicate index columns (if multiple nodes included their own target col)
    all_latents_df = all_latents_df.loc[:, ~all_latents_df.columns.duplicated()]

    print("[INFO] Final latent DataFrame shape:", all_latents_df.shape)
    return all_latents_df
    
################################### MAIN SAMPLING FUNCTIONS ###################################
def sample_full_dag(configuration_dict,
                    EXPERIMENT_DIR,
                    device,
                    do_interventions={},
                    predefined_latent_samples_df: pd.DataFrame = None,
                    number_of_samples: int = 10_000,
                    number_of_counterfactual_samples: int=1_000,
                    batch_size: int = 256,
                    delete_all_previously_sampled: bool = True,
                    verbose: bool = True,
                    debug: bool = False,
                    minmax_dict=None,
                    use_initial_weights_for_sampling: bool = False):
    """
    Sample values for all nodes in a DAG given trained TRAM models, respecting
    parental ordering. Supports both generative sampling (new U's) and
    reconstruction sampling (predefined U's), as well as node-level interventions.

    Parameters
    ----------
    configuration_dict : dict
        Full experiment configuration. Must contain a "nodes" entry where each node
        has metadata including:
            - 'node_type': str, either 'source' or other
            - 'parents': list of parent node names
            - 'min': float, minimum allowed value for the node
            - 'max': float, maximum allowed value for the node
            - 'data_type': str, e.g. "continuous" or "ordinal"
    EXPERIMENT_DIR : str
        Base directory where per-node models and sampling results are stored.
    device : torch.device
        Device for model evaluation (e.g., 'cuda' or 'cpu').
    do_interventions : dict, optional
        Mapping of node names to fixed values. For intervened nodes, the model is
        bypassed and all samples are set to the specified value. Example:
        {'x1': 1.0}.
    predefined_latent_samples_df : pd.DataFrame, optional
        DataFrame of predefined latent variables (U's) for reconstruction. Must
        contain one column per node in the form "{node}_U". If provided, the number
        of samples is set to the number of rows in this DataFrame.
    number_of_samples : int, default=10_000
        Number of samples to draw per node when no predefined latent samples are given.
    batch_size : int, default=32
        Batch size for DataLoader evaluation during sampling.
    delete_all_previously_sampled : bool, default=True
        If True, deletes all existing sampled.pt/latents.pt files before starting.
    verbose : bool, default=True
        If True, print progress information.
    debug : bool, default=False
        If True, print detailed debug information for troubleshooting.

    Returns
    -------
    sampled_by_node : dict
        Mapping from node name to a tensor of sampled values (on CPU).
    latents_by_node : dict
        Mapping from node name to the latent variables (U's) used to generate
        those samples (on CPU).

    Notes
    -----
    - The function respects DAG ordering by ensuring parents are sampled before
      their children.
    - In generative mode (no predefined_latent_samples_df), latent U's are sampled
      from a standard logistic distribution and parents are taken from sampled.pt.
    - In reconstruction mode (with predefined_latent_samples_df), latent U's are
      read from the DataFrame, but parent values are still loaded from sampled.pt
      unless explicitly overridden upstream.
    - Models are loaded from "best_model.pt" in each node's directory and applied
      in eval mode with no gradient tracking.
    - Continuous outcomes are sampled via vectorized root finding
      (Chandrupatla's algorithm), while ordinal outcomes use categorical sampling.
    """
    if verbose or debug:
        print(f"[INFO] Starting full DAG sampling with {number_of_samples} samples per node.")
        if do_interventions:
            print(f"[INFO] Interventions specified for nodes: {list(do_interventions.keys())}")
            
    if debug:
        print('[DEBUG] sample_full_dag: device:', device)
        
        
    if predefined_latent_samples_df is not None:
        number_of_samples = len(predefined_latent_samples_df)
        if verbose or debug:
            print(f'[INFO] Using predefined latents samples from dataframe -> therefore n_samples is set to the number of rows in the dataframe: {len(predefined_latent_samples_df)}')
    
    
    
    target_nodes_dict=configuration_dict["nodes"]

    # Collect results for direct use in notebooks
    sampled_by_node = {}
    latents_by_node = {}


    if delete_all_previously_sampled:
        if verbose or debug:
            print("[INFO] Deleting all previously sampled data.")
        delete_all_samplings(target_nodes_dict, EXPERIMENT_DIR)
    
    
    #### FOLLOWING ALONG THE CAUSAL ORDERING OF THE SPECIFICATION IN THE TARGET_NODES_DICT:
    for node in target_nodes_dict: # for each node in the target_nodes_dict
                    
        print(f'\n----*----------*-------------*--------Sample Node: {node} ------------*-----------------*-------------------*--') 
        
        ## 1. Paths collect samplings for each node in a subdirectory
        NODE_DIR = os.path.join(EXPERIMENT_DIR, f'{node}')
        SAMPLING_DIR = os.path.join(NODE_DIR, 'sampling')
        os.makedirs(SAMPLING_DIR, exist_ok=True)
        SAMPLED_PATH = os.path.join(SAMPLING_DIR, "sampled.pt")
        LATENTS_PATH = os.path.join(SAMPLING_DIR, "latents.pt")
        
        ## 2. Check if the parents are already sampled -> must be given due to the causal ordering
        for parent in target_nodes_dict[node]['parents']:
            parent_dir = os.path.join(EXPERIMENT_DIR, parent)
            try:
                check_sampled_and_latents(parent_dir, debug=debug)
            except FileNotFoundError:
                if verbose or debug:
                    print(f"[INFO] Skipping {node} as parent '{parent}' is not sampled yet.")

        
        ## INTERVENTION, if node is to be intervened on , data is just saved
        if do_interventions and node in do_interventions.keys():
                # For interventions make all the values the same for 
                intervention_value = do_interventions[node]
                if verbose or debug:
                    print(f"[INFO] Applying intervention for node '{node}' with value {intervention_value}")
                intervention_vals = torch.full((number_of_samples,), intervention_value)
                torch.save(intervention_vals, SAMPLED_PATH)
                
                ### dummy latents jsut for the check , not needed
                dummy_latents = torch.full((number_of_samples,), float('nan'))  
                torch.save(dummy_latents, LATENTS_PATH)
                
                # Store for immediate use
                sampled_by_node[node] = intervention_vals
                latents_by_node[node] = dummy_latents
                if verbose or debug:
                    print(f'[INFO] Interventional data for node {node} is saved')
                continue 

        ##### NO INTERVENTION, based on the sampled data from the parents the latents for each node the observational distribution is generated    
        else:

            #################################### load model and weigths 
            tram_model=load_tram_model_weights(node,configuration_dict, NODE_DIR,debug=False,verbose=False, device='cpu',  use_initial_weights_for_sampling=False)
            
            
            #### latens and laten intervals
            has_interval_latents = (
                predefined_latent_samples_df is not None
                and f"{node}_U_lower" in predefined_latent_samples_df.columns
                and f"{node}_U_upper" in predefined_latent_samples_df.columns
            )

            if (
                predefined_latent_samples_df is not None
                and f"{node}_U" in predefined_latent_samples_df.columns
                and not has_interval_latents
            ):
                predefinded_sample_name = f"{node}_U"
                predefinded_sample = predefined_latent_samples_df[predefinded_sample_name].values
                if verbose or debug:
                    print(f"[INFO] Using predefined latents samples for node {node} from dataframe column: {predefinded_sample_name}")
                latent_sample = torch.tensor(predefinded_sample, dtype=torch.float32).to(device)

            else:
                if verbose or debug:
                    if has_interval_latents:
                        print(f"[INFO] Detected '{node}_U_lower' and '{node}_U_upper' — switching to counterfactual logistic sampling mode.")
                        # latent_sample = torch.tensor(logistic.rvs(size=number_of_samples), dtype=torch.float32).to(device)
                    else:
                        print(f"[INFO] Sampling new latents for node {node} from standard logistic distribution")
                        
                        latent_sample = torch.tensor(logistic.rvs(size=number_of_samples), dtype=torch.float32).to(device)
            
            
            ###################################################### Counterfactual logic #########################################
            
            # Node has NO! parents with samples which are just distributions
            if not has_parents_with_distribution_samples(target_nodes_dict,node,debug=debug,EXPERIMENT_DIR=EXPERIMENT_DIR):
                if not has_interval_latents:
                    sampled=sample_node_detParents_detLatent(node=node,
                                                          device=device,
                                                          target_nodes_dict=target_nodes_dict,
                                                          number_of_samples=number_of_counterfactual_samples,
                                                          batch_size=batch_size,
                                                          tram_model=tram_model,
                                                          latent_sample=latent_sample,
                                                          debug=debug,
                                                          minmax_dict=minmax_dict,
                                                          EXPERIMENT_DIR=EXPERIMENT_DIR)
                if has_interval_latents:
                    sampled=sample_node_detParents_intervalLatent(node=node,
                                                            device=device,
                                                            target_nodes_dict=target_nodes_dict,
                                                            number_of_samples=number_of_counterfactual_samples,
                                                            batch_size=batch_size,
                                                            tram_model=tram_model,
                                                            predefined_latent_samples_df=predefined_latent_samples_df,
                                                            debug=debug,
                                                            EXPERIMENT_DIR=EXPERIMENT_DIR)
            
            # Node has parents with samples which are just distributions
            if has_parents_with_distribution_samples(target_nodes_dict,node,debug=debug,EXPERIMENT_DIR=EXPERIMENT_DIR):
                if has_interval_latents:
                    sampled=sample_node_distParents_intervalLatent(node=node,
                                                              device=device,
                                                              target_nodes_dict=target_nodes_dict,
                                                              number_of_samples=number_of_counterfactual_samples,
                                                              batch_size=batch_size,
                                                              tram_model=tram_model,
                                                              predefined_latent_samples_df=predefined_latent_samples_df,
                                                              debug=debug,
                                                              EXPERIMENT_DIR=EXPERIMENT_DIR)  
                if not has_interval_latents:
                    sampled=sample_node_distParents_detLatent(node=node,
                                                            device=device,
                                                            target_nodes_dict=target_nodes_dict,
                                                            number_of_samples=number_of_counterfactual_samples,
                                                            batch_size=batch_size,
                                                            tram_model=tram_model,
                                                            predefined_latent_samples_df=predefined_latent_samples_df,
                                                            latent_sample=latent_sample,
                                                            debug=debug,
                                                            minmax_dict=minmax_dict,
                                                            EXPERIMENT_DIR=EXPERIMENT_DIR)
                    
                
            ###*************************************************** Saving the latenst and sampled  ************************************************
            if torch.isnan(sampled).any():
                print(f"[WARNING] NaNs detected in sampled output for node '{node}'")
                
            torch.save(sampled, SAMPLED_PATH)
            torch.save(latent_sample, LATENTS_PATH)
            
            # Store CPU copies for immediate use
            sampled_by_node[node] = sampled.detach().cpu()
            latents_by_node[node] = latent_sample.detach().cpu()
            
            if verbose or debug:
                print(f"[INFO] Completed sampling for node '{node}'")
            
        
    if verbose or debug:
        print("[INFO] DAG sampling completed successfully for all nodes.")

    return sampled_by_node, latents_by_node

def sample_continous_modelled_target(node, target_nodes_dict, sample_loader, tram_model, latent_sample,device, debug=False,minmax_dict=None):
    number_of_samples = len(latent_sample)
    output_list = []
    tram_model.eval()
    with torch.no_grad():
        for (int_input, shift_list) in sample_loader:
            # Move everything to device
            int_input = int_input.to(device)
            shift_list = [s.to(device) for s in shift_list]
            model_outputs = tram_model(int_input=int_input, shift_input=shift_list)
            output_list.append(model_outputs)

    if not output_list:
        raise RuntimeError("sample_continous_modelled_target: Model output list is empty. Check the sample_loader or model.")

    if target_nodes_dict[node]['node_type'] == 'source':
        if debug:
            print("[DEBUG] sample_continous_modelled_target: source node, defaults to SI and 1 as inputs")
        if 'int_out' not in output_list[0]:
            raise KeyError("Missing 'int_out' in model output for source node.")
        
        theta_single = output_list[0]['int_out'][0]
        theta_single = transform_intercepts_continous(theta_single)
        
        thetas_expanded = theta_single.repeat(number_of_samples, 1)
        shifts = torch.zeros(number_of_samples, device=device)
        
    else:
        if debug:
            print("[DEBUG] sample_continous_modelled_target: node has parents, previously sampled data is loaded for each pa(node)")
        y_pred = merge_outputs(output_list, skip_nan=True)

        if 'int_out' not in y_pred:
            raise KeyError("Missing 'int_out' in merged model output.")
        if 'shift_out' not in y_pred:
            raise KeyError("Missing 'shift_out' in merged model output.")

        thetas = y_pred['int_out']
        
        shifts = y_pred['shift_out']
        
        if shifts is None:
            if debug:
                print("[DEBUG] sample_continous_modelled_target: shift_out was None; defaulting to zeros.")
            shifts = torch.zeros(number_of_samples, device=device)

        thetas_expanded = transform_intercepts_continous(thetas).squeeze()
        
        shifts = shifts.squeeze()

    # print("thetas_expanded ",thetas_expanded ) 
    if thetas_expanded.ndim == 1:
        thetas_expanded = thetas_expanded.unsqueeze(0)
    # Validate shapes
    if thetas_expanded.shape[0] != number_of_samples:
        raise ValueError(f"Mismatch in sample count: thetas_expanded has shape {thetas_expanded.shape}, expected {number_of_samples} rows.")

    if debug:
        print("[DEBUG] sample_continous_modelled_target: beginning root finding")
        print("[DEBUG] sample_continous_modelled_target: thetas_expanded shape:", thetas_expanded.shape)
        print("[DEBUG] sample_continous_modelled_target: shifts shape:", shifts.shape)
        print("[DEBUG] sample_continous_modelled_target: latent_sample shape:", latent_sample.shape)



    if minmax_dict is not None:
            min_vals = torch.tensor(minmax_dict[node][0], dtype=torch.float32, device=device)
            max_vals = torch.tensor(minmax_dict[node][1], dtype=torch.float32, device=device)
            min_max = torch.stack([min_vals, max_vals], dim=0)
            # minv, maxv = minmax_dict[node][0],minmax_dict[node][1]

    else:
        try:
            min_vals = torch.tensor(target_nodes_dict[node]['min'], dtype=torch.float32).to(device)
            max_vals = torch.tensor(target_nodes_dict[node]['max'], dtype=torch.float32).to(device)
            # minv, maxv = target_nodes_dict[node]['min'],target_nodes_dict[node]['max']
  
        except KeyError as e:
            raise KeyError(f"Missing 'min' or 'max' value in target_nodes_dict for node '{node}': {e}")
        min_max = torch.stack([min_vals, max_vals], dim=0)

    # # Root bounds
    low = torch.full((number_of_samples,), -1e5, device=device)
    high = torch.full((number_of_samples,), 1e5, device=device)
    # low = torch.full((number_of_samples,), float(minv - 2), device=device)
    # high = torch.full((number_of_samples,), float(maxv + 2), device=device)
    
    # Vectorized root-finding function # root finder has a f as input so we need to wrap it
    def f_vectorized(targets):
        return vectorized_object_function(
            thetas_expanded,
            targets,
            shifts,
            latent_sample,
            k_min=min_max[0],
            k_max=min_max[1]
        )

    # Root finding
    sampled = chandrupatla_root_finder(
        f_vectorized,
        low,
        high,
        max_iter=100,
        tol=1e-12
    )


    # # Root finding
    # sampled = bisection_root_finder(
    #     f_vectorized,
    #     low,
    #     high,
    #     max_iter=100,
    #     tol=1e-12
    # )
    
    
    if sampled is None or torch.isnan(sampled).any():
        raise RuntimeError("Root finding failed: returned None or contains NaNs.")

    if debug:
        print("[DEBUG] sample_continous_modelled_target: root finding complete. Sampled shape:", sampled.shape)

    return sampled

def sample_ordinal_modelled_target(sample_loader, tram_model, latent_sample, device, debug=False):
    tram_model.eval()
    model_outputs_list = []

    with torch.no_grad():
        for (int_input, shift_list) in sample_loader:
            int_input = int_input.to(device)
            shift_list = [s.to(device) for s in shift_list]
            outputs = tram_model(int_input=int_input, shift_input=shift_list)
            model_outputs_list.append(outputs)

    y_pred = merge_outputs(model_outputs_list, skip_nan=True)

    if 'int_out' not in y_pred:
        raise KeyError("Missing 'int_out' in merged model output.")
    if 'shift_out' not in y_pred:
        raise KeyError("Missing 'shift_out' in merged model output.")

    int_out = transform_intercepts_ordinal(y_pred['int_out'])  # [N, n_cut]
    shifts = y_pred['shift_out']

    if shifts is not None:
        shift_total = torch.stack(shifts, dim=1).sum(dim=1) if isinstance(shifts, list) else shifts
        h = int_out - shift_total
    else:
        h = int_out

    # h includes [-inf, thresholds..., inf]
    N = h.shape[0]

    if latent_sample.shape[0] != N:
        raise ValueError(f"latent_sample mismatch: got {latent_sample.shape[0]}, expected {N}")

    latent_sample = latent_sample.to(device).unsqueeze(1)

    # Classification based on latent space thresholds
    # Class k is selected if h[:, k] < latent <= h[:, k+1]
    categories = (latent_sample > h[:, :-1]) & (latent_sample <= h[:, 1:])
    samples = categories.float().argmax(dim=1)

    if debug:
        print("[DEBUG] sample_ordinal_modelled_target: h:", h[:5])
        print("[DEBUG] sample_ordinal_modelled_target: latent_sample:", latent_sample[:5])
        print("[DEBUG] sample_ordinal_modelled_target: samples:", samples[:5])

    return samples

################################### SAMPLE FULL DAG HELPERS ###################################
def create_df_from_sampled(node, target_nodes_dict, num_samples, EXPERIMENT_DIR, debug=False):
    sampling_dict = {}
    if debug:
        print("[DEBUG] create_df_from_sampled: initializing sampling dictionary with dummy variable")
        
    # Add dummy variable
    sampling_dict["DUMMY"] = torch.zeros(num_samples)

    # Try loading sampled values for each parent
    for parent in target_nodes_dict[node].get('parents', []):
        path = os.path.join(EXPERIMENT_DIR, parent, "sampling", "sampled.pt")
        if os.path.exists(path):
            if debug:
                print(f"[DEBUG] create_df_from_sampled: loading sampled data for parent '{parent}' from {path}")
            
            sampled=torch.load(path, map_location="cpu")
            if sampled.ndim > 1 and sampled.shape[1] > 1:
                if debug:
                    print(f"[DEBUG] create_df_from_sampled: skipping '{parent}' (shape {tuple(sampled.shape)}) - looks like probabilities")
                continue
            else:
                sampling_dict[parent] = sampled
                
        else:
            if debug:
                print(f"[DEBUG] create_df_from_sampled: no sampled data found for parent '{parent}' at {path}")

        # Skip if it’s a tuple of probabilities or multidimensional


    # Remove dummy if we have real variables
    if len(sampling_dict) > 1:
        if debug:
            print("[DEBUG] create_df_from_sampled: removing dummy variable since real variables are present")
        sampling_dict.pop("DUMMY")
    else:
        if debug:
            print("[DEBUG] create_df_from_sampled: only dummy variable present")

    # Move all tensors to CPU before creating the DataFrame
    sampling_dict_cpu = {k: v.cpu().numpy() for k, v in sampling_dict.items()}
    if debug:
        print("[DEBUG] create_df_from_sampled: creating DataFrame from variables:", list(sampling_dict_cpu.keys()))
        for k, v in sampling_dict_cpu.items():
            print(f"[DEBUG] create_df_from_sampled: {k} shape: {v.shape}")

    sampling_df = pd.DataFrame(sampling_dict_cpu)
    
    if debug:
        print("[DEBUG] create_df_from_sampled: final DataFrame shape:", sampling_df.shape)
    
    return sampling_df

def sample_node_detParents_detLatent(node,target_nodes_dict,number_of_samples,batch_size,tram_model,latent_sample,debug,minmax_dict,EXPERIMENT_DIR,device):
    
        # create dataframe from sampled parents + dummy if no parents 
        sampled_df=create_df_from_sampled(node, target_nodes_dict, number_of_samples, EXPERIMENT_DIR)
        
        sample_dataset = GenericDataset(sampled_df,target_col=node,
                                            target_nodes=target_nodes_dict,
                                            return_intercept_shift=True,
                                            return_y=False,
                                            debug=debug)
        
        sample_loader = DataLoader(sample_dataset, batch_size=batch_size, shuffle=False,num_workers=4, pin_memory=True)
        
        ###*************************************************** Continous Modelled Outcome ************************************************
        
        if is_outcome_modelled_continous(node,target_nodes_dict):
            sampled=sample_continous_modelled_target(node,target_nodes_dict,sample_loader,tram_model,latent_sample=latent_sample,device=device, debug=debug,minmax_dict=minmax_dict)
            
        ###*************************************************** Ordinal Modelled Outcome ************************************************
        
        elif is_outcome_modelled_ordinal(node,target_nodes_dict):
            sampled=sample_ordinal_modelled_target(sample_loader,tram_model,latent_sample=latent_sample,device=device, debug=debug)
        
        else:
            raise ValueError(f"Unsupported data_type '{target_nodes_dict[node]['data_type']}' for node '{node}' in sampling.")
        return sampled

def sample_node_detParents_intervalLatent(node,target_nodes_dict,number_of_samples,batch_size,tram_model,predefined_latent_samples_df,debug,EXPERIMENT_DIR,device):
    if debug:
            print(f"[DEBUG] -------sample_node_detParents_intervalLatent: start sampling for {node}---------")    
    sample_df=create_df_from_sampled(node, target_nodes_dict, number_of_samples, EXPERIMENT_DIR)
    counterfactual_frequency=[]
    for i ,_ in tqdm(enumerate(predefined_latent_samples_df[f"{node}"]),total=len(predefined_latent_samples_df[f"{node}"]),desc=f"Sampling {node}"):
        ### Sampling from truncated logistic
        u_lower=predefined_latent_samples_df.iloc[i][f"{node}_U_lower"]
        u_upper=predefined_latent_samples_df.iloc[i][f"{node}_U_upper"]
        latents_from_range=standart_logistic_truncated(u_lower, u_upper, number_of_samples)
        # save the sampled latens to csv
        RANGED_LATENT_PATH=os.path.join(EXPERIMENT_DIR,node ,"sampling","counterfactual",f'latents_range_obs_{i}.csv')
        # save to csv
        os.makedirs(os.path.dirname(RANGED_LATENT_PATH), exist_ok=True)
        np.savetxt(RANGED_LATENT_PATH, latents_from_range, delimiter=",")

        # repeat the i-th row number_of_samples times
        df_rep_i = sample_df.iloc[np.repeat(i, number_of_samples)].reset_index(drop=True)

        sample_dataset = GenericDataset(df_rep_i,target_col=node,
                                            target_nodes=target_nodes_dict,
                                            return_intercept_shift=True,
                                            return_y=False,
                                            debug=debug)
        sample_loader = DataLoader(sample_dataset, batch_size=batch_size, shuffle=False,num_workers=4, pin_memory=True)  
        
        if is_outcome_modelled_ordinal(node,target_nodes_dict):
            range_sampled=sample_ordinal_modelled_target(sample_loader,tram_model,latent_sample=torch.tensor(latents_from_range, dtype=torch.float32).to(device),device=device, debug=debug)
        

        RANGED_SAMPLED_PATH=os.path.join(EXPERIMENT_DIR,node ,"sampling","counterfactual",f'latents_sampled_obs_{i}.csv')
        os.makedirs(os.path.dirname(RANGED_SAMPLED_PATH), exist_ok=True)
        if isinstance(range_sampled, torch.Tensor):
            range_sampled = range_sampled.detach().cpu().squeeze().numpy()
            range_sampled = np.atleast_1d(range_sampled)
            np.savetxt(RANGED_SAMPLED_PATH, range_sampled, delimiter=",")

        # frequency count of how many times each category was sampled as proba distributin e.g. 3 classes: [0.1,0.3,0.6]     
        
        levels = target_nodes_dict[node]['levels']
        counts = np.bincount(range_sampled, minlength=levels)
        frequencies = counts / counts.sum()
        counterfactual_frequency.append(frequencies)
    if debug:
        print(f"[DEBUG] saved counterfactual range latents to e.g. {RANGED_LATENT_PATH}")
        print(f"[DEBUG] saved counterfactual sampling to e.g. {RANGED_SAMPLED_PATH}")
        print(f"[DEBUG]  Counterfactual frequencies: {counterfactual_frequency}")
        print(f"[DEBUG] -------sample_node_detParents_intervalLatent: sampling completed for {node}-------")    
    
    return torch.Tensor(np.array(counterfactual_frequency))
# TODO validiate sampling
def sample_node_distParents_intervalLatent(node,target_nodes_dict,number_of_samples,batch_size,tram_model,predefined_latent_samples_df,debug,EXPERIMENT_DIR,device):
    sample_df=create_df_from_sampled(node, target_nodes_dict, number_of_samples, EXPERIMENT_DIR)
    counterfactual_frequency=[]
    parents=target_nodes_dict[node]['parents']
    counterfactual_frequency=[]
    for i ,_ in tqdm(enumerate(predefined_latent_samples_df[f"{node}"]),
                    total=len(predefined_latent_samples_df[f"{node}"]),
                    desc=f"Sampling {node}"):
                u_lower=predefined_latent_samples_df.iloc[i][f"{node}_U_lower"]
                u_upper=predefined_latent_samples_df.iloc[i][f"{node}_U_upper"]
                latents_from_range=standart_logistic_truncated(u_lower, u_upper, number_of_samples)
                RANGED_LATENT_PATH=os.path.join(EXPERIMENT_DIR,node ,"sampling","counterfactual",f'latents_range_obs_{i}.csv')
                # save to csv
                os.makedirs(os.path.dirname(RANGED_LATENT_PATH), exist_ok=True)
                np.savetxt(RANGED_LATENT_PATH, latents_from_range, delimiter=",")
                df_rep_i=load_parents_with_range(sample_df, i, parents, number_of_samples, EXPERIMENT_DIR, debug=False)

                sample_dataset = GenericDataset(df_rep_i,target_col=node,
                                                    target_nodes=target_nodes_dict,
                                                    return_intercept_shift=True,
                                                    return_y=False,
                                                    debug=debug)
                sample_loader = DataLoader(sample_dataset, batch_size=batch_size, shuffle=False,num_workers=4, pin_memory=True)  
                
                
                
                if is_outcome_modelled_ordinal(node,target_nodes_dict):
                    range_sampled=sample_ordinal_modelled_target(sample_loader,tram_model,latent_sample=torch.tensor(latents_from_range, dtype=torch.float32).to(device),device=device, debug=debug)
                
                RANGED_SAMPLED_PATH=os.path.join(EXPERIMENT_DIR,node ,"sampling","counterfactual",f'latents_sampled_obs_{i}.csv')
                os.makedirs(os.path.dirname(RANGED_SAMPLED_PATH), exist_ok=True)
                if isinstance(range_sampled, torch.Tensor):
                    range_sampled = range_sampled.detach().cpu().squeeze().numpy()
                    range_sampled = np.atleast_1d(range_sampled)
                    np.savetxt(RANGED_SAMPLED_PATH, range_sampled, delimiter=",")
                # frequency count of how many times each category was sampled as proba distributin e.g. 3 classes: [0.1,0.3,0.6]     
                levels = target_nodes_dict[node]['levels']
                counts = np.bincount(range_sampled, minlength=levels)
                frequencies = counts / counts.sum()
                counterfactual_frequency.append(frequencies)
    if debug:
        print(f"[DEBUG] saved counterfactual range latents to e.g. {RANGED_LATENT_PATH}")
        print(f"[DEBUG] saved counterfactual sampling to e.g. {RANGED_SAMPLED_PATH}")
        print(f"[DEBUG] sample_node_distParents_intervalLatent: Counterfactual frequencies: {counterfactual_frequency}")
        print(f"[DEBUG]------- sample_node_distParents_intervalLatent: sampling completed for {node}-------")    
                
    return torch.Tensor(np.array(counterfactual_frequency))
# TODO solve sampling here
def sample_node_distParents_detLatent(node,target_nodes_dict,number_of_samples,batch_size,tram_model,predefined_latent_samples_df,latent_sample,debug,minmax_dict,EXPERIMENT_DIR):
    
    sample_df=create_df_from_sampled(node, target_nodes_dict, number_of_samples, EXPERIMENT_DIR)
    counterfactual_frequency=[]
    parents=target_nodes_dict[node]['parents']
    counterfactual_frequency=[]
    if debug:
        print(f"[DEBUG] starting sample_node_distParents_detLatent for node {node} with {number_of_samples} samples")
        
    for i ,_ in tqdm(enumerate(predefined_latent_samples_df[f"{node}"]),total=len(predefined_latent_samples_df[f"{node}"]),desc=f"Sampling {node}"):

        df_rep_i=load_parents_with_range(sample_df, i, parents, number_of_samples, EXPERIMENT_DIR, debug=False)

        sample_dataset = GenericDataset(df_rep_i,target_col=node,
                                            target_nodes=target_nodes_dict,
                                            return_intercept_shift=True,
                                            return_y=False,
                                            debug=debug)
        sample_loader = DataLoader(sample_dataset, batch_size=batch_size, shuffle=False,num_workers=4, pin_memory=True)  
        
        
        if is_outcome_modelled_continous(node,target_nodes_dict):
            range_sampled=sample_continous_modelled_target(node,target_nodes_dict,sample_loader,tram_model,latent_sample=latent_sample,device=device, debug=debug,minmax_dict=minmax_dict)
                
        if is_outcome_modelled_ordinal(node,target_nodes_dict):
            range_sampled=sample_ordinal_modelled_target(sample_loader,tram_model,latent_sample=latents,device=device, debug=debug)
        
        RANGED_SAMPLED_PATH=os.path.join(EXPERIMENT_DIR,node ,"sampling","counterfactual",f'latents_sampled_obs_{i}.csv')
        os.makedirs(os.path.dirname(RANGED_SAMPLED_PATH), exist_ok=True)
        if isinstance(range_sampled, torch.Tensor):
            range_sampled = range_sampled.detach().cpu().squeeze().numpy()
            range_sampled = np.atleast_1d(range_sampled)
            np.savetxt(RANGED_SAMPLED_PATH, range_sampled, delimiter=",")

        # frequency count of how many times each category was sampled as proba distributin e.g. 3 classes: [0.1,0.3,0.6]     
        levels = target_nodes_dict[node]['levels'] # in TODO continous case there will be no levels
        counts = np.bincount(range_sampled, minlength=levels)
        frequencies = counts / counts.sum()
        counterfactual_frequency.append(frequencies)
    if debug:
        print(f"[DEBUG] saved counterfactual sampling to e.g. {RANGED_SAMPLED_PATH}")
        print(f"[DEBUG] Counterfactual frequencies: {counterfactual_frequency}")
        print(f"[DEBUG] -------sample_node_distParents_detLatent: sampling completed for {node}-------")    

                # TODO what happens if node is continous and has porba parents
    return torch.Tensor(np.array(counterfactual_frequency))

def has_parents_with_distribution_samples(target_nodes_dict,node,debug=False,EXPERIMENT_DIR=None):
        for parent in target_nodes_dict[node]['parents']:
            path = os.path.join(EXPERIMENT_DIR, parent, "sampling", "sampled.pt")
            if os.path.exists(path):
                try:
                    sampled=torch.load(path, map_location="cpu")
                    if sampled.ndim > 1 and sampled.shape[1] > 1:
                        if debug:
                            print(f"[DEBUG] has_parents_with_distribution_samples:  '{parent}' (shape {tuple(sampled.shape)}) - looks like probabilities")
                        return True
                    else:
                        continue
                except FileNotFoundError:
                    if debug:
                        print(f"[DEBUG] has_parents_with_distribution_samples: Skipping {node} as parent '{parent}' is not sampled yet.")
        return False

def load_parents_with_range(df_sampled, i, parents, number_of_samples, EXPERIMENT_DIR, debug=False):
    # Check index validity
    if i < 0 or i >= len(df_sampled):
        raise IndexError(f"Index {i} out of bounds for df_sampled with {len(df_sampled)} rows.")
    
    # Create repeated DataFrame
    df_range = pd.DataFrame(
        np.tile(df_sampled.iloc[i].values, (number_of_samples, 1)),
        columns=df_sampled.columns
    )

    # For missing parent variables, load sampled ranges from files
    for parent in parents:
        if parent not in df_range.columns:
            path = os.path.join(EXPERIMENT_DIR, parent, "sampling", f"range_samples_{i}_.csv")
            
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing file: {path}")
            
            df_parent = pd.read_csv(path)
            
            if len(df_parent) != number_of_samples:
                raise ValueError(
                    f"File {path} has {len(df_parent)} samples, expected {number_of_samples}."
                )
            
            if df_parent.shape[1] != 1:
                raise ValueError(
                    f"File {path} has {df_parent.shape[1]} columns, expected exactly 1."
                )
            
            df_range[parent] = df_parent.iloc[:, 0].values
    
    if debug:
        print(f"[DEBUG] Loaded row {i} -> {len(df_range)} samples")
        print(f"[DEBUG] Columns: {list(df_range.columns)}")

    return df_range
    pass

def standart_logistic_truncated(u_lower, u_upper, size):
    # sample n in standard logistic and truncate to [u_lower, u_upper] resample if out of bounds:
    samples = []
    while len(samples) < size:
        u = logistic.rvs(size=size - len(samples))
        u_trunc = u[(u >= u_lower) & (u <= u_upper)]
        samples.extend(u_trunc)
    return np.array(samples[:size])

def truncated_logistic_sample(n, low, high, device='cpu'):
    samples = []
    while len(samples) < n:
        new_samples = logistic.rvs(size=n - len(samples))
        valid = new_samples[(new_samples >= low) & (new_samples <= high)]
        samples.extend(valid)
    return torch.tensor(samples, dtype=torch.float32).to(device)

def load_tram_model_weights(node,configuration_dict, NODE_DIR,debug=False,verbose=False, device='cpu',  use_initial_weights_for_sampling=False):

            tram_model = get_fully_specified_tram_model(
                node, configuration_dict, debug=debug, device=device, verbose=verbose
            ).to(device)

            BEST_MODEL_PATH = os.path.join(NODE_DIR, "best_model.pt")
            INIT_MODEL_PATH = os.path.join(NODE_DIR, "initial_model.pt")

            try:
                if use_initial_weights_for_sampling:
                    if verbose or debug:
                        print(f"[INFO] Using initial weights for sampling for node '{node}'")
                    if not os.path.exists(INIT_MODEL_PATH):
                        raise FileNotFoundError(f"Initial model not found at {INIT_MODEL_PATH}")
                    tram_model.load_state_dict(torch.load(INIT_MODEL_PATH, map_location=device))

                else:
                    if os.path.exists(BEST_MODEL_PATH):
                        tram_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
                        if verbose or debug:
                            print(f"[INFO] Loaded best model weights for node '{node}' from {BEST_MODEL_PATH}")
                    elif os.path.exists(INIT_MODEL_PATH):
                        tram_model.load_state_dict(torch.load(INIT_MODEL_PATH, map_location=device))
                        print(f"[WARNING] Best model not found for node '{node}'. Using initial weights instead.")
                    else:
                        raise FileNotFoundError(
                            f"No model weights found for node '{node}'. "
                            f"Expected one of: {BEST_MODEL_PATH} or {INIT_MODEL_PATH}"
                        )

            except Exception as e:
                print(f"[ERROR] Failed to load model weights for node '{node}': {e}")
                raise
            return tram_model

################################## Plotting Helpers #####################################
def show_hdag_for_single_source_node_continous(node,configuration_dict,device='cpu',xmin_plot=None,xmax_plot=None,EXPERIMENT_DIR=None,verbose=False,debug=False,minmax_dict=None):
        target_nodes=configuration_dict["nodes"]
        
        verbose=False
        n=1000
 
        
        
        if EXPERIMENT_DIR is None:
            EXPERIMENT_DIR=configuration_dict["PATHS"]["EXPERIMENT_DIR"]
        
        if minmax_dict is not None:
            min_vals = torch.tensor(minmax_dict[node][0], dtype=torch.float32).to(device)
            max_vals = torch.tensor(minmax_dict[node][1], dtype=torch.float32).to(device)
            min_max = torch.stack([min_vals, max_vals], dim=0)    
        
        diff = max_vals - min_vals
        
        
        if xmin_plot is None:
            xmin_plot = (min_vals - 0.1 * diff).item()
        if xmax_plot is None:
            xmax_plot = (max_vals + 0.1 * diff).item()
            
            
               #### 0.  paths
        NODE_DIR = os.path.join(EXPERIMENT_DIR, f'{node}')
        
        ##### 1.  load model 
        model_path = os.path.join(NODE_DIR, "best_model.pt")
        if not os.path.exists(model_path):
            print("[Warning] best_model.pt not found, falling back to initial_model.pt")

            model_path = os.path.join(NODE_DIR, "initial_model.pt")
        tram_model = get_fully_specified_tram_model(node, configuration_dict, debug=verbose, set_initial_weights=False,device=device)
        tram_model.load_state_dict(torch.load(model_path, map_location=device))
        tram_model = tram_model.to(device)

        
        sampled_df=create_df_from_sampled(node, target_nodes, num_samples=n, EXPERIMENT_DIR=EXPERIMENT_DIR)

        sample_dataset = GenericDataset(sampled_df,target_col=node,
                                            target_nodes=target_nodes,
                                            return_intercept_shift=True,
                                            return_y=False,
                                            verbose=verbose,
                                            debug=debug)
        
        sample_loader = DataLoader(sample_dataset, batch_size=1, shuffle=True,num_workers=4, pin_memory=True)
        
        output_list = []
        with torch.no_grad():
            for (int_input, shift_list) in sample_loader:
                # Move everything to device
                print(f"Model devices: {[p.device for p in tram_model.parameters()][:3]}")
                print(f"int_input: {int_input.device}")
                print(f"shift_list: {[s.device for s in shift_list]}")
                
                int_input = int_input.to(device, non_blocking=True)
                shift_list = [s.to(device, non_blocking=True) for s in shift_list]
                model_outputs = tram_model(int_input=int_input, shift_input=shift_list)
                output_list.append(model_outputs)
                break
        if verbose:
            print("source node, Defaults to SI and 1 as inputs")
            
        theta_single =     output_list[0]['int_out'][0]  # Shape: (20,)
        theta_single=transform_intercepts_continous(theta_single)
        thetas_expanded = theta_single.repeat(n, 1).to(device)  # Shape: (n, 20)
        

        
        if xmin_plot==None:
            xmin_plot=min_vals-1
        if xmax_plot==None:
            xmax_plot=max_vals+1        
        
        targets2 = torch.linspace(xmin_plot, xmax_plot, steps=n).to(device)  # 1000 points from 0 to 1
        
        min_val = torch.as_tensor(min_max[0], dtype=targets2.dtype, device=device)
        max_val = torch.as_tensor(min_max[1], dtype=targets2.dtype, device=device)
        
        hdag_extra_values=h_extrapolated(thetas_expanded, targets2, k_min=min_val, k_max=max_val)
        # Move to CPU for plotting
        targets2_cpu = targets2.cpu().numpy()
        hdag_extra_values_cpu = hdag_extra_values.cpu().detach().numpy()

        # # Split masks
        below_min_mask = targets2_cpu < min_val.item()
        between_mask = (targets2_cpu >= min_val.item()) & (targets2_cpu <= max_val.item())
        above_max_mask = targets2_cpu > max_val.item()

        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(targets2_cpu[below_min_mask], hdag_extra_values_cpu[below_min_mask], color='red', label='x < min_val')
        plt.plot(targets2_cpu[between_mask], hdag_extra_values_cpu[between_mask], color='blue', label='min_val <= x <= max_val')
        plt.plot(targets2_cpu[above_max_mask], hdag_extra_values_cpu[above_max_mask], color='red', label='x > max_val')
        plt.xlabel('Targets (x)');plt.ylabel('h_dag_extra(x)');plt.title('h_dag output over targets');plt.grid(True);plt.legend();plt.show()

def show_hdag_continous(df,
                        node,
                        configuration_dict,
                        device='cpu',
                        xmin_plot=None,
                        xmax_plot=None,
                        EXPERIMENT_DIR=None,
                        verbose=False,
                        debug=False,
                        minmax_dict=None,
                        plot_n_rows=1):
    
        target_nodes=configuration_dict["nodes"]
        
        verbose=False
        n_linspace=1000
        if EXPERIMENT_DIR is None:
            EXPERIMENT_DIR=configuration_dict["PATHS"]["EXPERIMENT_DIR"]
        
        if minmax_dict is not None:
            min_vals = torch.tensor(minmax_dict[node][0], dtype=torch.float32).to(device)
            max_vals = torch.tensor(minmax_dict[node][1], dtype=torch.float32).to(device)
            min_max = torch.stack([min_vals, max_vals], dim=0)    
        diff = max_vals - min_vals
        
        if xmin_plot is None:
            xmin_plot = (min_vals - 0.1 * diff).item()
        if xmax_plot is None:
            xmax_plot = (max_vals + 0.1 * diff).item()
            
        #### 0.  paths
        NODE_DIR = os.path.join(EXPERIMENT_DIR, f'{node}')
        
        ##### 1.  load model 
        model_path = os.path.join(NODE_DIR, "best_model.pt")
        if not os.path.exists(model_path):
            print("[Warning] best_model.pt not found, falling back to initial_model.pt")
            model_path = os.path.join(NODE_DIR, "initial_model.pt")
            
        tram_model = get_fully_specified_tram_model(node, configuration_dict, debug=verbose,device=device)
        tram_model.load_state_dict(torch.load(model_path, map_location=device))
        tram_model = tram_model.to(device)
        tram_model.eval()

        # sampled_df=create_df_from_sampled(node, target_nodes, num_samples=n, EXPERIMENT_DIR=EXPERIMENT_DIR)

        sample_dataset = GenericDataset(df,target_col=node,
                                            target_nodes=target_nodes,
                                            return_intercept_shift=True,
                                            return_y=False,
                                            verbose=verbose,
                                            debug=debug)
        
        sample_loader = DataLoader(sample_dataset, batch_size=1, shuffle=False,num_workers=4, pin_memory=True)
        
        counter = 0
        stop_plot = 1 if target_nodes[node]["node_type"] == "source" else min(len(df), plot_n_rows)

        with torch.no_grad():
            if target_nodes[node]["node_type"] == "source":
                print(f"{node}: Simple intercept — identical h() for all samples.")

            for i, (int_input, shift_list) in enumerate(sample_loader):
                if counter >= stop_plot:
                    break

                if target_nodes[node]["node_type"] != "source":
                    print(f"\n=== Sample {i+1}/{stop_plot} ===")
                    print(df.iloc[i])  # print current row's values

                int_input = int_input.to(device, non_blocking=True)
                shift_list = [s.to(device, non_blocking=True) for s in shift_list]

                model_outputs = tram_model(int_input=int_input, shift_input=shift_list)
                theta_single = model_outputs["int_out"][0]
                theta_single = transform_intercepts_continous(theta_single)
                thetas_expanded = theta_single.repeat(n_linspace, 1).to(device)

                if xmin_plot is None:
                    xmin_plot = min_vals - 1
                if xmax_plot is None:
                    xmax_plot = max_vals + 1

                targets2 = torch.linspace(xmin_plot, xmax_plot, steps=n_linspace).to(device)
                min_val = torch.as_tensor(min_max[0], dtype=targets2.dtype, device=device)
                max_val = torch.as_tensor(min_max[1], dtype=targets2.dtype, device=device)

                hdag_extra_values = h_extrapolated(
                    thetas_expanded, targets2, k_min=min_val, k_max=max_val
                )

                targets2_cpu = targets2.cpu().numpy()
                hdag_extra_values_cpu = hdag_extra_values.cpu().detach().numpy()

                below_min_mask = targets2_cpu < min_val.item()
                between_mask = (targets2_cpu >= min_val.item()) & (
                    targets2_cpu <= max_val.item()
                )
                above_max_mask = targets2_cpu > max_val.item()

                plt.figure(figsize=(8, 6))
                plt.plot(
                    targets2_cpu[below_min_mask],
                    hdag_extra_values_cpu[below_min_mask],
                    color="red",
                    label="x < min_val",
                )
                plt.plot(
                    targets2_cpu[between_mask],
                    hdag_extra_values_cpu[between_mask],
                    color="blue",
                    label="min_val <= x <= max_val",
                )
                plt.plot(
                    targets2_cpu[above_max_mask],
                    hdag_extra_values_cpu[above_max_mask],
                    color="red",
                    label="x > max_val",
                )
                plt.xlabel(f" ({node})")
                plt.ylabel(f"h({node}|{' '.join(target_nodes[node]['parents'])})")
                plt.title(f"transformation function {node} — sample {i+1}")
                plt.grid(True)
                plt.legend()
                plt.show()

                counter += 1

def show_hdag_ordinal(df,
                      node,
                      configuration_dict,
                      device='cpu',
                      xmin_plot=None,
                      xmax_plot=None,
                      EXPERIMENT_DIR=None,
                      verbose=False,
                      debug=False,
                      plot_n_rows=1): 

    target_nodes = configuration_dict["nodes"]
    verbose = False
    if EXPERIMENT_DIR is None:
        EXPERIMENT_DIR = configuration_dict["PATHS"]["EXPERIMENT_DIR"]

    # === Paths ===
    NODE_DIR = os.path.join(EXPERIMENT_DIR, f'{node}')
    model_path = os.path.join(NODE_DIR, "best_model.pt")
    if not os.path.exists(model_path):
        print("[Warning] best_model.pt not found, falling back to initial_model.pt")
        model_path = os.path.join(NODE_DIR, "initial_model.pt")

    # === Load model ===
    tram_model = get_fully_specified_tram_model(node, configuration_dict, debug=verbose, device=device)
    tram_model.load_state_dict(torch.load(model_path, map_location=device))
    tram_model = tram_model.to(device)
    tram_model.eval()

    # === Dataset and loader ===
    sample_dataset = GenericDataset(
        df,
        target_col=node,
        target_nodes=target_nodes,
        return_intercept_shift=True,
        return_y=False,
        verbose=verbose,
        debug=debug
    )
    sample_loader = DataLoader(sample_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    counter = 0
    stop_plot = 1 if target_nodes[node]["node_type"] == "source" else min(len(df), plot_n_rows)

    with torch.no_grad():
        if target_nodes[node]["node_type"] == "source":
            print(f"{node}: Simple intercept — identical h() for all samples.")

        for i, (int_input, shift_list) in enumerate(sample_loader):
            if counter >= stop_plot:
                break

            if target_nodes[node]["node_type"] != "source":
                print(f"\n=== Sample {i+1}/{stop_plot} ===")
                print(df.iloc[i])

            int_input = int_input.to(device, non_blocking=True)
            shift_list = [s.to(device, non_blocking=True) for s in shift_list]

            # === Forward pass ===
            model_outputs = tram_model(int_input=int_input, shift_input=shift_list)

            # === Extract and transform intercepts ===
            int_in = model_outputs['int_out']
            shift_in = model_outputs['shift_out']
            int_trans = transform_intercepts_ordinal(int_in)  # [B, K+1]

            # === Handle shifts ===
            if shift_in is not None:
                shift = torch.stack(shift_in, dim=1).sum(dim=1).view(-1)
                int_shifted = int_trans - shift.unsqueeze(1)
            else:
                int_shifted = int_trans

            # === Select first (and only) sample in batch ===
            cutpoints = int_shifted[0, 1:-1].cpu().numpy()

            # === Plot cutpoints vs class indices ===
            plt.figure(figsize=(6, 4))
            x_classes = np.arange(1, len(cutpoints) + 1)

            plt.plot(x_classes, cutpoints, 'o', color='C1', linewidth=2, markersize=8)
            plt.xticks(x_classes)
            plt.xlabel("Ordinal class index (k)")
            plt.ylabel("Shifted cutpoint (θₖ - ηᵢ)")
            plt.title(f"{node} — Sample {i+1}")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.show()

            counter += 1

def show_hdag_for_source_nodes(configuration_dict,EXPERIMENT_DIR,device,xmin_plot=-5,xmax_plot=5):
    target_nodes=configuration_dict["nodes"]
    for node in target_nodes:

        print(f'\n----*----------*-------------*--------Inspect TRAFO Node: {node} ------------*-----------------*-------------------*--')
        
        if (target_nodes[node]['node_type'] != 'source'):
            print("skipped.. since h does depend on parents and is different for every instance")
            continue
        else:
            if is_outcome_modelled_continous(node, target_nodes):
                return show_hdag_for_single_source_node_continous(node=node,configuration_dict=configuration_dict,EXPERIMENT_DIR=EXPERIMENT_DIR,device=device,xmin_plot=xmin_plot,xmax_plot=xmax_plot)
            
            if is_outcome_modelled_ordinal(node, target_nodes):
                print('not implemeneted yet for ordinal (nominally encoded)')
 
def inspect_trafo_standart_logistic(configuration_dict, EXPERIMENT_DIR, train_df, val_df, device, verbose=False):
    target_nodes = configuration_dict["nodes"]
    h_train_outputs = []
    h_val_outputs = []
    for node in target_nodes:
        print(f'----*----------*-------------*--------h(data) should be standard logistic: {node} ------------*-----------------*-------------------*--')
        if is_outcome_modelled_ordinal(node, target_nodes):
            print('not defined for ordinal target variables')
            continue
        else:
            # Get h_train and h_val for this node
            h_train, h_val = inspect_single_standart_logistic(
                node, configuration_dict, EXPERIMENT_DIR, train_df, val_df, device, verbose=verbose, return_intercept_shift=True
            )
            h_train_outputs.append(h_train)
            h_val_outputs.append(h_val)
    # Stack outputs into arrays with shape (n_samples, n_nodes)
    # Each h_train/h_val is a 1D array for a node; stack as columns
    h_train_arr = np.column_stack(h_train_outputs) if h_train_outputs else np.array([])
    h_val_arr = np.column_stack(h_val_outputs) if h_val_outputs else np.array([])
    return h_train_arr, h_val_arr

def inspect_single_standart_logistic(
    node,
    configuration_dict,
    EXPERIMENT_DIR,
    train_df,
    val_df,
    device,
    return_intercept_shift: bool = True,
    verbose: bool = False
):
    
    target_nodes=configuration_dict["nodes"]
    
    #### 0. Paths
    NODE_DIR = os.path.join(EXPERIMENT_DIR, f"{node}")
    
    ##### 1. Load model 
    model_path = os.path.join(NODE_DIR, "best_model.pt")
    tram_model = get_fully_specified_tram_model(node, configuration_dict, debug=verbose, set_initial_weights=False)
    tram_model = tram_model.to(device)
    tram_model.load_state_dict(torch.load(model_path, map_location=device))
    tram_model.eval()

    ##### 2. Dataloader
    train_loader, val_loader = get_dataloader(
        node,
        target_nodes,
        train_df,
        val_df,
        batch_size=4112,
        return_intercept_shift=return_intercept_shift,
        debug=verbose
    )
    
    #### 3. Forward Pass
    min_vals = torch.tensor(target_nodes[node]["min"], dtype=torch.float32, device=device)
    max_vals = torch.tensor(target_nodes[node]["max"], dtype=torch.float32, device=device)
    min_max = torch.stack([min_vals, max_vals], dim=0)

    h_train_list, h_val_list = [], []
    with torch.no_grad():
        for (int_input, shift_list), y in train_loader:
            # Move everything to device
            int_input = int_input.to(device)
            shift_list = [s.to(device) for s in shift_list]
            y = y.to(device)                            # ← move targets to GPU/CPU
            y_pred = tram_model(int_input=int_input, shift_input=shift_list)
            h_train, _ = contram_nll(
                y_pred, y, min_max=min_max, return_h=True
            )
            h_train_list.extend(h_train.cpu().numpy())            

        for (int_input, shift_list), y in val_loader:
            int_input = int_input.to(device)
            shift_list = [s.to(device) for s in shift_list]
            y = y.to(device)                            # ← move targets as well!
            y_pred = tram_model(int_input=int_input, shift_input=shift_list)
            h_val, _ = contram_nll(
                y_pred, y, min_max=min_max, return_h=True
            )
            h_val_list.extend(h_val.cpu().numpy())

    h_train_array = np.array(h_train_list)
    h_val_array = np.array(h_val_list)

    # Uncomment below to plot diagnostics
    fig, axs = plt.subplots(1, 4, figsize=(22, 5))
    
    # Train Histogram
    axs[0].hist(h_train_array, bins=50)
    axs[0].set_title(f"Train Histogram ({node})")
    
    # Train QQ Plot with R-style Confidence Bands
    probplot(h_train_array, dist="logistic", plot=axs[1])
    add_r_style_confidence_bands(axs[1], h_train_array)
    
    # Validation Histogram
    axs[2].hist(h_val_array, bins=50)
    axs[2].set_title(f"Val Histogram ({node})")
    
    # Validation QQ Plot with R-style Confidence Bands
    probplot(h_val_array, dist="logistic", plot=axs[3])
    add_r_style_confidence_bands(axs[3], h_val_array)
    
    plt.suptitle(f"Distribution Diagnostics for Node: {node}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    return h_train_array, h_val_array

def add_r_style_confidence_bands(ax, sample, dist=logistic, confidence=0.95, simulations=1000):
    """
    Adds accurate confidence bands to a QQ plot using simulation under the null hypothesis.
    """
    n = len(sample)
    quantiles = np.linspace(0, 1, n, endpoint=False) + 0.5 / n
    theo_q = dist.ppf(quantiles)

    # Simulate order statistics from the theoretical distribution
    sim_data = dist.rvs(size=(simulations, n))
    sim_order_stats = np.sort(sim_data, axis=1)

    # Compute confidence bands for each order statistic
    lower = np.percentile(sim_order_stats, 100 * (1 - confidence) / 2, axis=0)
    upper = np.percentile(sim_order_stats, 100 * (1 + confidence) / 2, axis=0)

    # Sort the empirical sample
    sample_sorted = np.sort(sample)

    # Plot the empirical Q-Q line
    ax.plot(theo_q, sample_sorted, linestyle='None', marker='o', markersize=3, alpha=0.6)
    ax.plot(theo_q, theo_q, 'b--', label='y = x')

    # Confidence band
    ax.fill_between(theo_q, lower, upper, color='gray', alpha=0.3, label=f'{int(confidence*100)}% CI')
    ax.legend()
   
def show_samples_vs_true(
    df,
    target_nodes,
    experiment_dir,
    *,
    bins=100,
    hist_true_color="blue",
    hist_est_color="orange",
    figsize=(14, 5),
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for node in target_nodes:
        sample_path = os.path.join(experiment_dir, f"{node}/sampling/sampled.pt")
        if not os.path.isfile(sample_path):
            print(f"[WARNING] skip {node}: {sample_path} not found.")
            continue

        try:
            sampled = torch.load(sample_path, map_location=device).cpu().numpy()
        except Exception as e:
            print(f"[ERROR] Could not load {sample_path}: {e}")
            continue

        if sampled.ndim == 2:
            sampled = np.argmax(sampled, axis=1)

        sampled = sampled[np.isfinite(sampled)]

        if node not in df.columns:
            print(f"[WARNING] skip {node}: column not found in DataFrame.")
            continue

        true_vals = df[node].dropna().values
        true_vals = true_vals[np.isfinite(true_vals)]

        if sampled.size == 0 or true_vals.size == 0:
            print(f"[WARNING] skip {node}: empty array after NaN/Inf removal.")
            continue

        fig, axs = plt.subplots(1, 2, figsize=figsize)

        if is_outcome_modelled_continous(node, target_nodes):
            axs[0].hist(true_vals, bins=bins, density=True, alpha=0.6,
                        color=hist_true_color, label=f"True {node}")
            axs[0].hist(sampled, bins=bins, density=True, alpha=0.6,
                        color=hist_est_color, label="Sampled")
            axs[0].set_xlabel("Value")
            axs[0].set_ylabel("Density")
            axs[0].set_title(f"Histogram overlay for {node}")
            axs[0].legend()
            axs[0].grid(True, ls="--", alpha=0.4)

            qqplot_2samples(true_vals, sampled, line="45", ax=axs[1])
            axs[1].set_xlabel("True quantiles")
            axs[1].set_ylabel("Sampled quantiles")
            axs[1].set_title(f"QQ plot for {node}")
            axs[1].grid(True, ls="--", alpha=0.4)

        elif is_outcome_modelled_ordinal(node, target_nodes):
            unique_vals = np.union1d(np.unique(true_vals), np.unique(sampled))
            unique_vals = np.sort(unique_vals)
            true_counts = np.array([(true_vals == val).sum() for val in unique_vals])
            sampled_counts = np.array([(sampled == val).sum() for val in unique_vals])

            axs[0].bar(unique_vals - 0.2, true_counts / true_counts.sum(),
                       width=0.4, color=hist_true_color, alpha=0.7, label="True")
            axs[0].bar(unique_vals + 0.2, sampled_counts / sampled_counts.sum(),
                       width=0.4, color=hist_est_color, alpha=0.7, label="Sampled")
            axs[0].set_xticks(unique_vals)
            axs[0].set_xlabel("Ordinal Level")
            axs[0].set_ylabel("Relative Frequency")
            axs[0].set_title(f"Ordinal bar plot for {node}")
            axs[0].legend()
            axs[0].grid(True, ls="--", alpha=0.4)
            axs[1].axis("off")

        else:
            unique_vals = np.union1d(np.unique(true_vals), np.unique(sampled))
            unique_vals = sorted(unique_vals, key=str)
            true_counts = np.array([(true_vals == val).sum() for val in unique_vals])
            sampled_counts = np.array([(sampled == val).sum() for val in unique_vals])

            axs[0].bar(np.arange(len(unique_vals)) - 0.2, true_counts / true_counts.sum(),
                       width=0.4, color=hist_true_color, alpha=0.7, label="True")
            axs[0].bar(np.arange(len(unique_vals)) + 0.2, sampled_counts / sampled_counts.sum(),
                       width=0.4, color=hist_est_color, alpha=0.7, label="Sampled")
            axs[0].set_xticks(np.arange(len(unique_vals)))
            axs[0].set_xticklabels(unique_vals, rotation=45)
            axs[0].set_xlabel("Category")
            axs[0].set_ylabel("Relative Frequency")
            axs[0].set_title(f"Categorical bar plot for {node}")
            axs[0].legend()
            axs[0].grid(True, ls="--", alpha=0.4)
            axs[1].axis("off")

        plt.tight_layout()
        plt.show()

def show_latent_sampling(EXPERIMENT_DIR,conf_dict):
    for node in conf_dict.keys():
        latents_path = os.path.join(EXPERIMENT_DIR,f'{node}/sampling/latents.pt')
        latents = torch.load(latents_path, map_location="cpu").cpu().numpy()
        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        # Histogram
        axs[0].hist(latents, bins=100)
        axs[0].set_title(f"Histogram of Latents ({node})")
        axs[0].set_xlabel("Latent values")
        axs[0].set_ylabel("Frequency")
        axs[0].grid(True)
        # QQ Plot against standard logistic
        probplot(latents, dist="logistic", plot=axs[1])
        axs[1].set_title(f"QQ Plot (Logistic) - {node}")
        axs[1].grid(True)
        plt.tight_layout()
        plt.show()
        
