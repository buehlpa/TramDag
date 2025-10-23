
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import warnings
import os
import shutil
from statsmodels.graphics.gofplots import qqplot_2samples
from scipy.stats import logistic

from utils.tram_model_helpers import *        
from utils.loss_continous import *   
from utils.tram_data import *

# helpers

def merge_outputs(dict_list, skip_nan=True):
    int_outs = []
    shift_outs = []
    skipped_count = 0

    for d in dict_list:
        int_tensor = d['int_out']
        if type(d['shift_out']) is list:
            shift_tensor = d['shift_out'][0]
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


def show_hdag_for_single_source_node_continous(node,configuration_dict,EXPERIMENT_DIR,device,xmin_plot=-5,xmax_plot=5,verbose=False,debug=False):
        target_nodes=configuration_dict["nodes"]
        
        verbose=False
        n=1000
        #### 0.  paths
        NODE_DIR = os.path.join(EXPERIMENT_DIR, f'{node}')
        
        ##### 1.  load model 
        model_path = os.path.join(NODE_DIR, "best_model.pt")
        tram_model = get_fully_specified_tram_model(node, configuration_dict, debug=verbose, set_initial_weights=False)
        tram_model = tram_model.to(device)
        tram_model.load_state_dict(torch.load(model_path, map_location=device))
        
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
                int_input = int_input.to(device)
                shift_list = [s.to(device) for s in shift_list]
                model_outputs = tram_model(int_input=int_input, shift_input=shift_list)
                output_list.append(model_outputs)
                break
        if verbose:
            print("source node, Defaults to SI and 1 as inputs")
            
        theta_single =     output_list[0]['int_out'][0]  # Shape: (20,)
        theta_single=transform_intercepts_continous(theta_single)
        thetas_expanded = theta_single.repeat(n, 1).to(device)  # Shape: (n, 20)
        
        min_vals = torch.tensor(target_nodes[node]['min'], dtype=torch.float32).to(device)
        max_vals = torch.tensor(target_nodes[node]['max'], dtype=torch.float32).to(device)
        min_max = torch.stack([min_vals, max_vals], dim=0)
        
        if xmin_plot==None:
            xmin_plot=min_vals-1
        if xmax_plot==None:
            xmax_plot=max_vals+1        
        
        targets2 = torch.linspace(xmin_plot, xmax_plot, steps=n).to(device)  # 1000 points from 0 to 1
        
        min_val = min_max[0].clone().detach() if isinstance(min_max[0], torch.Tensor) else torch.tensor(min_max[0], dtype=targets2.dtype, device=targets2.device)
        max_val = min_max[1].clone().detach() if isinstance(min_max[1], torch.Tensor) else torch.tensor(min_max[1], dtype=targets2.dtype, device=targets2.device) 

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
            # Continuous: histogram + QQ
            axs[0].hist(
                true_vals,
                bins=bins,
                density=True,
                alpha=0.6,
                color=hist_true_color,
                label=f"True {node}",
            )
            axs[0].hist(
                sampled,
                bins=bins,
                density=True,
                alpha=0.6,
                color=hist_est_color,
                label="Sampled",
            )
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
            # Ordinal: bar plot only
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

            axs[1].axis("off")  # No QQ for ordinal

        else:
            # Fallback: assume categorical
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
        
def truncated_logistic_sample(n, low, high, device='cpu'):
    samples = []
    while len(samples) < n:
        new_samples = logistic.rvs(size=n - len(samples))
        valid = new_samples[(new_samples >= low) & (new_samples <= high)]
        samples.extend(valid)
    return torch.tensor(samples, dtype=torch.float32).to(device)



################################### SAMPLING HELPERS #####################################

        
from utils.loss_ordinal import get_pdf_ordinal, get_cdf_ordinal

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
            sampling_dict[parent] = torch.load(path, map_location="cpu")
        else:
            if debug:
                print(f"[DEBUG] create_df_from_sampled: no sampled data found for parent '{parent}' at {path}")

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

def sample_ordinal_modelled_target(sample_loader, tram_model, device, debug=False):
    all_outputs = []
    tram_model.eval()
    with torch.no_grad():
        for (int_input, shift_list) in sample_loader:
            int_input = int_input.to(device)
            shift_list = [s.to(device) for s in shift_list]

            model_outputs = tram_model(int_input=int_input, shift_input=shift_list)
            all_outputs.append(model_outputs)

            if debug:
                print("[DEBUG] Batch model_outputs keys:", model_outputs.keys())
                print("[DEBUG] int_out shape:", model_outputs['int_out'].shape)
                if model_outputs['shift_out'] is not None:
                    print("[DEBUG] shift_out shapes:", [s.shape for s in model_outputs['shift_out']])

    # Concatenate all 'int_out' and 'shift_out' elements across batches
    int_out_all = torch.cat([out['int_out'] for out in all_outputs], dim=0)

    # If shift_out is present, we assume it's a list of tensors
    if all_outputs[0]['shift_out'] is not None:
        shift_out_all = []
        for i in range(len(all_outputs[0]['shift_out'])):
            shift_i = torch.cat([out['shift_out'][i] for out in all_outputs], dim=0)
            shift_out_all.append(shift_i)
    else:
        shift_out_all = None

    merged_outputs = {
        'int_out': int_out_all,
        'shift_out': shift_out_all
    }

    cdf = get_cdf_ordinal(merged_outputs)
    pdf = get_pdf_ordinal(cdf)
    sampled = pdf.argmax(dim=1)

    if debug:
        print("[DEBUG] Final sampled shape:", sampled.shape)
        print("[DEBUG] Sampled labels (first 3):", sampled[:3])

    return sampled



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

    # Validate shapes
    if thetas_expanded.shape[0] != number_of_samples:
        raise ValueError(f"Mismatch in sample count: thetas_expanded has shape {thetas_expanded.shape}, expected {number_of_samples} rows.")

    if debug:
        print("[DEBUG] sample_continous_modelled_target: beginning root finding")
        print("[DEBUG] sample_continous_modelled_target: thetas_expanded shape:", thetas_expanded.shape)
        print("[DEBUG] sample_continous_modelled_target: shifts shape:", shifts.shape)
        print("[DEBUG] sample_continous_modelled_target: latent_sample shape:", latent_sample.shape)

    # Root bounds
    low = torch.full((number_of_samples,), -1e5, device=device)
    high = torch.full((number_of_samples,), 1e5, device=device)

    if minmax_dict is not None:
            min_vals = torch.tensor(minmax_dict[node][0], dtype=torch.float32, device=device)
            max_vals = torch.tensor(minmax_dict[node][1], dtype=torch.float32, device=device)
            min_max = torch.stack([min_vals, max_vals], dim=0)

    else:
        try:
            min_vals = torch.tensor(target_nodes_dict[node]['min'], dtype=torch.float32).to(device)
            max_vals = torch.tensor(target_nodes_dict[node]['max'], dtype=torch.float32).to(device)
        except KeyError as e:
            raise KeyError(f"Missing 'min' or 'max' value in target_nodes_dict for node '{node}': {e}")
        min_max = torch.stack([min_vals, max_vals], dim=0)

    
    # Vectorized root-finding function
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

    if sampled is None or torch.isnan(sampled).any():
        raise RuntimeError("Root finding failed: returned None or contains NaNs.")

    if debug:
        print("[DEBUG] sample_continous_modelled_target: root finding complete. Sampled shape:", sampled.shape)

    return sampled



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

def provide_latents_for_input_data(
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


def create_latent_df_for_full_dag(configuration_dict, EXPERIMENT_DIR, df, verbose=False, min_max_dict=None):

    all_latents_dfs = []

    for node in configuration_dict["nodes"]:
        
        if min_max_dict is not None and node in min_max_dict:
            min_vals = torch.tensor(min_max_dict[node][0], dtype=torch.float32)
            max_vals = torch.tensor(min_max_dict[node][1], dtype=torch.float32)
            min_max = torch.stack([min_vals, max_vals], dim=0)
        else:
            min_max=None
            
        # Skip ordinal outcomes if not supported
        if is_outcome_modelled_ordinal(node, configuration_dict["nodes"]):
            print(f"[INFO] Skipping node '{node}' (ordinal targets not yet supported).")
            continue

        if verbose:
            print(f"[INFO] Processing node '{node}'")

        # Build dataset and dataloader for this node
        node_dataset = GenericDataset(
            df,
            target_col=node,
            target_nodes=configuration_dict["nodes"]
        )
        node_loader = DataLoader(
            node_dataset,
            batch_size=4096,   # you can tune this
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # Compute latents
        node_latents_df = provide_latents_for_input_data(
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
    
    
def sample_full_dag(configuration_dict,
                    EXPERIMENT_DIR,
                    device,
                    do_interventions={},
                    predefined_latent_samples_df: pd.DataFrame = None,
                    number_of_samples: int = 10_000,
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
            # if latents are predefined use them
            if predefined_latent_samples_df is not None and node in predefined_latent_samples_df.columns.str.replace('_U',''):
                predefinded_sample_name = node + "_U" 

                if predefinded_sample_name not in predefined_latent_samples_df.columns:
                    raise ValueError(
                        f"Predefined latent samples for node '{node}' not found in dataframe columns. "
                        f"Must be named '{predefinded_sample_name}'.")
                    
                predefinded_sample = predefined_latent_samples_df[predefinded_sample_name].values
                if verbose or debug:
                    print(f'[INFO] Using predefined latents samples for node {node} from dataframe column: {predefinded_sample_name}')
                
                latent_sample = torch.tensor(predefinded_sample, dtype=torch.float32).to(device)
                
                ## IF not predefined latents are sampled from standard logistic distribution
            else:
                if verbose or debug:
                    print(f'[INFO] Sampling new latents for node {node} from standard logistic distribution')
                    
                latent_sample = torch.tensor(logistic.rvs(size=number_of_samples), dtype=torch.float32).to(device)
            
            
            ### load modelweights
            
            
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
                sampled=sample_continous_modelled_target(node,target_nodes_dict,sample_loader,tram_model,latent_sample,device=device, debug=debug,minmax_dict=minmax_dict)
                
            ###*************************************************** Ordinal Modelled Outcome ************************************************
            
            elif is_outcome_modelled_ordinal(node,target_nodes_dict):
                sampled=sample_ordinal_modelled_target(sample_loader,tram_model,device=device, debug=debug)
            
            else:
                raise ValueError(f"Unsupported data_type '{target_nodes_dict[node]['data_type']}' for node '{node}' in sampling.")
                
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