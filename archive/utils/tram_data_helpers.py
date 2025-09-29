
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

from utils import logger   # global logger


def merge_outputs(dict_list, skip_nan=True, **kwargs):
    if "debug" in kwargs or "verbose" in kwargs:
        logger.debug("merge_outputs: Ignoring legacy 'debug'/'verbose' arguments — controlled by global logger.")

    int_outs, shift_outs = [], []
    skipped_count = 0

    for d in dict_list:
        int_tensor = d["int_out"]
        shift_tensor = d["shift_out"][0] if isinstance(d["shift_out"], list) else d["shift_out"]

        if skip_nan and torch.isnan(int_tensor).all():
            skipped_count += 1
            continue

        int_outs.append(int_tensor)
        if shift_tensor is not None:
            shift_outs.append(shift_tensor)

    if skipped_count > 0:
        logger.warning(f"{skipped_count} entries with all-NaN 'int_out' were skipped.")

    merged = {
        "int_out": torch.cat(int_outs, dim=0) if int_outs else None,
        "shift_out": torch.cat(shift_outs, dim=0) if shift_outs else None,
    }
    return merged


def delete_all_samplings(conf_dict, EXPERIMENT_DIR, **kwargs):
    if "debug" in kwargs or "verbose" in kwargs:
        logger.debug("delete_all_samplings: Ignoring legacy 'debug'/'verbose' arguments — controlled by global logger.")

    for node in conf_dict:
        NODE_DIR = os.path.join(EXPERIMENT_DIR, f"{node}")
        SAMPLING_DIR = os.path.join(NODE_DIR, "sampling")

        if os.path.exists(SAMPLING_DIR):
            shutil.rmtree(SAMPLING_DIR)
            logger.info(f"Deleted directory: {SAMPLING_DIR}")
        else:
            logger.warning(f"Directory does not exist: {SAMPLING_DIR}")


def show_hdag_for_single_source_node_continous(node,
                                               configuration_dict,
                                               EXPERIMENT_DIR,
                                               device,xmin_plot=-5,
                                               xmax_plot=5,
                                               **kwargs):
    
    
        if "debug" in kwargs or "verbose" in kwargs:
            logger.debug("show_hdag_for_single_source_node_continous: Ignoring legacy 'debug'/'verbose' arguments — controlled by global logger.")

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
                                            return_y=False)

        
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

        logger.info("source node, Defaults to SI and 1 as inputs")
            
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




def show_hdag_for_source_nodes(configuration_dict, EXPERIMENT_DIR, device, xmin_plot=-5, xmax_plot=5, **kwargs):
    if "debug" in kwargs or "verbose" in kwargs:
        logger.debug("show_hdag_for_source_nodes: Ignoring legacy args — controlled by global logger.")

    target_nodes = configuration_dict["nodes"]

    for node in target_nodes:
        logger.info(f"Inspect TRAFO Node: {node}")

        if target_nodes[node]["node_type"] != "source":
            logger.info("Skipped: h depends on parents and varies per instance.")
            continue

        if is_outcome_modelled_continous(node, target_nodes):
            return show_hdag_for_single_source_node_continous(
                node=node,
                configuration_dict=configuration_dict,
                EXPERIMENT_DIR=EXPERIMENT_DIR,
                device=device,
                xmin_plot=xmin_plot,
                xmax_plot=xmax_plot,
            )

        if is_outcome_modelled_ordinal(node, target_nodes):
            logger.warning("Not implemented yet for ordinal (nominally encoded).")


def inspect_trafo_standart_logistic(configuration_dict, EXPERIMENT_DIR, train_df, val_df, device, **kwargs):
    if "verbose" in kwargs or "debug" in kwargs:
        logger.debug("inspect_trafo_standart_logistic: Ignoring legacy args — controlled by global logger.")

    target_nodes = configuration_dict["nodes"]
    h_train_outputs = []
    h_val_outputs = []

    for node in target_nodes:
        logger.info(f"h(data) should be standard logistic: {node}")

        if is_outcome_modelled_ordinal(node, target_nodes):
            logger.warning("Not defined for ordinal target variables")
            continue

        h_train, h_val = inspect_single_standart_logistic(
            node,
            configuration_dict,
            EXPERIMENT_DIR,
            train_df,
            val_df,
            device,
            return_intercept_shift=True,
        )
        h_train_outputs.append(h_train)
        h_val_outputs.append(h_val)

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
    **kwargs,
):
    if "verbose" in kwargs or "debug" in kwargs:
        logger.debug("inspect_single_standart_logistic: Ignoring legacy args — controlled by global logger.")

    target_nodes = configuration_dict["nodes"]
    NODE_DIR = os.path.join(EXPERIMENT_DIR, f"{node}")

    model_path = os.path.join(NODE_DIR, "best_model.pt")
    tram_model = get_fully_specified_tram_model(node, configuration_dict, set_initial_weights=False)
    tram_model = tram_model.to(device)
    tram_model.load_state_dict(torch.load(model_path, map_location=device))
    tram_model.eval()
    logger.info(f"Loaded TRAM model for node '{node}'")

    train_loader, val_loader = get_dataloader(
        node,
        target_nodes,
        train_df,
        val_df,
        batch_size=4112,
        return_intercept_shift=return_intercept_shift,
    )
    logger.info(f"Created dataloaders for node '{node}'")

    min_vals = torch.tensor(target_nodes[node]["min"], dtype=torch.float32, device=device)
    max_vals = torch.tensor(target_nodes[node]["max"], dtype=torch.float32, device=device)
    min_max = torch.stack([min_vals, max_vals], dim=0)

    h_train_list, h_val_list = [], []
    with torch.no_grad():
        for (int_input, shift_list), y in train_loader:
            int_input = int_input.to(device)
            shift_list = [s.to(device) for s in shift_list]
            y = y.to(device)
            y_pred = tram_model(int_input=int_input, shift_input=shift_list)
            h_train, _ = contram_nll(y_pred, y, min_max=min_max, return_h=True)
            h_train_list.extend(h_train.cpu().numpy())

        for (int_input, shift_list), y in val_loader:
            int_input = int_input.to(device)
            shift_list = [s.to(device) for s in shift_list]
            y = y.to(device)
            y_pred = tram_model(int_input=int_input, shift_input=shift_list)
            h_val, _ = contram_nll(y_pred, y, min_max=min_max, return_h=True)
            h_val_list.extend(h_val.cpu().numpy())

    h_train_array = np.array(h_train_list)
    h_val_array = np.array(h_val_list)

    fig, axs = plt.subplots(1, 4, figsize=(22, 5))
    axs[0].hist(h_train_array, bins=50)
    axs[0].set_title(f"Train Histogram ({node})")
    probplot(h_train_array, dist="logistic", plot=axs[1])
    add_r_style_confidence_bands(axs[1], h_train_array)
    axs[2].hist(h_val_array, bins=50)
    axs[2].set_title(f"Val Histogram ({node})")
    probplot(h_val_array, dist="logistic", plot=axs[3])
    add_r_style_confidence_bands(axs[3], h_val_array)

    plt.suptitle(f"Distribution Diagnostics for Node: {node}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    return h_train_array, h_val_array


def add_r_style_confidence_bands(ax, sample, dist=logistic, confidence=0.95, simulations=1000):
    n = len(sample)
    quantiles = np.linspace(0, 1, n, endpoint=False) + 0.5 / n
    theo_q = dist.ppf(quantiles)

    sim_data = dist.rvs(size=(simulations, n))
    sim_order_stats = np.sort(sim_data, axis=1)

    lower = np.percentile(sim_order_stats, 100 * (1 - confidence) / 2, axis=0)
    upper = np.percentile(sim_order_stats, 100 * (1 + confidence) / 2, axis=0)

    sample_sorted = np.sort(sample)
    ax.plot(theo_q, sample_sorted, linestyle="None", marker="o", markersize=3, alpha=0.6)
    ax.plot(theo_q, theo_q, "b--", label="y = x")
    ax.fill_between(theo_q, lower, upper, color="gray", alpha=0.3, label=f"{int(confidence*100)}% CI")
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
            logger.warning(f"Skip {node}: {sample_path} not found.")
            continue

        try:
            sampled = torch.load(sample_path, map_location=device).cpu().numpy()
        except Exception as e:
            logger.error(f"Could not load {sample_path}: {e}")
            continue

        sampled = sampled[np.isfinite(sampled)]

        if node not in df.columns:
            logger.warning(f"Skip {node}: column not found in DataFrame.")
            continue

        true_vals = df[node].dropna().values
        true_vals = true_vals[np.isfinite(true_vals)]

        if sampled.size == 0 or true_vals.size == 0:
            logger.warning(f"Skip {node}: empty array after NaN/Inf removal.")
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

            axs[1].axis("off")

        else:
            # Fallback: categorical
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


from utils.loss_ordinal import get_pdf_ordinal, get_cdf_ordinal
from utils import logger

def create_df_from_sampled(node, target_nodes_dict, num_samples, EXPERIMENT_DIR, **kwargs):
    """
    Create a DataFrame of sampled values for a given node's parents.
    Falls back to a dummy variable if no sampled data is found.
    """
    if "debug" in kwargs or "verbose" in kwargs:
        logger.debug("create_df_from_sampled: Ignoring legacy 'debug'/'verbose' arguments — controlled by global logger.")

    sampling_dict = {}
    logger.debug("create_df_from_sampled: initializing sampling dictionary with dummy variable")

    # Add dummy variable
    sampling_dict["DUMMY"] = torch.zeros(num_samples)

    # Try loading sampled values for each parent
    for parent in target_nodes_dict[node].get('parents', []):
        path = os.path.join(EXPERIMENT_DIR, parent, "sampling", "sampled.pt")
        if os.path.exists(path):
            logger.debug(f"create_df_from_sampled: loading sampled data for parent '{parent}' from {path}")
            sampling_dict[parent] = torch.load(path, map_location="cpu")
        else:
            logger.debug(f"create_df_from_sampled: no sampled data found for parent '{parent}' at {path}")

    # Remove dummy if we have real variables
    if len(sampling_dict) > 1:
        logger.debug("create_df_from_sampled: removing dummy variable since real variables are present")
        sampling_dict.pop("DUMMY")
    else:
        logger.debug("create_df_from_sampled: only dummy variable present")

    # Move all tensors to CPU before creating the DataFrame
    sampling_dict_cpu = {k: v.cpu().numpy() for k, v in sampling_dict.items()}
    logger.debug("create_df_from_sampled: creating DataFrame from variables: %s", list(sampling_dict_cpu.keys()))
    for k, v in sampling_dict_cpu.items():
        logger.debug("create_df_from_sampled: %s shape: %s", k, v.shape)

    sampling_df = pd.DataFrame(sampling_dict_cpu)
    logger.debug("create_df_from_sampled: final DataFrame shape: %s", sampling_df.shape)

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
    
from utils.loss_ordinal import get_pdf_ordinal, get_cdf_ordinal
from utils import logger

def sample_ordinal_modelled_target(sample_loader, tram_model, device, **kwargs):
    """
    Sample targets for nodes with ordinal outcomes using the TRAM model.

    Parameters
    ----------
    sample_loader : DataLoader
        DataLoader providing (int_input, shift_list) pairs.
    tram_model : torch.nn.Module
        Trained TRAM model.
    device : torch.device
        Device to run inference on.
    kwargs : ignored
        For backward compatibility (e.g., debug, verbose).
    """
    if "debug" in kwargs or "verbose" in kwargs:
        logger.debug("sample_ordinal_modelled_target: Ignoring legacy 'debug'/'verbose' arguments — use logger instead.")

    all_outputs = []
    tram_model.eval()
    with torch.no_grad():
        for (int_input, shift_list) in sample_loader:
            int_input = int_input.to(device)
            shift_list = [s.to(device) for s in shift_list]

            model_outputs = tram_model(int_input=int_input, shift_input=shift_list)
            all_outputs.append(model_outputs)

            logger.debug("Batch model_outputs keys: %s", list(model_outputs.keys()))
            logger.debug("int_out shape: %s", model_outputs['int_out'].shape)
            if model_outputs['shift_out'] is not None:
                logger.debug("shift_out shapes: %s", [s.shape for s in model_outputs['shift_out']])

    # Concatenate all 'int_out' and 'shift_out' elements across batches
    int_out_all = torch.cat([out['int_out'] for out in all_outputs], dim=0)

    # If shift_out is present, concatenate across batches
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

    logger.debug("Final sampled shape: %s", sampled.shape)
    logger.debug("Sampled labels (first 3): %s", sampled[:3])

    return sampled

from utils import logger

def sample_continous_modelled_target(node, target_nodes_dict, sample_loader, tram_model, latent_sample, device, **kwargs):
    """
    Sample targets for nodes with continuous outcomes using the TRAM model.

    Parameters
    ----------
    node : str
        Node name.
    target_nodes_dict : dict
        Dictionary of all target node metadata.
    sample_loader : DataLoader
        Provides (int_input, shift_list) pairs.
    tram_model : torch.nn.Module
        Trained TRAM model.
    latent_sample : torch.Tensor
        Pre-sampled latent variables (U's).
    device : torch.device
        Device for inference.
    kwargs : ignored
        For backward compatibility (e.g., debug, verbose).
    """
    if "debug" in kwargs or "verbose" in kwargs:
        logger.debug("sample_continous_modelled_target: Ignoring legacy 'debug'/'verbose' arguments — use logger instead.")

    number_of_samples = len(latent_sample)
    output_list = []

    tram_model.eval()
    with torch.no_grad():
        for (int_input, shift_list) in sample_loader:
            int_input = int_input.to(device)
            shift_list = [s.to(device) for s in shift_list]
            model_outputs = tram_model(int_input=int_input, shift_input=shift_list)
            output_list.append(model_outputs)

    if not output_list:
        raise RuntimeError("sample_continous_modelled_target: Model output list is empty. Check the sample_loader or model.")

    if target_nodes_dict[node]['node_type'] == 'source':
        logger.debug("Source node detected → defaults to SI and 1 as inputs")
        if 'int_out' not in output_list[0]:
            raise KeyError("Missing 'int_out' in model output for source node.")

        theta_single = output_list[0]['int_out'][0]
        theta_single = transform_intercepts_continous(theta_single)
        thetas_expanded = theta_single.repeat(number_of_samples, 1)
        shifts = torch.zeros(number_of_samples, device=device)

    else:
        logger.debug("Node has parents → using previously sampled data")
        y_pred = merge_outputs(output_list, skip_nan=True)

        if 'int_out' not in y_pred:
            raise KeyError("Missing 'int_out' in merged model output.")
        if 'shift_out' not in y_pred:
            raise KeyError("Missing 'shift_out' in merged model output.")

        thetas = y_pred['int_out']
        shifts = y_pred['shift_out']
        if shifts is None:
            logger.debug("shift_out was None → defaulting to zeros")
            shifts = torch.zeros(number_of_samples, device=device)

        thetas_expanded = transform_intercepts_continous(thetas).squeeze()
        shifts = shifts.squeeze()

    # Validate shapes
    if thetas_expanded.shape[0] != number_of_samples:
        raise ValueError(
            f"Mismatch in sample count: thetas_expanded has shape {thetas_expanded.shape}, expected {number_of_samples} rows."
        )

    logger.debug("Beginning root finding")
    logger.debug("    thetas_expanded shape: %s", thetas_expanded.shape)
    logger.debug("    shifts shape: %s", shifts.shape)
    logger.debug("    latent_sample shape: %s", latent_sample.shape)

    # Root bounds
    low = torch.full((number_of_samples,), -1e5, device=device)
    high = torch.full((number_of_samples,), 1e5, device=device)

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
        max_iter=10_000,
        tol=1e-12
    )

    if sampled is None or torch.isnan(sampled).any():
        raise RuntimeError("Root finding failed: returned None or contains NaNs.")

    logger.debug("Root finding complete. Sampled shape: %s", sampled.shape)

    return sampled

def check_sampled_and_latents(NODE_DIR, **kwargs):
    """
    Check whether both 'sampled.pt' and 'latents.pt' exist in the sampling directory.

    Parameters
    ----------
    NODE_DIR : str
        Path to the node directory.
    kwargs : ignored
        For backward compatibility (e.g., debug, verbose).
    """
    if "debug" in kwargs or "verbose" in kwargs:
        logger.debug("check_sampled_and_latents: Ignoring legacy 'debug'/'verbose' arguments — use logger instead.")

    sampling_dir = os.path.join(NODE_DIR, "sampling")
    root_path = os.path.join(sampling_dir, "sampled.pt")
    latents_path = os.path.join(sampling_dir, "latents.pt")

    if not os.path.exists(root_path):
        raise FileNotFoundError(f"'sampled.pt' not found in {sampling_dir}")
    if not os.path.exists(latents_path):
        raise FileNotFoundError(f"'latents.pt' not found in {sampling_dir}")

    logger.debug("Found 'sampled.pt' in %s", sampling_dir)
    logger.debug("Found 'latents.pt' in %s", sampling_dir)

    return True


def provide_latents_for_input_data(
    node,
    configuration_dict,
    EXPERIMENT_DIR,
    data_loader,
    base_df,
    **kwargs,
):
    """
    Compute latent representations for each observation in base_df
    and return a DataFrame with columns [node, "_U"] aligned with base_df.

    Parameters
    ----------
    node : str
        Node name.
    configuration_dict : dict
        Experiment configuration dictionary.
    EXPERIMENT_DIR : str
        Base experiment directory.
    data_loader : DataLoader
        Torch DataLoader providing (int_input, shift_list), y pairs.
    base_df : pandas.DataFrame
        DataFrame containing the true values to align with.
    kwargs : ignored
        For backward compatibility (e.g., debug, verbose).
    """
    if "debug" in kwargs or "verbose" in kwargs:
        logger.debug("provide_latents_for_input_data: Ignoring legacy 'debug'/'verbose' arguments — use logger instead.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_nodes = configuration_dict["nodes"]
    if is_outcome_modelled_ordinal(node, target_nodes):
        raise ValueError("Not yet defined for ordinal target variables")

    # 0. Paths
    NODE_DIR = os.path.join(EXPERIMENT_DIR, f"{node}")

    # 1. Load model 
    model_path = os.path.join(NODE_DIR, "best_model.pt")
    tram_model = get_fully_specified_tram_model(
        node, configuration_dict, set_initial_weights=False
    )
    tram_model = tram_model.to(device)
    tram_model.load_state_dict(torch.load(model_path, map_location=device))
    tram_model.eval()

    # 2. Forward Pass
    min_vals = torch.tensor(target_nodes[node]["min"], dtype=torch.float32, device=device)
    max_vals = torch.tensor(target_nodes[node]["max"], dtype=torch.float32, device=device)
    min_max = torch.stack([min_vals, max_vals], dim=0)

    latents_list = []
    with torch.no_grad():
        for (int_input, shift_list), y in data_loader:
            int_input = int_input.to(device)
            shift_list = [s.to(device) for s in shift_list]
            y = y.to(device)
            y_pred = tram_model(int_input=int_input, shift_input=shift_list)

            latents, _ = contram_nll(y_pred, y, min_max=min_max, return_h=True)
            latents_list.extend(latents.cpu().numpy())

    # Turn into DataFrame aligned with base_df
    latents_df = pd.DataFrame(
        {
            node: base_df[node].values,
            f"{node}_U": latents_list,
        },
        index=base_df.index,
    )

    return latents_df


def create_latent_df_for_full_dag(configuration_dict, EXPERIMENT_DIR, df, **kwargs):
    """
    Compute latent representations for all nodes in the DAG
    and return a combined DataFrame with one `_U` column per node.

    Parameters
    ----------
    configuration_dict : dict
        Experiment configuration dictionary.
    EXPERIMENT_DIR : str
        Base experiment directory.
    df : pandas.DataFrame
        Input DataFrame with true values.
    kwargs : ignored
        For backward compatibility (e.g., debug, verbose).
    """
    if "debug" in kwargs or "verbose" in kwargs:
        logger.debug("create_latent_df_for_full_dag: Ignoring legacy 'debug'/'verbose' arguments — use logger instead.")

    all_latents_dfs = []

    for node in configuration_dict["nodes"]:
        # Skip ordinal outcomes if not supported
        if is_outcome_modelled_ordinal(node, configuration_dict["nodes"]):
            logger.info(f"Skipping node '{node}' (ordinal targets not yet supported).")
            continue

        logger.info(f"Processing node '{node}'")

        # Build dataset and dataloader for this node
        node_dataset = GenericDataset(
            df,
            target_col=node,
            target_nodes=configuration_dict["nodes"],
        )
        node_loader = DataLoader(
            node_dataset,
            batch_size=4096,   # can be tuned
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Compute latents
        node_latents_df = provide_latents_for_input_data(
            node=node,
            configuration_dict=configuration_dict,
            EXPERIMENT_DIR=EXPERIMENT_DIR,
            data_loader=node_loader,
            base_df=df,
        )

        all_latents_dfs.append(node_latents_df)

    # Concatenate all node-specific latent dataframes
    all_latents_df = pd.concat(all_latents_dfs, axis=1)

    # Remove duplicate index columns (if multiple nodes included their own target col)
    all_latents_df = all_latents_df.loc[:, ~all_latents_df.columns.duplicated()]

    logger.info(f"Final latent DataFrame shape: {all_latents_df.shape}")
    return all_latents_df


def sample_full_dag(
    configuration_dict,
    EXPERIMENT_DIR,
    device,
    do_interventions={},
    predefined_latent_samples_df: pd.DataFrame = None,
    number_of_samples: int = 10_000,
    batch_size: int = 32,
    delete_all_previously_sampled: bool = True,
    **kwargs,
):
    """
    Sample values for all nodes in a DAG given trained TRAM models, respecting
    parental ordering. Supports both generative sampling (new U's) and
    reconstruction sampling (predefined U's), as well as node-level interventions.
    """
    if "debug" in kwargs or "verbose" in kwargs:
        logger.debug("sample_full_dag: Ignoring legacy 'debug'/'verbose' args — use logger instead.")

    if predefined_latent_samples_df is not None:
        logger.info(
            f"Using predefined latent samples from dataframe → n_samples set to {len(predefined_latent_samples_df)}"
        )
        number_of_samples = len(predefined_latent_samples_df)

    target_nodes_dict = configuration_dict["nodes"]

    sampled_by_node = {}
    latents_by_node = {}

    if delete_all_previously_sampled:
        logger.info("Deleting all previously sampled data.")
        delete_all_samplings(target_nodes_dict, EXPERIMENT_DIR)

    max_iterations, iteration = 200, 0
    processed_nodes = []

    while set(processed_nodes) != set(target_nodes_dict.keys()):
        iteration += 1
        if iteration > max_iterations:
            raise RuntimeError("Too many iterations in sampling loop, possible infinite loop.")

        for node in target_nodes_dict:
            if node in processed_nodes:
                logger.debug(f"Node '{node}' already processed.")
                continue

            logger.info(f"---- Sampling Node: {node} ----")

            NODE_DIR = os.path.join(EXPERIMENT_DIR, f"{node}")
            SAMPLING_DIR = os.path.join(NODE_DIR, "sampling")
            os.makedirs(SAMPLING_DIR, exist_ok=True)
            SAMPLED_PATH = os.path.join(SAMPLING_DIR, "sampled.pt")
            LATENTS_PATH = os.path.join(SAMPLING_DIR, "latents.pt")

            try:
                if check_sampled_and_latents(NODE_DIR, debug=False):
                    processed_nodes.append(node)
                    continue
            except FileNotFoundError:
                pass

            skipping_node = False
            if target_nodes_dict[node]["node_type"] != "source":
                for parent in target_nodes_dict[node]["parents"]:
                    parent_dir = os.path.join(EXPERIMENT_DIR, parent)
                    try:
                        check_sampled_and_latents(parent_dir, debug=False)
                    except FileNotFoundError:
                        skipping_node = True
                        logger.debug(f"Skipping {node}: parent '{parent}' not sampled yet.")
                        break
            if skipping_node:
                continue

            # Intervention branch
            if do_interventions and node in do_interventions:
                intervention_value = do_interventions[node]
                logger.info(f"Applying intervention for node '{node}' with value {intervention_value}")

                intervention_vals = torch.full((number_of_samples,), intervention_value)
                torch.save(intervention_vals, SAMPLED_PATH)

                dummy_latents = torch.full((number_of_samples,), float("nan"))
                torch.save(dummy_latents, LATENTS_PATH)

                sampled_by_node[node] = intervention_vals
                latents_by_node[node] = dummy_latents

                processed_nodes.append(node)
                logger.info(f"Interventional data for node '{node}' saved.")
                continue

            # Observational branch
            if predefined_latent_samples_df is not None:
                latent_col = f"{node}_U"
                if latent_col not in predefined_latent_samples_df.columns:
                    raise ValueError(
                        f"Predefined latent samples for node '{node}' not found in DataFrame. "
                        f"Expected column '{latent_col}'."
                    )
                logger.info(f"Using predefined latent samples for node '{node}' from column: {latent_col}")
                predef_values = predefined_latent_samples_df[latent_col].values
                latent_sample = torch.tensor(predef_values, dtype=torch.float32).to(device)
            else:
                logger.info(f"Sampling new latents for node '{node}' from standard logistic distribution")
                latent_sample = torch.tensor(
                    logistic.rvs(size=number_of_samples), dtype=torch.float32
                ).to(device)

            MODEL_PATH = os.path.join(NODE_DIR, "best_model.pt")
            tram_model = get_fully_specified_tram_model(node, configuration_dict, set_initial_weights=False).to(device)
            tram_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

            sampled_df = create_df_from_sampled(node, target_nodes_dict, number_of_samples, EXPERIMENT_DIR)
            sample_dataset = GenericDataset(
                sampled_df,
                target_col=node,
                target_nodes=target_nodes_dict,
                return_intercept_shift=True,
                return_y=False,
            )
            sample_loader = DataLoader(sample_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

            if is_outcome_modelled_continous(node, target_nodes_dict):
                sampled = sample_continous_modelled_target(
                    node, target_nodes_dict, sample_loader, tram_model, latent_sample, device=device
                )
            elif is_outcome_modelled_ordinal(node, target_nodes_dict):
                sampled = sample_ordinal_modelled_target(sample_loader, tram_model, device=device)
            else:
                raise ValueError(
                    f"Unsupported data_type '{target_nodes_dict[node]['data_type']}' for node '{node}' in sampling."
                )

            if torch.isnan(sampled).any():
                logger.warning(f"NaNs detected in sampled output for node '{node}'")

            torch.save(sampled, SAMPLED_PATH)
            torch.save(latent_sample, LATENTS_PATH)

            sampled_by_node[node] = sampled.detach().cpu()
            latents_by_node[node] = latent_sample.detach().cpu()

            logger.info(f"Completed sampling for node '{node}'")
            processed_nodes.append(node)

    logger.info("DAG sampling completed successfully for all nodes.")
    return sampled_by_node, latents_by_node