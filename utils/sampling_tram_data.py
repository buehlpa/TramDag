import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import warnings
import os
import shutil

from utils.tram_model_helpers import ordered_parents
from utils.tram_model_helpers import *           
from utils.continous import *   
from utils.tram_data import get_dataloader

class SamplingDataset(Dataset):
    def __init__(self, node,EXPERIMENT_DIR,number_of_samples=100,rootfinder='bisection', conf_dict=None, transform=None):
        """
        Args:
            node (str): Name of the target node.
            conf_dict (dict): Mapping variable names to their types: "cont", "other", "ord".
            transform (callable, optional): Optional transform to apply to image data.
        """
        self.EXPERIMENT_DIR=EXPERIMENT_DIR
        self.node = node
        self.number_of_samples=number_of_samples
        self.conf_dict = conf_dict
        self.transform = transform
        self.rootfinder=rootfinder
        self.variables = None if conf_dict is None else self._ordered_keys()
        self.datatensors = self._get_sampled_parent_tensors()# shape: (num_parents, num_samples, dim)

    def _get_sampled_parent_tensors(self):
        ## loads parent roots if they are available 
        tensor_list = []
        if self.conf_dict is None or self.conf_dict[self.node]['node_type'] == 'source':
            tensor_list.append(torch.ones(self.number_of_samples) * 1.0)
            return tensor_list 
        else:        
            parents_dataype_dict, _, _ = ordered_parents(self.node, self.conf_dict)
        
        for parent_pair in parents_dataype_dict:
            # print(parent_pair)
            PARENT_DIR = os.path.join(self.EXPERIMENT_DIR, f'{parent_pair}')
            tensor = load_roots(PARENT_DIR,rootfinder=self.rootfinder)  # expected shape: (num_samples, feature_dim)
            tensor_list.append(torch.tensor(tensor))  # ensure tensor type
            # print(tensor_list)
        return tensor_list  # list of tensors, each (num_samples, dim)

    def _ordered_keys(self):
        parents_dataype_dict, _, _ = ordered_parents(self.node, self.conf_dict)
        return list(parents_dataype_dict.keys())

    def __len__(self):        
        return self.datatensors[0].shape[0]  # assuming same num_samples for all

    def __getitem__(self, idx):
        x_data = []

        if self.conf_dict is None or self.conf_dict[self.node]['node_type'] == 'source':
            return (torch.tensor([1.0], dtype=torch.float32),)

        for i, var in enumerate(self.variables):
            if self.conf_dict[var]['data_type'] == "cont":
                val = self.datatensors[i][idx]
                x_data.append(val.unsqueeze(0))  # ensure shape (1,)
            elif self.conf_dict[var]['data_type'] == "ord":
                val = self.datatensors[i][idx]
                x_data.append(val.unsqueeze(0).long())  # ensure shape (1,) and long
            elif self.conf_dict[var]['data_type'] == "other":
                img_path = self.datatensors[i][idx]
                image = Image.open(img_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                x_data.append(image)
        return tuple(x_data)
    
    
# helpers
def check_roots_and_latents(NODE_DIR,rootfinder='bisection',verbose=True):
    root_path = os.path.join(NODE_DIR, 'sampling',f"roots_{rootfinder}.pt")
    latents_path=os.path.join(NODE_DIR, 'sampling', "latents.pt")
    if os.path.exists(root_path) and os.path.exists(latents_path):
        return True
    else:
        if verbose:
            print(f'Root or latent files not found in {os.path.join(NODE_DIR,"sampling")}')
        return False
    

def load_roots(NODE_DIR,rootfinder='bisection'):
    root_path = os.path.join(NODE_DIR, 'sampling',f"roots_{rootfinder}.pt")
    root=torch.load(root_path)
    return root

def load_latents(NODE_DIR):
    latents_path=os.path.join(NODE_DIR, 'sampling', "latents.pt")
    latents=torch.load(latents_path)
    return latents

def load_roots_and_latents(NODE_DIR,rootfinder='bisection'):
    root=load_roots(NODE_DIR,whirootfinder=rootfinder)
    latents=load_latents(NODE_DIR)
    return root, latents




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
            
            

def show_hdag_for_source_nodes(conf_dict,EXPERIMENT_DIR,device,xmin_plot=-5,xmax_plot=5):
    verbose=False
    n=1000
    for node in conf_dict:
        print(f'\n----*----------*-------------*--------Inspect TRAFO Node: {node} ------------*-----------------*-------------------*--')
        if (conf_dict[node]['node_type'] != 'source'):
            print("skipped.. since h does depend on parents and is different for every instance")
            continue
        
        #### 0.  paths
        NODE_DIR = os.path.join(EXPERIMENT_DIR, f'{node}')
        
        ##### 1.  load model 
        model_path = os.path.join(NODE_DIR, "best_model.pt")
        tram_model = get_fully_specified_tram_model(node, conf_dict, verbose=verbose)
        tram_model = tram_model.to(device)
        tram_model.load_state_dict(torch.load(model_path))
        
        #### 2. Sampling Dataloader
        dataset = SamplingDataset(node=node,EXPERIMENT_DIR=EXPERIMENT_DIR,number_of_samples=n, conf_dict=conf_dict, transform=None)
        sample_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
        output_list = []
        with torch.no_grad():
            for x in tqdm(sample_loader, desc=f"h() for  {node}"):
                x = [xi.to(device) for xi in x]
                int_input, shift_list = preprocess_inputs(x, device=device)
                model_outputs = tram_model(int_input=int_input, shift_input=shift_list)
                output_list.append(model_outputs)
                break
        if verbose:
            print("source node, Defaults to SI and 1 as inputs")
            
        theta_single =     output_list[0]['int_out'][0]  # Shape: (20,)
        theta_single=transform_intercepts_continous(theta_single)
        thetas_expanded = theta_single.repeat(n, 1).to(device)  # Shape: (n, 20)
        
        targets2 = torch.linspace(xmin_plot, xmax_plot, steps=n).to(device)  # 1000 points from 0 to 1
        
        min_vals = torch.tensor(conf_dict[node]['min'], dtype=torch.float32).to(device)
        max_vals = torch.tensor(conf_dict[node]['max'], dtype=torch.float32).to(device)
        min_max = torch.stack([min_vals, max_vals], dim=0)
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
        
        
        
def inspect_trafo_standart_logistic(conf_dict, EXPERIMENT_DIR, train_df, val_df, device, verbose=False):
    batch_size = 4112
    for node in conf_dict:
        print(f'\n----*----------*-------------*--------h(data) should be standard logistic: {node} ------------*-----------------*-------------------*--')

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

        with torch.no_grad():
            for x, y in tqdm(train_loader, desc=f"Train loader ({node})", total=len(train_loader)):
                y = y.to(device)
                int_input, shift_list = preprocess_inputs(x, device=device)
                y_pred = tram_model(int_input=int_input, shift_input=shift_list)
                h_train, _ = contram_nll(y_pred, y, min_max=min_max, return_h=True)
                h_train_list.extend(h_train.cpu().numpy())

            for x, y in tqdm(val_loader, desc=f"Val loader ({node})", total=len(val_loader)):
                y = y.to(device)
                int_input, shift_list = preprocess_inputs(x, device=device)
                y_pred = tram_model(int_input=int_input, shift_input=shift_list)
                h_val, _ = contram_nll(y_pred, y, min_max=min_max, return_h=True)
                h_val_list.extend(h_val.cpu().numpy())

        h_train_array = np.array(h_train_list)
        h_val_array = np.array(h_val_list)

        # Plotting
        fig, axs = plt.subplots(1, 4, figsize=(22, 5))

        # Train Histogram
        axs[0].hist(h_train_array, bins=50)
        axs[0].set_title(f'Train Histogram ({node})')

        # Train QQ Plot with R-style Confidence Bands
        probplot(h_train_array, dist="logistic", plot=axs[1])
        add_r_style_confidence_bands(axs[1], h_train_array)

        # Validation Histogram
        axs[2].hist(h_val_array, bins=50)
        axs[2].set_title(f'Val Histogram ({node})')

        # Validation QQ Plot with R-style Confidence Bands
        probplot(h_val_array, dist="logistic", plot=axs[3])
        add_r_style_confidence_bands(axs[3], h_val_array)

        plt.suptitle(f'Distribution Diagnostics for Node: {node}', fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


def add_r_style_confidence_bands(ax, sample, dist=logistic, confidence=0.95):
    """
    Adds confidence bands to a QQ plot around the theoretical quantiles.
    These are based on order statistics and the binomial distribution.
    """
    n = len(sample)
    quantiles = np.linspace(0.001, 0.999, n)
    theo_q = dist.ppf(quantiles)
    sample_sorted = np.sort(sample)

    # Compute confidence interval for each order statistic
    alpha = 1 - confidence
    lower_ci = dist.ppf(np.maximum(quantiles - 1.96 * np.sqrt(quantiles * (1 - quantiles) / n), 0.001))
    upper_ci = dist.ppf(np.minimum(quantiles + 1.96 * np.sqrt(quantiles * (1 - quantiles) / n), 0.999))

    # Plot the confidence band
    ax.fill_between(theo_q, lower_ci, upper_ci, color='gray', alpha=0.3, label='95% CI')
    ax.legend()

        
        
def show_samples_vs_true(df,conf_dict,EXPERIMENT_DIR,rootfinder='chandrupatla'):
    
    for node in conf_dict.keys():
        root_np = torch.load(os.path.join(EXPERIMENT_DIR,f'{node}/sampling/roots_{rootfinder}.pt')).cpu().numpy()
        
        true_values = df[node].dropna().values  # Drop NaNs for fair comparison
        
        # Sort both arrays for QQ plot
        sorted_true = np.sort(true_values)
        sorted_roots = np.sort(root_np)
        
        min_len = min(len(sorted_true), len(sorted_roots))
        sorted_true = sorted_true[:min_len]
        sorted_roots = sorted_roots[:min_len]

        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        axs[0].hist(true_values, bins=100, alpha=0.5, label=f'True Values {node}', color='blue', density=True)
        axs[0].hist(root_np, bins=100, alpha=0.5, label='Estimated Roots', color='red', density=True)
        axs[0].set_xlabel("Value")
        axs[0].set_ylabel("Density")
        axs[0].set_title(f"Overlay of True vs Estimated Values for {node}")
        axs[0].legend()
        axs[0].grid(True)

        # QQ plot
        axs[1].plot(sorted_true, sorted_true, color='blue', label='Observed Quantiles (True)')
        axs[1].scatter(sorted_true, sorted_roots, color='red', s=10, alpha=0.6, label='Estimated Roots')
        axs[1].set_xlabel("Observed Quantiles")
        axs[1].set_ylabel("Estimated Quantiles")
        axs[1].set_title(f"QQ Plot for {node}")
        axs[1].grid(True)
        axs[1].legend()

        plt.tight_layout()
        plt.show()

def show_latent_sampling(EXPERIMENT_DIR,conf_dict):
    for node in conf_dict.keys():
        latents_path = os.path.join(EXPERIMENT_DIR,f'{node}/sampling/latents.pt')
        latents = torch.load(latents_path).cpu().numpy()
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
                
                
                sm = tram_model.nn_shift[0]
                print("shift_input[0].shape →", shift_list[0].shape)
                print("fc1.out_features    →", sm.fc1.out_features)
                print("bn1.num_features    →", sm.bn1.num_features)
                
                im = tram_model.nn_int
                print("int_input.shape     →", int_input.shape)
                print("int fc1.out_features→", im.fc1.out_features)
                print("int bn1.num_features→", im.bn1.num_features)
                
                
                
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