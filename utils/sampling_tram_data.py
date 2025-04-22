import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import warnings
import os
import shutil

from utils.tram_model_helpers import ordered_parents


class SamplingDataset(Dataset):
    def __init__(self, node,EXPERIMENT_DIR,number_of_samples=100, conf_dict=None, transform=None):
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
            tensor = load_roots(PARENT_DIR)  # expected shape: (num_samples, feature_dim)
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
def check_roots_and_latents(NODE_DIR,verbose=True):
    root_path = os.path.join(NODE_DIR, 'sampling',"roots.pt")
    latents_path=os.path.join(NODE_DIR, 'sampling', "latents.pt")
    if os.path.exists(root_path) and os.path.exists(latents_path):
        return True
    else:
        if verbose:
            print(f'Root or latent files not found in {os.path.join(NODE_DIR,"sampling")}')
        return False
    

def load_roots(NODE_DIR):
    root_path = os.path.join(NODE_DIR, 'sampling',"roots.pt")
    root=torch.load(root_path)
    return root

def load_latents(NODE_DIR):
    latents_path=os.path.join(NODE_DIR, 'sampling', "latents.pt")
    latents=torch.load(latents_path)
    return latents

def load_roots_and_latents(NODE_DIR):
    root=load_roots(NODE_DIR)
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