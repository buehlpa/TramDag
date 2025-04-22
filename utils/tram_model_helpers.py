from utils.tram_models import *
from utils.graph import *
import os
import re
from collections import OrderedDict
import json
import time
from functools import wraps
import shutil

# Time decorator
def timeit(name=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            label = f"{name or func.__name__}"
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start
            print(f"[ {label}] completed in {duration:.4f} seconds")
            return result
        return wrapper
    return decorator








def get_fully_specified_tram_model(node,conf_dict,verbose=True):
    

    ### iF node is a source -> no deep nn is needed
    if conf_dict[node]['node_type'] == 'source':
        nn_int = SimpleIntercept()
        tram_model = TramModel(nn_int, None)  
        if verbose:
            print('>>>>>>>>>>>>  source node --> only  modelled only  by si')
            print(tram_model)
        return tram_model
    
    
    ### if node is not a source node 
    else:
        # read terms and model names form the config
        
        _,terms_dict,model_names_dict=ordered_parents(node, conf_dict)
        
        #old
        # terms_dict=conf_dict[node]['transformation_terms_in_h()']
        # model_names_dict=conf_dict[node]['transformation_term_nn_models_in_h()']
        
        # Combine terms and model names and divide in intercept and shift terms
        model_dict=merge_transformation_dicts(terms_dict, model_names_dict)
        intercepts_dict = {k: v for k, v in model_dict.items() if "ci" in v['h_term'] or 'si' in v['h_term']}        
        shifts_dict = {k: v for k, v in model_dict.items() if "ci" not in v['h_term'] and  'si' not in v['h_term']}        
        
        # make sure that nns are correctly defined afterwards
        nn_int, nn_shifts_list = None, None
        
        # intercept term
        if not np.any(np.array([True for diction in intercepts_dict.values() if 'ci' in diction['h_term']]) == True):
            print('>>>>>>>>>>>> No ci detected --> intercept defaults to si') if verbose else None
            nn_int = SimpleIntercept()
        
        else:
            
            # intercept term -> model
            nn_int_name = list(intercepts_dict.items())[0][1]['class_name'] # TODO this doesnt work for multi inpout CI's
            nn_int = globals()[nn_int_name]()
        
        # shift term -> lsit of models         
        nn_shift_names=[v["class_name"] for v in shifts_dict.values() if "class_name" in v]
        nn_shifts_list = [globals()[name]() for name in nn_shift_names]
        
        # ontram model
        tram_model = TramModel(nn_int, nn_shifts_list)    
        
        print('>>> TRAM MODEL:\n',tram_model) if verbose else None
    return tram_model


def ordered_parents(node, conf_dict) -> dict:
    
    """
    Orders the transformation terms and their corresponding data types and nn models used for the models and the dataloader
    """
    
    order = ['ci', 'ciXX', 'cs', 'csXX', 'ls']

    # Extract dictionaries
    transformation_terms = conf_dict[node]['transformation_terms_in_h()']
    datatype_dict = conf_dict[node]['parents_datatype']
    nn_models_dict = conf_dict[node]['transformation_term_nn_models_in_h()']

    def get_sort_key(val):
        """Map each transformation value to a sorting index."""
        base = re.match(r'[a-zA-Z]+', val)
        digits = re.findall(r'\d+', val)
        base = base.group(0) if base else ''
        digits = int(digits[0]) if digits else -1

        if base in ['ci', 'cs'] and digits != -1:
            # For ciXX/csXX
            return (order.index(base + 'XX'), digits)
        elif base in order:
            return (order.index(base), -1)
        else:
            return (len(order), digits)  # unknown terms go last

    # Sort the items based on the transformation_terms
    sorted_keys = sorted(transformation_terms.keys(), key=lambda k: get_sort_key(transformation_terms[k]))

    # Create ordered dicts
    ordered_transformation_terms_in_h = OrderedDict((k, transformation_terms[k]) for k in sorted_keys)
    ordered_parents_datatype = OrderedDict((k, datatype_dict[k]) for k in sorted_keys)
    ordered_transformation_term_nn_models_in_h = OrderedDict((k, nn_models_dict[k]) for k in sorted_keys)

    return ordered_parents_datatype, ordered_transformation_terms_in_h, ordered_transformation_term_nn_models_in_h


def preprocess_inputs(x, device='cuda'):
    """
    Prepares model input by:
    - Accepting a tuple or list of tensors
    - Adding channel dim (unsqueeze)
    - Moving everything to the device once
    - Returning int_inputs and shift_list
    """
    # Unpack x if it's a tuple/list
    x = [xi.unsqueeze(1) for xi in x]  # shape: (B, 1, ...)
    x = [xi.to(device, non_blocking=True) for xi in x]  # single device transfer

    int_inputs = x[0]
    shift_list = x[1:] if len(x) > 1 else None

    return int_inputs, shift_list

# print training history
def load_history(node, experiment_dir):
    node_dir = os.path.join(experiment_dir, node)
    train_hist_path = os.path.join(node_dir, "train_loss_hist.json")
    val_hist_path = os.path.join(node_dir, "val_loss_hist.json")

    if os.path.exists(train_hist_path) and os.path.exists(val_hist_path):
        with open(train_hist_path, 'r') as f:
            train_loss_hist = json.load(f)
        with open(val_hist_path, 'r') as f:
            val_loss_hist = json.load(f)
        return train_loss_hist, val_loss_hist
    else:
        return None, None
    

