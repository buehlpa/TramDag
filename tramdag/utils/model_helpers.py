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


import os
import re
from collections import OrderedDict, defaultdict
import json
import time
from functools import wraps
from scipy.stats import logistic, kstest, anderson
from torch.utils.data import DataLoader


from ..models.tram_models import *
from .continous import contram_nll, inverse_transform_intercepts_continous,transform_intercepts_continous
from .ordinal import   ontram_nll,  inverse_transform_intercepts_ordinal,transform_intercepts_ordinal
from .configuration import *
from .r_subprocess import fit_r_model_subprocess



################################################ Model Construction #####################################

def get_fully_specified_tram_model(
    node: str,
    configuration_dict: dict,
    set_initial_weights: bool = False,
    initial_data: str = None,
    verbose: bool = True,
    debug: bool = False,
    device='auto',
    **kwargs) -> TramModel:
    """
    Construct and return a fully specified TramModel for a given node based on
    its configuration (ordinal or continuous outcome) and parent inputs.

    This function:
    - Analyzes the metadata of the target node and its parents.
    - Builds an intercept network (handling ordinal or continuous cases).
    - Builds shift networks for covariate-dependent terms.
    - Optionally initializes the intercept network with R-based intercept
      estimates (via COLR/POLR).
    - Returns the combined TramModel consisting of intercept and shift parts.

    Parameters
    ----------
    node : str
        The target node for which the TramModel is constructed.
    configuration_dict : dict
        Dictionary describing all nodes and their metadata. Expected to contain:
        - 'nodes': dict mapping node names to metadata, including:
            - 'data_type': str (e.g., 'ordinal_yo', 'continuous')
            - 'levels': int (for ordinal variables)
            - 'parents_datatype': dict mapping parent names to data types
    set_initial_weights : bool, optional (default=False)
        If True, initializes the intercept model with increasing weights based
        on COLR/POLR fits obtained from R. Requires `TRAIN_DATA_PATH`.
    initial_data : str or pandas.DataFrame or None, optional
        Either a path to a CSV file or a pandas DataFrame. Required if
        `set_initial_weights=True`. Ignored otherwise.
    verbose : bool, optional (default=True)
        If True, prints high-level information about the model construction.
    debug : bool, optional (default=False)
        If True, prints detailed debug information during model construction and
        initialization.

    Returns
    -------
    TramModel
        A fully specified TramModel object with:
        - Intercept component (SimpleIntercept or a learned intercept network).
        - Shift components (one per parent group).

    Raises
    ------
    ValueError
        - If multiple complex intercept groups are detected.
        - If `set_initial_weights=True` but `TRAIN_DATA_PATH` is not provided.
        - If an unknown data type is encountered.
    RuntimeError
        If initialization via COLR/POLR fails.
    FileNotFoundError
        If the training data file does not exist at `TRAIN_DATA_PATH`.

    Notes
    -----
    - Ordinal outcomes use (levels - 1) thetas.
    - Continuous outcomes default to 20 thetas unless otherwise specified.
    - Intercept and shift networks are grouped by transformation prefixes
      (ci/si for intercepts, cs/ls for shifts).
    - Initialization requires R with the `tram` package if
      `set_initial_weights=True`.

    Examples
    --------
    >>> tram_model = get_fully_specified_tram_model(
    ...     node="y",
    ...     configuration_dict=config,
    ...     set_initial_weights=True,
    ...     TRAIN_DATA_PATH="data/train.csv",
    ...     debug=True
    ... )
    >>> print(tram_model)
    TramModel(intercept=..., shifts=[...])
    """
    
    # Settign the device
    if device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device
    device = torch.device(device_str)
    
    if debug:
        print(f'[DEBUG] get_fully_specified_tram_model(): device: {device}')
    
    from .sampling import is_outcome_modelled_ordinal, is_outcome_modelled_continous
    
    target_nodes = configuration_dict['nodes']
    default_number_thetas = 20  # default for continuous outcomes

    if debug:
        print(f"[DEBUG] default_number_thetas for continuous outcomes: {default_number_thetas}")

        
    # Determine number of thetas
    def compute_n_thetas(node, target_nodes: dict):
        if is_outcome_modelled_continous(node,target_nodes):
            return default_number_thetas
        if is_outcome_modelled_ordinal(node,target_nodes):
            return target_nodes[node]['levels'] - 1 

    # Compute number of input features for a given feature set
    def compute_n_features(feats, parents_datatype):
        n_features = 0
        for parent_name, _ in feats:
            parent_dt = parents_datatype[parent_name]
            if 'xn' in parent_dt.lower():
                n_features += target_nodes[parent_name]['levels']
            else:
                n_features += 1
        return n_features

    # Gather transformation terms and model class names
    _, terms_dict, model_names_dict = ordered_parents(node, target_nodes)
    model_dict = merge_transformation_dicts(terms_dict, model_names_dict)

    # Split into intercept and shift components by h_term prefixes
    intercepts_dict = {
        k: v for k, v in model_dict.items()
        if any(pref in v['h_term'] for pref in ("ci", "si"))
    }
    shifts_dict = {
        k: v for k, v in model_dict.items()
        if not any(pref in v['h_term'] for pref in ("ci", "si"))
    }

    # Build intercept network
    intercept_groups = group_by_base(intercepts_dict, prefixes=("ci", "si"))
    
    # Use specified thetas or default for continuous
    n_thetas = compute_n_thetas(node, target_nodes)
    
    if intercept_groups:
        if len(intercept_groups) > 1:
            raise ValueError("Multiple complex intercept groups detected; only one is supported.")
        feats = next(iter(intercept_groups.values()))
        cls_name = feats[0][1]['class_name']
        base_cls = get_base_model_class(cls_name)

        n_features = compute_n_features(feats, target_nodes[node]['parents_datatype'])
        nn_int = globals()[base_cls](n_features=n_features, n_thetas=n_thetas)
    else:
        nn_int = SimpleIntercept(n_thetas=n_thetas) 
        
    # set initial weights to be increasing for Intercept Model
    if set_initial_weights and nn_int is not None:

        if initial_data is None:
            raise ValueError(
                "initial_data (path to .csv OR pandas Dataframe) must be provided when set_initial_weights=True"
            )
            

        if initial_data is not None:
            # case 1: DataFrame -> write to temporary CSV

            
            
            if hasattr(initial_data, "to_csv"):
                
                TEMP_DIR = "temp"
                os.makedirs(TEMP_DIR, exist_ok=True)
                
                TEMP_CSV_PATH = os.path.join(TEMP_DIR, f"initial_data_{int(time.time())}.csv")
                initial_data.to_csv(TEMP_CSV_PATH, index=False)
                initial_data = TEMP_CSV_PATH
                if debug:
                    print(f"[DEBUG] Wrote DataFrame to temporary CSV: {TEMP_CSV_PATH}")
                    
                    
            # case 2: assume it's already a path -> check existence
            elif isinstance(initial_data, str):
                if not os.path.exists(initial_data):
                    raise FileNotFoundError(f"[ERROR] CSV path not found: {initial_data}")
            else:
                raise TypeError("[ERROR] initial_data must be either a str path, a DataFrame, or None.")
    
        # init_last_layer_increasing(nn_int, start=-3.0, end=3.0)
        # init_last_layer_hardcoded(nn_int)
        init_last_layer_COLR_POLR(
            nn_int,
            node,
            configuration_dict,
            n_thetas,
            TRAIN_DATA_PATH=TEMP_CSV_PATH,
            debug=debug
        )
        if debug or verbose:
            print(f"[INFO] Initialized intercept model with preinitialized weights: {nn_int}")
        
        if TEMP_CSV_PATH and os.path.exists(TEMP_CSV_PATH):
            try:
                os.remove(TEMP_CSV_PATH)
                if debug:
                    print(f"[DEBUG] Removed temporary CSV file: {TEMP_CSV_PATH}")
            except Exception as e:
                print(f"[WARNING] Could not delete temporary file: {e}")

    # Build shift networks
    shift_groups = group_by_base(shifts_dict, prefixes=("cs", "ls"))
    nn_shifts = []
    for feats in shift_groups.values():
        cls_name = feats[0][1]['class_name']
        base_cls = get_base_model_class(cls_name)
        n_features = compute_n_features(feats, target_nodes[node]['parents_datatype'])
        nn_shifts.append(globals()[base_cls](n_features=n_features))

    # Combine into final TramModel
    tram_model = TramModel(nn_int, nn_shifts,device=device)
    return tram_model


def preprocess_inputs(x, transformation_terms, device='cuda'):
    """
    Prepares model input by grouping features by transformation term base:
      - ci11, ci12 → 'ci1' (intercept)
      - cs11, cs12 → 'cs1' (shift)
      - cs21 → 'cs2' (another shift group)
      - cs, ls → treated as full group keys
    Returns:
      - int_inputs: Tensor of shape (B, n_features) for intercept model
      - shift_list: List of tensors for each shift model, shape (B, group_features)
    """
    
    transformation_terms=list(transformation_terms)
    
    ## if there is only a source so transforamtion terms is 0:
    x = [xi.to(device, non_blocking=True) for xi in x]
    if len(transformation_terms)== 0:
        x = [xi.unsqueeze(1) for xi in x] 
        int_inputs= x[0]
        return int_inputs, None

    # Always ensure there's an intercept term
    if not any('ci' in str(value) for value in transformation_terms):
        transformation_terms.insert(0, 'si')

    # Lists to collect intercept tensors and shift‐groups
    int_tensors = []
    shift_groups = []

    # Helpers to track the “current” shift‐group for numbered suffixes
    current_group = None
    current_key = None

    for tensor, term in zip(x, transformation_terms):
        # 1) INTERCEPT terms (si*, ci*)
        if term.startswith(('si','ci')):
            int_tensors.append(tensor)

        # 2) SHIFT terms (cs*, ls*)
        elif term.startswith(('cs','ls')):
            # numbered suffix → group by the first 3 chars (e.g. 'cs11'/'cs12' → 'cs1')
            if len(term) > 2 and term[2].isdigit():
                key = term[:3]
                # start a new group if key changed
                if current_group is None or current_key != key:
                    current_group = []
                    shift_groups.append(current_group)
                    current_key = key
                current_group.append(tensor)

            # lone 'cs' or 'ls' → always its own group
            else:
                current_group = [tensor]
                shift_groups.append(current_group)
                current_key = None
        else:
            raise ValueError(f"[ERROR] Unknown transformation term: {term}")

    # Intercept: should be exactly one group
    if len(int_tensors) == 0:
        raise ValueError("[ERROR] No intercept tensors found!")
    int_inputs = torch.cat(
        [t.to(device, non_blocking=True).view(t.shape[0], -1) for t in int_tensors],
        dim=1
    )

    shift_list = [
        torch.cat([t.to(device, non_blocking=True).view(t.shape[0], -1) for t in group], dim=1)
        for group in shift_groups
    ]

    return int_inputs, shift_list if shift_list else None

def get_base_model_class(class_name: str):
    """
    Strip trailing digits from a class name to get its base name.
    e.g. "cs12" → "cs"
    """
    for i, c in enumerate(class_name):
        if c.isdigit():
            return class_name[:i]
    return class_name

def group_by_base(term_dict, prefixes):
    """
    Group features by their h_term “base,” but if the h_term is exactly
    equal to one of the prefixes (e.g. "cs" or "ls"), keep each feature separate.

    :param term_dict: { feature_name: { 'h_term': h_term, ... }, ... }
    :param prefixes:  single prefix or iterable of prefixes, e.g. "cs" or ("cs","ls")
    :return: defaultdict(list) mapping group key → list of (feature_name, conf) pairs
    """
    if isinstance(prefixes, str):
        prefixes = (prefixes,)
    groups = defaultdict(list)

    for feat, conf in term_dict.items():
        h_term = conf['h_term']
        for prefix in prefixes:
            if h_term.startswith(prefix):
                # Case 1: exact prefix match → separate group per feature
                if h_term == prefix:
                    key = feat
                # Case 2: prefix plus a digit → group by prefix+first digit, e.g. "cs11","cs12" → "cs1"
                elif len(h_term) > len(prefix) and h_term[len(prefix)].isdigit():
                    key = h_term[:len(prefix) + 1]
                # Case 3: anything else (e.g. "csA", "lsXyz") → group by full h_term
                else:
                    key = h_term
                groups[key].append((feat, conf))
                break

    return groups

@torch.no_grad()
def init_last_layer_COLR_POLR(
    module: nn.Module,
    node: str,
    configuration_dict: dict,
    theta_count: int,
    TRAIN_DATA_PATH: str,
    debug: bool = False,
):
    """
    Initialize the last linear layer of a PyTorch module with intercept weights
    estimated from an equivalent R model (COLR for continuous outcomes, POLR for ordinal outcomes).

    This function:
    - Finds the last `nn.Linear` layer in the given module.
    - Determines the target type (continuous vs. ordinal) from the configuration.
    - Runs an R subprocess to fit a COLR or POLR model and extract intercepts.
    - Applies the appropriate inverse transformation to convert R-estimated thetas
      into theta_tilde values compatible with TRAM models.
    - Initializes the last linear layer such that:
        * The first input channel encodes the transformed intercepts.
        * All other input channels are zero-initialized.
        * Bias is reset to zero (if present).

    Parameters
    ----------
    module : nn.Module
        PyTorch module containing at least one `nn.Linear` layer.
        The last linear layer will be initialized.
    node : str
        Target node name (must exist in `configuration_dict['nodes']`).
    configuration_dict : dict
        Experiment configuration dictionary. Must contain:
        - 'nodes': dict mapping node names → metadata (including data types).
        - 'PATHS': dict with dataset paths.
        - 'experiment_name': str, used to resolve dataset file paths.
    theta_count : int
        Expected number of intercepts (thetas) for the outcome.
    debug : bool, optional (default=False)
        If True, prints debug information from the R subprocess and
        initialization steps.
    TRAIN_DATA_PATH : str, optional
        Path to the training dataset CSV file. Must exist.

    Returns
    -------
    nn.Linear
        The initialized last linear layer.

    Raises
    ------
    ValueError
        - If no `nn.Linear` layer is found in the module.
        - If the number of transformed intercepts does not match
          the output dimension of the last linear layer.
    FileNotFoundError
        If the training data file does not exist at `TRAIN_DATA_PATH`.
    RuntimeError
        If the R subprocess or the inverse transformation fails.

    Notes
    -----
    - Continuous targets use COLR + `inverse_transform_intercepts_continous`.
    - Ordinal targets use POLR + `inverse_transform_intercepts_ordinal`
      (with `-inf` and `+inf` padding).
    - Requires R with `MASS`, `tram`, and `readr` installed and callable
      via `Rscript`.

    Examples
    --------
    >>> model = TramDag(...)
    >>> last_layer = init_last_layer_COLR_POLR(
    ...     model,
    ...     node="y",
    ...     configuration_dict=config,
    ...     theta_count=3,
    ...     TRAIN_DATA_PATH="data/train.csv",
    ...     debug=True
    ... )
    >>> print(last_layer.weight)
    tensor([...])  # initialized intercept weights
    """
    from .sampling import is_outcome_modelled_ordinal, is_outcome_modelled_continous
    
    target_nodes_dict=configuration_dict['nodes']
    
    last_linear = None
    for m in reversed(list(module.modules())):
        if isinstance(m, nn.Linear):
            last_linear = m
            break
    if last_linear is None:
        raise ValueError("No nn.Linear layer found in module.")

    if is_outcome_modelled_continous(node,target_nodes_dict):
        dtype='continous'
    if is_outcome_modelled_ordinal(node,target_nodes_dict):
        dtype='ordinal'
            
    
    if not os.path.exists(TRAIN_DATA_PATH):
        raise FileNotFoundError(
            f"[ERROR] train_df does not exist.\n"
            f"Please provide it under attribute: TRAIN_DATA_PATH= \n"
            f"Expected type: {'*.csv'}"
        )
    
    thetas_R=fit_r_model_subprocess(node, dtype, theta_count, TRAIN_DATA_PATH, debug=debug)
    thetas_R=torch.tensor(thetas_R)



    if dtype=='continous':
        if debug:
            print(f"[DEBUG] R-estimated thetas for node '{node}': {thetas_R.numpy()}")
            print(f"[DEBUG] R-estimated thetas shape: {thetas_R.shape}")
            print(f"[DEBUG] Expected number of thetas: {theta_count}")
            print(f"[DEBUG] Last linear layer shape: {last_linear.weight.shape}")
            print(f"[DEBUG] Inverse transforming thetas for continuous outcome...")
        try:
            theta_tilde=inverse_transform_intercepts_continous(thetas_R)
        except Exception as e:
            raise RuntimeError(f"Error in inverse_transform_intercepts_continous: {e}")
        
        
    if dtype == 'ordinal':
        if debug:
            print(f"[DEBUG] R-estimated thetas for node '{node}': {thetas_R.numpy()}")
            print(f"[DEBUG] Expected number of thetas: {theta_count}")
            print(f"[DEBUG] Last linear layer shape: {last_linear.weight.shape}")
            print(f"[DEBUG] Inverse transforming thetas for ordinal outcome...")

        try:
            # Add -inf and +inf around thetas_R
            int_out = torch.cat([
                torch.full((1, 1), -float("inf"), dtype=thetas_R.dtype),
                thetas_R.unsqueeze(0) if thetas_R.ndim == 1 else thetas_R,
                torch.full((1, 1), float("inf"), dtype=thetas_R.dtype)
            ], dim=1)

            theta_tilde = inverse_transform_intercepts_ordinal(int_out)
        except Exception as e:
            raise RuntimeError(f"Error in inverse_transform_intercepts_ordinal: {e}")
    
    if debug:
        print(f"[DEBUG] Transformed thetas -> theta_tilde: {theta_tilde.numpy()}")
        print(f"[DEBUG] Transformed thetas -> theta_tilde shape: {theta_tilde.shape}")
    
    
    # theta_tilde = torch.tensor(theta_tilde, dtype=last_linear.weight.dtype, device=last_linear.weight.device)
    theta_tilde = theta_tilde.to(dtype=last_linear.weight.dtype,device=last_linear.weight.device)

    if theta_tilde.numel() != last_linear.out_features:
        raise ValueError(
            f"Hardcoded vector has {theta_tilde.numel()} elements, "
            f"but last layer expects {last_linear.out_features}"
        )

    # Expand to (out_features, in_features)
    w = torch.zeros((last_linear.out_features, last_linear.in_features),
                    dtype=last_linear.weight.dtype,
                    device=last_linear.weight.device)
    w[:, 0] = theta_tilde  # fill the first input channel

    # Copy into the model
    last_linear.weight.copy_(w)
    if last_linear.bias is not None:
        last_linear.bias.zero_()

    return last_linear

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

######################################### Model Training #####################################

def model_train_val_paths(NODE_DIR):
    MODEL_PATH = os.path.join(NODE_DIR, "best_model.pt")
    LAST_MODEL_PATH = os.path.join(NODE_DIR, "last_model.pt")
    TRAIN_HIST_PATH = os.path.join(NODE_DIR, "train_loss_hist.json")
    VAL_HIST_PATH = os.path.join(NODE_DIR, "val_loss_hist.json")
    return MODEL_PATH,LAST_MODEL_PATH,TRAIN_HIST_PATH,VAL_HIST_PATH

def check_if_training_complete(node, NODE_DIR, epochs):
    """
    Check if the training for the given node is complete.
    Returns True if training is complete, False otherwise.
    """
    MODEL_PATH, _, TRAIN_HIST_PATH, VAL_HIST_PATH = model_train_val_paths(NODE_DIR)
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(TRAIN_HIST_PATH) and os.path.exists(VAL_HIST_PATH):
            with open(TRAIN_HIST_PATH, 'r') as f:
                train_loss_hist = json.load(f)

            start_epoch = len(train_loss_hist)
            
            # return false if already trained to skip to the next node 
            if start_epoch >= epochs:
                print(f"Node {node} already trained for {epochs} epochs. Skipping.")
                return False
            else:
                print(f"Node {node} not trained yet or training incomplete. Starting from epoch {start_epoch}.")
                return True
        else:
            
            return True
        
    except Exception as e:
        print(f"Error checking training status for node {node}: {e}")
        return False


def train_val_loop( 
    node,
    target_nodes,
    NODE_DIR,
    tram_model: torch.nn.Module,
    train_loader,  # can be DataLoader or Dataset
    val_loader,    # can be DataLoader or Dataset
    epochs: int,
    optimizer: torch.optim.Optimizer,
    use_scheduler: bool,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    save_linear_shifts: bool = False,
    save_simple_intercepts: bool = False,
    verbose: bool = True,
    device: str = 'cpu',
    debug: bool = False,
    min_max=None,
):

    """
    Train and validate a TRAM model (TramModel) with optional saving of shift and intercept weights.

    The function supports both ONTRAM and CONTRAM loss and can work with either PyTorch DataLoader
    or Dataset objects. It manages checkpointing, model reloading, loss tracking, learning rate
    scheduling, and optional saving of linear shift and simple intercept layer weights across epochs.

    Parameters
    ----------
    node : str
        Identifier for the target node being trained.
    target_nodes : dict
        Dictionary of all target nodes containing metadata such as min/max values and data_type.
    NODE_DIR : str
        Directory where model checkpoints, histories, and weight files will be saved.
    tram_model : torch.nn.Module
        TRAM model instance combining intercept and (optionally) shift networks.
    train_loader : torch.utils.data.DataLoader or torch.utils.data.Dataset
        Training data iterable or dataset.
    val_loader : torch.utils.data.DataLoader or torch.utils.data.Dataset
        Validation data iterable or dataset.
    epochs : int
        Number of training epochs.
    optimizer : torch.optim.Optimizer
        Optimizer for training the model.
    use_scheduler : bool
        Whether to use a learning rate scheduler.
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        Learning rate scheduler instance to step each epoch.
    save_linear_shifts : bool, default=False
        If True, saves the weights of all linear shift networks per epoch.
    save_simple_intercepts : bool, default=False
        If True, saves the weights of the simple intercept network per epoch.
    verbose : bool, default=True
        If True, prints progress information per epoch.
    device : str, default='cpu'
        Target device for training ('cpu' or 'cuda').
    debug : bool, default=False
        Enables detailed timing and batch-level debug information.
    min_max : torch.Tensor, optional
        Precomputed tensor containing min and max normalization values (shape: [2, n_vars]).
        If None, values are inferred from target_nodes[node].

    Returns
    -------
    tuple[list[float], list[float]]
        train_loss_hist : list of average training losses per epoch.
        val_loss_hist : list of average validation losses per epoch.

    Side Effects
    -------------
    - Saves model checkpoints:
        * `<NODE_DIR>/model_best.pt` — best validation model
        * `<NODE_DIR>/model_last.pt` — most recent model
    - Saves training and validation loss histories:
        * `<NODE_DIR>/train_loss_hist.json`
        * `<NODE_DIR>/val_loss_hist.json`
    - Optionally saves:
        * `<NODE_DIR>/linear_shifts_all_epochs.json`
        * `<NODE_DIR>/simple_intercepts_all_epochs.json`

    Notes
    -----
    - For ONTRAM models, the `ontram_nll` loss is used.
    - For CONTRAM models, the `contram_nll` loss is used, with normalization controlled by `min_max`.
    - The intercept model (`SimpleIntercept`) outputs weights of shape (n_thetas, 1), default n_thetas=20.
    - If an existing model and loss history exist, training resumes from the last epoch.
    """




    ################################## 1. PREPARATION ##################################
    device = torch.device(device)

    if debug:
        print(f'[DEBUG] train_val_loop(): device: {device}')

    BEST_MODEL_PATH, LAST_MODEL_PATH, TRAIN_HIST_PATH, VAL_HIST_PATH = model_train_val_paths(NODE_DIR)
    tram_model = tram_model.to(device)

    # Prepare min/max normalization values
    if min_max is None:
        min_vals = torch.tensor(target_nodes[node]['min'], dtype=torch.float32, device=device)
        max_vals = torch.tensor(target_nodes[node]['max'], dtype=torch.float32, device=device)
        min_max = torch.stack([min_vals, max_vals], dim=0)

    # Load existing model if available
    if os.path.exists(LAST_MODEL_PATH) and os.path.exists(TRAIN_HIST_PATH) and os.path.exists(VAL_HIST_PATH):
        if verbose or debug:
            print("[INFO] Existing model found. Loading weights and history from LAST model...")
        state_dict = torch.load(LAST_MODEL_PATH, map_location=device)
        tram_model.load_state_dict(state_dict)
        tram_model.to(device)
        with open(TRAIN_HIST_PATH, 'r') as f:
            train_loss_hist = json.load(f)
        with open(VAL_HIST_PATH, 'r') as f:
            val_loss_hist = json.load(f)
        start_epoch = len(train_loss_hist)
        best_val_loss = min(val_loss_hist)
    else:
        if verbose or debug:
            print(f"[INFO] No existing trained model found.\n Starting fresh...")
        
        train_loss_hist, val_loss_hist = [], []
        start_epoch = 0
        best_val_loss = float('inf')

    # CHECK whetere the loaders are DataLoader or Dataset
    train_is_dataloader = isinstance(train_loader, DataLoader)
    val_is_dataloader = isinstance(val_loader, DataLoader)
 
    # flag to choose correct loss function
    is_ontram = 'yo' in target_nodes[node]['data_type'].lower()

    ## needed to save shift terms
    _, terms_dict, _ =ordered_parents(node, target_nodes)
    
    ################################## 2. MAIN LOOP ##################################
    for epoch in range(start_epoch, epochs):
        tram_model.train()
        train_loss = 0.0

        if verbose or debug:
            print(f"\n===== Epoch {epoch+1}/{epochs} =====")

        epoch_start = time.time() if (verbose or debug) else None

        if debug:
            prev_fetch_end = time.time()

        ################################## 2.1 TRAINING ##################################
        if train_is_dataloader:
            train_iterable = enumerate(train_loader)
        else:
            train_iterable = enumerate(range(len(train_loader)))


        for batch_idx, item in train_iterable:
            if train_is_dataloader:
                (int_input, shift_list), y = item
            else:
                (int_input, shift_list), y = train_loader[item]

                # ensure tensors have batch dimension
                if torch.is_tensor(int_input):
                    int_input = int_input.unsqueeze(0)
                shift_list = [s.unsqueeze(0) if torch.is_tensor(s) else s for s in shift_list]
                if torch.is_tensor(y):
                    y = y.unsqueeze(0)

            if debug:
                fetch_time = time.time() - prev_fetch_end
                print(f"[DEBUG] Batch {batch_idx} fetch: {fetch_time:.4f}s")
                step_start = time.time()

            # Move to device
            if debug:
                t0 = time.time()
            int_input = int_input.to(device)
            shift_list = [s.to(device) for s in shift_list]
            y = y.to(device)
            
            
            
            if debug:
                print(f"[DEBUG] Move to device: {time.time() - t0:.4f}s")

            # Forward
            if debug:
                t1 = time.time()
            y_pred = tram_model(int_input=int_input, shift_input=shift_list)
            if debug:
                print(f"[DEBUG] Forward: {time.time() - t1:.4f}s")

            # Loss
            if debug:
                t2 = time.time()
            if is_ontram:
                loss = ontram_nll(y_pred, y)
            else:
                loss = contram_nll(y_pred, y, min_max=min_max)
            if debug:
                print(f"[DEBUG] Loss: {time.time() - t2:.4f}s")

            # Backward + Step
            if debug:
                t3 = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if debug:
                print(f"[DEBUG] Backward+Step: {time.time() - t3:.4f}s")
                print(f"[DEBUG] BATCH {batch_idx} Total time: {time.time() - step_start:.4f}s\n")

            train_loss += loss.item()
            if debug:
                prev_fetch_end = time.time()

        # Scheduler
        if use_scheduler and scheduler is not None:
            if debug:
                t_sched = time.time()
            scheduler.step()
            if debug:
                print(f"[DEBUG] Scheduler step: {time.time() - t_sched:.4f}s")

        avg_train_loss = train_loss / (len(train_loader) if len(train_loader) > 0 else 1)
        train_loss_hist.append(avg_train_loss)

        ################################## 2.2 Validation ##################################
        tram_model.eval()
        val_loss = 0.0
        if debug:
            prev_val_fetch = time.time()
        if val_loader is not None:
            if val_is_dataloader:
                val_iterable = enumerate(val_loader)
            else:
                val_iterable = enumerate(range(len(val_loader)))

            with torch.no_grad():
                for batch_idx, item in val_iterable:
                    if val_is_dataloader:
                        (int_input, shift_list), y = item
                    else:
                        (int_input, shift_list), y = val_loader[item]
                        if torch.is_tensor(int_input):
                            int_input = int_input.unsqueeze(0)
                        shift_list = [s.unsqueeze(0) if torch.is_tensor(s) else s for s in shift_list]
                        if torch.is_tensor(y):
                            y = y.unsqueeze(0)

                    if debug:
                        fetch_time = time.time() - prev_val_fetch
                        print(f"[DEBUG] VAL Batch {batch_idx} fetch: {fetch_time:.4f}s")
                        bval_start = time.time()

                    int_input = int_input.to(device)
                    shift_list = [s.to(device) for s in shift_list]
                    y = y.to(device)

                    if debug:
                        tval0 = time.time()
                    y_pred = tram_model(int_input=int_input, shift_input=shift_list)
                    if debug:
                        print(f"[DEBUG] VAL Forward: {time.time() - tval0:.4f}s")

                    if debug:
                        tval1 = time.time() 
                    if is_ontram:
                        loss = ontram_nll(y_pred, y)
                    else:
                        loss = contram_nll(y_pred, y, min_max=min_max)
                        
                        
                    if debug:
                        print(f"[DEBUG] VAL Loss: {time.time() - tval1:.4f}s")
                        print(f"[DEBUG] VAL BATCH {batch_idx} Total: {time.time() - bval_start:.4f}s")

                    val_loss += loss.item()
                    if debug:
                        prev_val_fetch = time.time()

            avg_val_loss = val_loss / (len(val_loader) if len(val_loader) > 0 else 1)
        else:
            avg_val_loss = float('nan')

        val_loss_hist.append(avg_val_loss)

        ################################## 2.2.5 SAVE LINEAR SHIFTS ##################################
        if save_linear_shifts and hasattr(tram_model, "nn_shift") and tram_model.nn_shift is not None:
            shift_path = os.path.join(NODE_DIR, "linear_shifts_all_epochs.json")

            # Load existing weights if available
            if os.path.exists(shift_path):
                with open(shift_path, "r") as f:
                    all_shift_weights = json.load(f)
            else:
                all_shift_weights = {}
                
            
            # Collect current epoch’s shift weights
            epoch_weights = {}
            for i, shift_layer in enumerate(tram_model.nn_shift):
                module_name = shift_layer.__class__.__name__
                if hasattr(shift_layer, "fc") and hasattr(shift_layer.fc, "weight") and module_name == 'LinearShift': 
                    epoch_weights[f"ls({list(terms_dict.keys())[i]})"] = shift_layer.fc.weight.detach().cpu().squeeze().tolist()
                else:
                    if debug:
                        print(f"[DEBUG] shift_{i}: 'fc' or 'weight' not found.")

            # Append to global dict under current epoch
            all_shift_weights[f"epoch_{epoch+1}"] = epoch_weights

            # Write back to disk
            with open(shift_path, "w") as f:
                json.dump(all_shift_weights, f)

            if verbose or debug:
                print(f"[INFO] Saved linear shift weights for epoch {epoch+1} -> {shift_path}")


        ################################## 2.2.6 SAVE SI INTERCEPTS ##################################
        if save_simple_intercepts and hasattr(tram_model, "nn_int") and tram_model.nn_int is not None and isinstance(tram_model.nn_int, SimpleIntercept):
            si_path = os.path.join(NODE_DIR, "simple_intercepts_all_epochs.json")

            # Load existing weights if available
            if os.path.exists(si_path):
                with open(si_path, "r") as f:
                    all_int_weights = json.load(f)
            else:
                all_int_weights = {}

            # Collect current epoch’s intercept weights
            if hasattr(tram_model.nn_int, "fc") and hasattr(tram_model.nn_int.fc, "weight"):
                epoch_weights = tram_model.nn_int.fc.weight.detach().cpu().tolist()
            else:
                epoch_weights = None
                if debug:
                    print(f"[DEBUG] nn_int: 'fc' or 'weight' not found.")

            
            epoch_weights_tensor = torch.Tensor(epoch_weights)
            if debug:
                print(f'[DEBUG] epoch_weights_tensor (theta tilde) {epoch_weights_tensor} ')
                print(f'[DEBUG] epoch_weights_tensor.shape (theta tilde) {epoch_weights_tensor.shape} ')
                
            if is_ontram:
                # transform the theta tilde to thetas # removed the -inf and +inf for ordinal reshapeing to preserve order, 
                epoch_weights = transform_intercepts_ordinal(epoch_weights_tensor.reshape(1, -1))[:, 1:-1].reshape(-1, 1)
                
            else:
                epoch_weights=transform_intercepts_continous(epoch_weights_tensor.reshape(1, -1)).reshape(-1, 1)
                
            # Append to global dict under current epoch
            if debug:
                print(f'[DEBUG] epoch_weights  (theta) after transformatin {epoch_weights} ')
            
            epoch_weights=epoch_weights.tolist()
            
            all_int_weights[f"epoch_{epoch+1}"] = epoch_weights

            # Write back to disk
            with open(si_path, "w") as f:
                json.dump(all_int_weights, f)

            if verbose or debug:
                print(f"[INFO] Saved simple intercept weights for epoch {epoch+1} -> {si_path}")

        ################################## 2.3 SAVING MODEL STATE ##################################
        if debug:
            save_start = time.time()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(tram_model.state_dict(), BEST_MODEL_PATH)
            if verbose or debug:
                print("[INFO] Saved new best model.")

        torch.save(tram_model.state_dict(), LAST_MODEL_PATH)
        with open(TRAIN_HIST_PATH, 'w') as f:
            json.dump(train_loss_hist, f)
        with open(VAL_HIST_PATH, 'w') as f:
            json.dump(val_loss_hist, f)

        if debug:
            print(f"[DEBUG] Saving epoch artifacts: {time.time() - save_start:.4f}s")

        if verbose or debug:
            total_time = time.time() - epoch_start if epoch_start is not None else 0.0
            print(f"[INFO] Epoch {epoch+1}: Train NLL={avg_train_loss:.4f} | Val NLL={avg_val_loss:.4f} | Time={total_time:.2f}s")
    return (train_loss_hist, val_loss_hist)


def evaluate_tramdag_model(
    node: str,
    target_nodes: dict,
    NODE_DIR: str,
    tram_model: torch.nn.Module,
    data_loader,
    min_max=None,
    device: str = 'cpu',
    debug: bool = False,
):
    """
    Evaluate a trained TRAM model by computing the average NLL loss on a given dataset.

    This function performs forward passes only (no gradients, no saving, no checkpointing)
    and returns the mean NLL value across all batches.

    Parameters
    ----------
    node : str
        Identifier for the target node being evaluated.
    target_nodes : dict
        Dictionary of all target nodes containing metadata such as min/max values and data_type.
    NODE_DIR : str
        Directory path (only used for context consistency).
    tram_model : torch.nn.Module
        Trained TRAM model to be evaluated.
    data_loader : torch.utils.data.DataLoader or torch.utils.data.Dataset
        Iterable or dataset containing evaluation data.
    min_max : torch.Tensor, optional
        Precomputed tensor containing min and max normalization values.
        If None, will be inferred from target_nodes[node].
    device : str, default='cpu'
        Device for model evaluation.
    debug : bool, default=False
        Enables timing and detailed debug information.

    Returns
    -------
    float
        Average NLL loss value over the dataset.
    """

    device = torch.device(device)
    tram_model = tram_model.to(device)
    tram_model.eval()

    if min_max is None:
        min_vals = torch.tensor(target_nodes[node]['min'], dtype=torch.float32, device=device)
        max_vals = torch.tensor(target_nodes[node]['max'], dtype=torch.float32, device=device)
        min_max = torch.stack([min_vals, max_vals], dim=0)

    is_ontram = 'yo' in target_nodes[node]['data_type'].lower()
    is_dataloader = isinstance(data_loader, torch.utils.data.DataLoader)

    total_loss = 0.0
    n_batches = len(data_loader) if len(data_loader) > 0 else 1

    with torch.no_grad():
        if is_dataloader:
            iterable = enumerate(data_loader)
        else:
            iterable = enumerate(range(len(data_loader)))

        for batch_idx, item in iterable:
            if is_dataloader:
                (int_input, shift_list), y = item
            else:
                (int_input, shift_list), y = data_loader[item]
                if torch.is_tensor(int_input):
                    int_input = int_input.unsqueeze(0)
                shift_list = [s.unsqueeze(0) if torch.is_tensor(s) else s for s in shift_list]
                if torch.is_tensor(y):
                    y = y.unsqueeze(0)

            int_input = int_input.to(device)
            shift_list = [s.to(device) for s in shift_list]
            y = y.to(device)

            if debug:
                t0 = time.time()
            y_pred = tram_model(int_input=int_input, shift_input=shift_list)
            if debug:
                print(f"[DEBUG] Forward batch {batch_idx}: {time.time() - t0:.4f}s")

            if is_ontram:
                loss = ontram_nll(y_pred, y)
            else:
                loss = contram_nll(y_pred, y, min_max=min_max)

            total_loss += loss.item()

    avg_nll = total_loss / n_batches
    if debug:
        print(f"[DEBUG] Evaluation done. Average NLL: {avg_nll:.4f}")

    return avg_nll


######################################### History #####################################


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

def show_training_history(conf_dict,EXPERIMENT_DIR):
    plt.figure(figsize=(14, 12))
    # --- Full history (top plot) ---
    plt.subplot(2, 1, 1)
    for node in conf_dict:
        train_hist, val_hist = load_history(node, EXPERIMENT_DIR)
        if train_hist is None or val_hist is None:
            print(f"No history found for node: {node}")
            continue
        epochs = range(1, len(train_hist) + 1)
        plt.plot(epochs, train_hist, label=f"{node} - train", linestyle="--")
        plt.plot(epochs, val_hist, label=f"{node} - val")
    plt.title("Training and Validation NLL Across Nodes - Full History")
    plt.xlabel("Epoch")
    plt.ylabel("NLL")
    plt.legend()
    plt.grid(True)

    # --- Last 10% of epochs (bottom plot) ---
    plt.subplot(2, 1, 2)
    for node in conf_dict:
        train_hist, val_hist = load_history(node, EXPERIMENT_DIR)
        if train_hist is None or val_hist is None:
            continue
        total_epochs = len(train_hist)
        start_idx = int(total_epochs * 0.9)
        epochs = range(start_idx + 1, total_epochs + 1)
        plt.plot(epochs, train_hist[start_idx:], label=f"{node} - train", linestyle="--")
        plt.plot(epochs, val_hist[start_idx:], label=f"{node} - val")
    plt.title("Training and Validation NLL - Last 10% of Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("NLL")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def evaluate_standard_logistic_fit(data: np.ndarray, num_quantiles: int = 100):
    """
    Evaluate how well the input data fits a standard logistic distribution.

    Parameters:
    - data: np.ndarray of raw sample values
    - num_quantiles: Number of quantiles used in quantile-based metrics

    Returns:
    - dict with RMSE, MAE, KS test result, and Anderson-Darling statistic
    """
    data = np.asarray(data)
    probs = np.linspace(0, 1, num_quantiles + 2)[1:-1]  # exclude 0 and 1
    empirical_q = np.quantile(data, probs)
    theoretical_q = logistic.ppf(probs)  # standard logistic: loc=0, scale=1

    rmse = np.sqrt(np.mean((empirical_q - theoretical_q) ** 2))
    mae = np.mean(np.abs(empirical_q - theoretical_q))

    ks_stat, ks_pval = kstest(data, 'logistic')  # against standard logistic
    ad_result = anderson(data, dist='logistic')
    ad_stat = ad_result.statistic

    return {
        'quantile_rmse': rmse,
        'quantile_mae': mae,
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pval,
        'ad_statistic': ad_stat
    }               
                

