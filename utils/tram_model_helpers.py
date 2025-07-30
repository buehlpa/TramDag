from utils.tram_models import *
from utils.loss_continous import contram_nll
from utils.loss_ordinal import ontram_nll
from utils.configuration import *



import os
import re
from collections import OrderedDict, defaultdict
import json
import time
from functools import wraps
import shutil
from scipy.stats import logistic, kstest, anderson,probplot



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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
            raise ValueError(f"Unknown transformation term: {term}")

    # Intercept: should be exactly one group
    if len(int_tensors) == 0:
        raise ValueError("No intercept tensors found!")
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


def get_fully_specified_tram_model(node: str, target_nodes: dict, verbose=True):
    """
    returns a Trammodel fully specified , according to CI groups and CS groups , for ordinal outcome and inputs

    """
    # Helper to detect ordinal with 'yo'
    def is_ordinal_yo(data_type: str) -> bool:
        return 'ordinal' in data_type and 'yo' in data_type.lower()

    # Determine number of thetas for ordinal nodes
    def compute_n_thetas(node_meta: dict):
        return node_meta['levels'] - 1 if is_ordinal_yo(node_meta['data_type']) else None

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
    if intercept_groups:
        if len(intercept_groups) > 1:
            raise ValueError("Multiple complex intercept groups detected; only one is supported.")
        feats = next(iter(intercept_groups.values()))
        cls_name = feats[0][1]['class_name']
        base_cls = get_base_model_class(cls_name)
        # Use specified thetas or default for continuous
        number_thetas = compute_n_thetas(target_nodes[node]) or 20
        n_features = compute_n_features(feats, target_nodes[node]['parents_datatype'])
        nn_int = globals()[base_cls](n_features=n_features, n_thetas=number_thetas)
    else:
        theta_count = compute_n_thetas(target_nodes[node])
        nn_int = SimpleIntercept(n_thetas=theta_count) if theta_count is not None else SimpleIntercept()

    # Build shift networks
    shift_groups = group_by_base(shifts_dict, prefixes=("cs", "ls"))
    nn_shifts = []
    for feats in shift_groups.values():
        cls_name = feats[0][1]['class_name']
        base_cls = get_base_model_class(cls_name)
        n_features = compute_n_features(feats, target_nodes[node]['parents_datatype'])
        nn_shifts.append(globals()[base_cls](n_features=n_features))

    # Combine into final TramModel
    tram_model = TramModel(nn_int, nn_shifts)
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
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    use_scheduler: bool,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    save_linear_shifts: bool = False,
    verbose: int = 1,
    device: str = 'cpu'
):
    # Resolve device
    device = torch.device(device)
    # Paths for saving
    MODEL_PATH, LAST_MODEL_PATH, TRAIN_HIST_PATH, VAL_HIST_PATH = \
        model_train_val_paths(NODE_DIR)

    # Move model (and any existing params) onto device immediately
    tram_model = tram_model.to(device)

    # Prepare min/max tensors for contram scaling
    min_vals = torch.tensor(target_nodes[node]['min'], dtype=torch.float32, device=device)
    max_vals = torch.tensor(target_nodes[node]['max'], dtype=torch.float32, device=device)
    min_max = torch.stack([min_vals, max_vals], dim=0)

    # Load old training state if it exists
    if os.path.exists(MODEL_PATH) and os.path.exists(TRAIN_HIST_PATH) and os.path.exists(VAL_HIST_PATH):
        if verbose:
            print("Existing model found. Loading weights and history...")
        tram_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        with open(TRAIN_HIST_PATH, 'r') as f:
            train_loss_hist = json.load(f)
        with open(VAL_HIST_PATH, 'r') as f:
            val_loss_hist = json.load(f)
        start_epoch = len(train_loss_hist)
        best_val_loss = min(val_loss_hist)
    else:
        if verbose:
            print("No existing model found. Starting fresh...")
        train_loss_hist, val_loss_hist = [], []
        start_epoch = 0
        best_val_loss = float('inf')

    # Main loop
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # --- Training ---
        tram_model.train()
        train_loss = 0.0
        train_start = time.time()
        for (int_input, shift_list), y in train_loader:
            # Move everything to device
            int_input = int_input.to(device)
            shift_list = [s.to(device) for s in shift_list]
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = tram_model(int_input=int_input, shift_input=shift_list)

            if 'yo' in target_nodes[node]['data_type'].lower():
                loss = ontram_nll(y_pred, y)
            else:
                loss = contram_nll(y_pred, y, min_max=min_max)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if use_scheduler and scheduler is not None:
            scheduler.step()

        avg_train_loss = train_loss / len(train_loader)
        train_loss_hist.append(avg_train_loss)
        train_time = time.time() - train_start

        # --- Validation ---
        tram_model.eval()
        val_loss = 0.0
        val_start = time.time()
        with torch.no_grad():
            for (int_input, shift_list), y in val_loader:
                int_input = int_input.to(device)
                shift_list = [s.to(device) for s in shift_list]
                y = y.to(device)

                y_pred = tram_model(int_input=int_input, shift_input=shift_list)
                if 'yo' in target_nodes[node]['data_type'].lower():
                    loss = ontram_nll(y_pred, y)
                else:
                    loss = contram_nll(y_pred, y, min_max=min_max)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_loss_hist.append(avg_val_loss)
        val_time = time.time() - val_start

        # --- Save linear shifts if requested ---
        if save_linear_shifts and hasattr(tram_model, 'nn_shift') and tram_model.nn_shift:
            shift_path = os.path.join(NODE_DIR, "linear_shifts_all_epochs.json")
            if os.path.exists(shift_path):
                with open(shift_path, 'r') as f:
                    all_shift_weights = json.load(f)
            else:
                all_shift_weights = {}

            epoch_weights = {}
            for i, shift_layer in enumerate(tram_model.nn_shift):
                if hasattr(shift_layer, 'fc') and hasattr(shift_layer.fc, 'weight'):
                    epoch_weights[f"shift_{i}"] = shift_layer.fc.weight.detach().cpu().tolist()
                else:
                    if verbose > 1:
                        print(f"shift_{i}: missing 'fc.weight'")
            all_shift_weights[f"epoch_{epoch+1}"] = epoch_weights
            with open(shift_path, 'w') as f:
                json.dump(all_shift_weights, f)
            if verbose > 1:
                print(f"Appended linear shift weights for epoch {epoch+1}")

        # --- Checkpointing ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(tram_model.state_dict(), MODEL_PATH)
            if verbose:
                print("Saved new best model.")
        # Always save last
        torch.save(tram_model.state_dict(), LAST_MODEL_PATH)
        with open(TRAIN_HIST_PATH, 'w') as f:
            json.dump(train_loss_hist, f)
        with open(VAL_HIST_PATH, 'w') as f:
            json.dump(val_loss_hist, f)

        # --- Epoch summary ---
        if verbose:
            total_time = time.time() - epoch_start
            print(
                f"Epoch {epoch+1}/{epochs}  "
                f"Train Loss: {avg_train_loss:.4f}  Val Loss: {avg_val_loss:.4f}  "
                f"[Train: {train_time:.2f}s  Val: {val_time:.2f}s  Total: {total_time:.2f}s]"
            )



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
                
