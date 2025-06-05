from utils.tram_models import *
from utils.continous import contram_nll
from utils.graph import *
import os
import re
from collections import OrderedDict
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

    int_inputs = x[0]  # TODO remove hardcoded stuff   use tuple as input 
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
    plt.title("Training and Validation Loss Across Nodes - Full History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
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
    plt.title("Training and Validation Loss - Last 10% of Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
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


#old test val loop 27.5
def train_val_loop(start_epoch,
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
                   NODE_DIR,
                   save_linear_shifts=False):
    
        """
        Executes the training and validation loop for the specified model.

        Args:
            start_epoch (int): The starting epoch number, useful for resuming training.
            epochs (int): Total number of epochs to train the model.
            tram_model (torch.nn.Module): The model to be trained and validated.
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            train_loss_hist (list): A list to store training loss values per epoch.
            val_loss_hist (list): A list to store validation loss values per epoch.
            best_val_loss (float): The current best validation loss; used for saving the best model.
            device (torch.device): The device to run the computations on (CPU or GPU).
            optimizer (torch.optim.Optimizer): Optimizer used to update model weights.
            use_scheduler (bool): Whether to use a learning rate scheduler.
            scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler to adjust learning rate.
            min_max (tuple): Tuple of (min, max) values for normalization used in the loss function.
            NODE_DIR (str): Directory path where model and training history should be saved.

        Saves:
            - Best model (with lowest validation loss) to a predefined path.
            - Last model of current epoch.
            - Training and validation loss histories as JSON files.

        Prints:
            - Training and validation losses per epoch.
            - Time taken for training, validation, and saving for each epoch.
        """
    
        MODEL_PATH,LAST_MODEL_PATH,TRAIN_HIST_PATH,VAL_HIST_PATH=model_train_val_paths(NODE_DIR)
        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()

            #####  Training #####
            train_start = time.time()
            train_loss = 0.0
            tram_model.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                y = y.to(device)

                int_input, shift_list = preprocess_inputs(x, device=device)

                y_pred = tram_model(int_input=int_input, shift_input=shift_list)
                loss = contram_nll(y_pred, y, min_max=min_max)
                loss.backward()
                optimizer.step()


                if use_scheduler:
                    scheduler.step()

                train_loss += loss.item()
            train_time = time.time() - train_start

            avg_train_loss = train_loss / len(train_loader)
            train_loss_hist.append(avg_train_loss)

            ##### Validation #####
            val_start = time.time()
            val_loss = 0.0
            tram_model.eval()
            
            with torch.no_grad():
                for x, y in val_loader:
                    y = y.to(device)
                    int_input, shift_list = preprocess_inputs(x, device=device)
                    y_pred = tram_model(int_input=int_input, shift_input=shift_list)

                    loss = contram_nll(y_pred, y, min_max=min_max)
                    val_loss += loss.item()
            val_time = time.time() - val_start

            avg_val_loss = val_loss / len(val_loader)
            val_loss_hist.append(avg_val_loss)

            ##### Save linear shift weights #####

            if save_linear_shifts and tram_model.nn_shift is not None:
                # Define the path for the cumulative JSON file
                shift_path = os.path.join(NODE_DIR, "linear_shifts_all_epochs.json")

                # Load existing data if the file exists
                if os.path.exists(shift_path):
                    with open(shift_path, 'r') as f:
                        all_shift_weights = json.load(f)
                else:
                    all_shift_weights = {}

                # Prepare current epoch's shift weights
                epoch_weights = {}
                for i in range(len(tram_model.nn_shift)):
                    shift_layer = tram_model.nn_shift[i]
                    
                    if hasattr(shift_layer, 'fc') and hasattr(shift_layer.fc, 'weight'):
                        epoch_weights[f"shift_{i}"] = shift_layer.fc.weight.detach().cpu().tolist()
                    else:
                        print(f"shift_{i}: 'fc' or 'weight' layer does not exist.")
                
                # Add to the dictionary under current epoch
                all_shift_weights[f"epoch_{epoch+1}"] = epoch_weights

                # Write back the updated dictionary
                with open(shift_path, 'w') as f:
                    json.dump(all_shift_weights, f)

                print(f"Appended linear shift weights for epoch {epoch+1} to: {shift_path}")

            ##### Saving #####
            save_start = time.time()
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(tram_model.state_dict(), MODEL_PATH)
                print("Saved new best model.")

            torch.save(tram_model.state_dict(), LAST_MODEL_PATH)

            with open(TRAIN_HIST_PATH, 'w') as f:
                json.dump(train_loss_hist, f)
            with open(VAL_HIST_PATH, 'w') as f:
                json.dump(val_loss_hist, f)
            save_time = time.time() - save_start

            epoch_total = time.time() - epoch_start

            ##### Epoch Summary #####
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"  [Train: {train_time:.2f}s | Val: {val_time:.2f}s | Save: {save_time:.2f}s | Total: {epoch_total:.2f}s]")

# from torch.cuda.amp import autocast, GradScaler

# def train_val_loop(start_epoch,
#                    epochs,
#                    tram_model,
#                    train_loader,
#                    val_loader,
#                    train_loss_hist,
#                    val_loss_hist,
#                    best_val_loss,
#                    device,
#                    optimizer,
#                    use_scheduler,
#                    scheduler,
#                    min_max,
#                    NODE_DIR):

#     """
#     Executes the training and validation loop for the specified model using AMP.

#     Args:
#         start_epoch (int): The starting epoch number, useful for resuming training.
#         epochs (int): Total number of epochs to train the model.
#         tram_model (torch.nn.Module): The model to be trained and validated.
#         train_loader (torch.utils.data.DataLoader): DataLoader for training data.
#         val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
#         train_loss_hist (list): A list to store training loss values per epoch.
#         val_loss_hist (list): A list to store validation loss values per epoch.
#         best_val_loss (float): The current best validation loss; used for saving the best model.
#         device (torch.device): The device to run the computations on (CPU or GPU).
#         optimizer (torch.optim.Optimizer): Optimizer used to update model weights.
#         use_scheduler (bool): Whether to use a learning rate scheduler.
#         scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler to adjust learning rate.
#         min_max (tuple): Tuple of (min, max) values for normalization used in the loss function.
#         NODE_DIR (str): Directory path where model and training history should be saved.

#     Saves:
#         - Best model (with lowest validation loss) to a predefined path.
#         - Last model of current epoch.
#         - Training and validation loss histories as JSON files.

#     Prints:
#         - Training and validation losses per epoch.
#         - Time taken for training, validation, and saving for each epoch.
#     """

#     MODEL_PATH, LAST_MODEL_PATH, TRAIN_HIST_PATH, VAL_HIST_PATH = model_train_val_paths(NODE_DIR)
#     scaler = GradScaler()

#     for epoch in range(start_epoch, epochs):
#         epoch_start = time.time()

#         ##### Training #####
#         train_start = time.time()
#         train_loss = 0.0
#         tram_model.train()

#         for x, y in train_loader:
#             optimizer.zero_grad()
#             y = y.to(device)

#             int_input, shift_list = preprocess_inputs(x, device=device)

#             with autocast():
#                 y_pred = tram_model(int_input=int_input, shift_input=shift_list)
#                 loss = contram_nll(y_pred, y, min_max=min_max)

#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             if use_scheduler:
#                 scheduler.step()

#             train_loss += loss.item()

#         train_time = time.time() - train_start
#         avg_train_loss = train_loss / len(train_loader)
#         train_loss_hist.append(avg_train_loss)

#         ##### Validation #####
#         val_start = time.time()
#         val_loss = 0.0
#         tram_model.eval()

#         with torch.no_grad():
#             for x, y in val_loader:
#                 y = y.to(device)
#                 int_input, shift_list = preprocess_inputs(x, device=device)

#                 with autocast():
#                     y_pred = tram_model(int_input=int_input, shift_input=shift_list)
#                     loss = contram_nll(y_pred, y, min_max=min_max)

#                 val_loss += loss.item()

#         val_time = time.time() - val_start
#         avg_val_loss = val_loss / len(val_loader)
#         val_loss_hist.append(avg_val_loss)

#         ##### Saving #####
#         save_start = time.time()
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             torch.save(tram_model.state_dict(), MODEL_PATH)
#             print("Saved new best model.")

#         torch.save(tram_model.state_dict(), LAST_MODEL_PATH)

#         with open(TRAIN_HIST_PATH, 'w') as f:
#             json.dump(train_loss_hist, f)
#         with open(VAL_HIST_PATH, 'w') as f:
#             json.dump(val_loss_hist, f)

#         save_time = time.time() - save_start
#         epoch_total = time.time() - epoch_start

#         ##### Epoch Summary #####
#         print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
#         print(f"  [Train: {train_time:.2f}s | Val: {val_time:.2f}s | Save: {save_time:.2f}s | Total: {epoch_total:.2f}s]")


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
    


