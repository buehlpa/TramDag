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
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

import torch
from torch.optim import Adam
from joblib import Parallel, delayed

from statsmodels.graphics.gofplots import qqplot_2samples
from scipy.stats import logistic, probplot

from .utils.tram_model_helpers import train_val_loop,evaluate_tramdag_model, get_fully_specified_tram_model , model_train_val_paths ,ordered_parents
from .utils.tram_data_helpers import create_latent_df_for_full_dag, sample_full_dag,is_outcome_modelled_ordinal,is_outcome_modelled_continous,show_hdag_for_single_source_node_continous,show_hdag_continous
from .utils.continous_helpers import transform_intercepts_continous
from .utils.ordinal_helpers import transform_intercepts_ordinal

from .models.tram_models import SimpleIntercept

from .TramDagConfig import TramDagConfig
from .TramDagDataset import TramDagDataset



## TODO from x via h^-1 to latent z function for ordinal
## TODO complex shifts fucniton display
## TODO documentation with docusaurus

# BUG if cfg is passed directy and not loaded with TD config cfg.update causes an issue 

class TramDagModel:
    
    # ---- defaults used at construction time ----
    DEFAULTS_CONFIG = {
        "set_initial_weights": False,
        "debug":False,
        "verbose": False,
        "device":'auto',
        "initial_data":None,
    }

    # ---- defaults used at fit() time ----
    DEFAULTS_FIT = {
        "epochs": 100,
        "train_list": None,
        "callbacks": None,
        "learning_rate": 0.01,
        "device": "auto",
        "optimizers": None,
        "schedulers": None,
        "use_scheduler": False,
        "save_linear_shifts": True,
        "save_simple_intercepts": True,
        "debug":False,
        "verbose": True,
        "train_mode": "sequential",  # or "parallel"
        "return_history": False,
        "overwrite_inital_weights": True,
        "num_workers" : 0,
        "persistent_workers" : True,
        "prefetch_factor" : 0,
        "batch_size":1000,
        
    }

    def __init__(self):
        """Empty init. Use classmethods like .from_config()."""
        self.debug = False
        self.verbose = False
        self.device = 'auto'
        pass

    @staticmethod
    def get_device(settings):
        device_arg = settings.get("device", "auto")
        if device_arg == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_str = device_arg
        return device_str

    def _validate_kwargs(self, kwargs: dict, defaults_attr: str = "DEFAULTS_FIT", context: str = None):
        """Validate keyword arguments against a defaults attribute of the class.

        Args:
            kwargs: Dictionary of keyword arguments to validate.
            defaults_attr: Name of the attribute containing allowed defaults (default: "DEFAULTS").
            context: Optional string identifying the caller for clearer error messages.

        Raises:
            ValueError: If any keys in kwargs are not listed in the defaults attribute.
        """
        defaults = getattr(self, defaults_attr, None)
        if defaults is None:
            raise AttributeError(f"{self.__class__.__name__} has no attribute '{defaults_attr}'")

        unknown = set(kwargs) - set(defaults)
        if unknown:
            prefix = f"[{context}] " if context else ""
            raise ValueError(f"{prefix}Unknown parameter(s): {', '.join(sorted(unknown))}")
            
    ## CREATE A TRAMDADMODEL
    @classmethod
    def from_config(cls, cfg, **kwargs):
        """
        Construct a `TramDagModel` instance by building one TramModel per node 
        defined in the configuration file.

        Each node model is initialized with its own set of settings, which can 
        either be shared across all nodes (scalar arguments) or individually 
        specified for each node (dictionary arguments keyed by node name).

        Parameters
        ----------
        cfg : object
            Configuration object containing at least:
                - `conf_dict["nodes"]`: mapping of node names to configurations.
                - `conf_dict["PATHS"]["EXPERIMENT_DIR"]`: directory for saving model states.

        **kwargs : dict
            Optional keyword arguments for customizing model instantiation.
            Each argument can be either:
                - scalar: applied to all nodes.
                - dict: mapping `{node_name: value}` to apply per node.

            Recognized keys include:
                - `device`: "auto", "cpu", or "cuda" (default: "auto").
                - `debug`: bool, print diagnostic information (default: False).
                - `verbose`: bool, toggle verbose output (default: False).
                - `overwrite_initial_weights`: bool, whether to overwrite saved model weights (default: True).
                - Other model-specific parameters defined in `DEFAULTS_CONFIG`.

        Returns
        -------
        self : TramDagModel
            An initialized TramDagModel instance with one TramModel per node.

        Raises
        ------
        ValueError
            If a dict-typed argument does not include all nodes from 
            `cfg.conf_dict["nodes"].keys()`.

        Notes
        -----
        - For each node, the method:
            1. Resolves scalar and dict arguments into per-node settings.
            2. Builds a TramModel via `get_fully_specified_tram_model()`.
            3. Saves the initialized model state to 
               `<EXPERIMENT_DIR>/<node>/initial_model.pt`.
        - If `overwrite_initial_weights` is False, existing model states will not be overwritten.
        - Device is automatically set to "cuda" if available unless explicitly specified.
        """
        
        self = cls()
        self.cfg = cfg
        self.cfg.update()# call to ensure latest config is loaded
        
        self.nodes_dict = self.cfg.conf_dict["nodes"] 

        self._validate_kwargs(kwargs, defaults_attr='DEFAULTS_CONFIG', context="from_config")

        # update defaults with kwargs
        settings = dict(cls.DEFAULTS_CONFIG)
        settings.update(kwargs)

        # resolve device
        device_arg = settings.get("device", "auto")
        if device_arg == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_str = device_arg
        self.device = torch.device(device_str)

        # set flags on the instance so they are accessible later
        self.debug = settings.get("debug", False)
        self.verbose = settings.get("verbose", False)

        if  self.debug:
            print(f"[DEBUG] TramDagModel using device: {self.device}")
            
        # initialize settings storage
        self.settings = {k: {} for k in settings.keys()}

        # validate dict-typed args
        for k, v in settings.items():
            if isinstance(v, dict):
                expected = set(self.nodes_dict.keys())
                given = set(v.keys())
                if expected != given:
                    raise ValueError(
                        f"[ERROR] the provided argument '{k}' keys are not same as in cfg.conf_dict['nodes'].keys().\n"
                        f"Expected: {expected}, but got: {given}\n"
                        f"Please provide values for all variables.")

        # build one model per node
        self.models = {}
        for node in self.nodes_dict.keys():
            per_node_kwargs = {}
            for k, v in settings.items():
                resolved = v[node] if isinstance(v, dict) else v
                per_node_kwargs[k] = resolved
                self.settings[k][node] = resolved
            if self.debug:
                print(f"\n[INFO] Building model for node '{node}' with settings: {per_node_kwargs}")
            self.models[node] = get_fully_specified_tram_model(
                node=node,
                configuration_dict=self.cfg.conf_dict,
                **per_node_kwargs)
            
            try:
                EXPERIMENT_DIR = self.cfg.conf_dict["PATHS"]["EXPERIMENT_DIR"]
                NODE_DIR = os.path.join(EXPERIMENT_DIR, f"{node}")
                os.makedirs(NODE_DIR, exist_ok=True)

                model_path = os.path.join(NODE_DIR, "initial_model.pt")
                overwrite = settings.get("overwrite_initial_weights", True)

                if overwrite or not os.path.exists(model_path):
                    torch.save(self.models[node].state_dict(), model_path)
                    if self.debug:
                        print(f"[DEBUG] Saved initial model state for node '{node}' to {model_path} (overwrite={overwrite})")
                else:
                    if self.debug:
                        print(f"[DEBUG] Skipped saving initial model for node '{node}' (already exists at {model_path})")
            except Exception as e:
                print(f"[ERROR] Could not save initial model state for node '{node}': {e}")
                            
        return self
    
    @classmethod
    def from_directory(cls, EXPERIMENT_DIR: str, device: str = "auto", debug: bool = False, verbose: bool = False):
        """
        Reconstruct a TramDagModel from an experiment directory.

        This loads:
        - The configuration file (config.json).
        - The minmax scaling file (min_max_scaling.json).
        - Initializes all per-node models (like from_config).

        Parameters
        ----------
        experiment_dir : str
            Path to the experiment directory containing `config.json` and `min_max_scaling.json`.
        device : str, optional
            Device string ("cpu", "cuda", or "auto"). Default is "auto".
        debug : bool, optional
            Enable debug printing. Default = False.
        verbose : bool, optional
            Enable info printing. Default = True.

        Returns
        -------
        TramDagModel
            A fully initialized TramDagModel with config and minmax loaded.
        """

        # --- load config file ---
        config_path = os.path.join(EXPERIMENT_DIR, "configuration.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"[ERROR] Config file not found at {config_path}")

        with open(config_path, "r") as f:
            cfg_dict = json.load(f)

        # Create TramConfig wrapper (adjust if your cfg is a dict already)
        cfg = TramDagConfig(cfg_dict)

        # --- build model from config ---
        self = cls.from_config(cfg, device=device, debug=debug, verbose=verbose, overwrite_initial_weights=False)

        # --- load minmax scaling ---
        minmax_path = os.path.join(EXPERIMENT_DIR, "min_max_scaling.json")
        if not os.path.exists(minmax_path):
            raise FileNotFoundError(f"[ERROR] MinMax file not found at {minmax_path}")

        with open(minmax_path, "r") as f:
            self.minmax_dict = json.load(f)

        if self.verbose or self.debug:
            print(f"[INFO] Loaded TramDagModel from {EXPERIMENT_DIR}")
            print(f"[INFO] Config loaded from {config_path}")
            print(f"[INFO] MinMax scaling loaded from {minmax_path}")

        return self

    def _ensure_dataset(self, data, is_val=False,**kwargs):
        """
        Ensure the input is converted to a TramDagDataset if needed.

        Parameters
        ----------
        data : pd.DataFrame, TramDagDataset, or None
            Input data to be converted or passed through.
        is_val : bool, default=False
            Whether the dataset is validation data (affects shuffle flag).

        Returns
        -------
        TramDagDataset or None
        """
        
        if isinstance(data, pd.DataFrame):
            return TramDagDataset.from_dataframe(data, self.cfg, shuffle=not is_val,**kwargs)
        elif isinstance(data, TramDagDataset):
            return data
        elif data is None:
            return None
        else:
            raise TypeError(
                f"[ERROR] data must be pd.DataFrame, TramDagDataset, or None, got {type(data)}"
            )

    def load_or_compute_minmax(self, td_train_data=None,use_existing=False, write=True):
        """
        Load an existing Min–Max scaling dictionary from disk or compute a new one 
        from the provided training dataset.

        Parameters
        ----------
        use_existing : bool, optional (default=False)
            If True, attempts to load an existing `min_max_scaling.json` file 
            from the experiment directory. Raises an error if the file is missing 
            or unreadable.

        write : bool, optional (default=True)
            If True, writes the computed Min–Max scaling dictionary to 
            `<EXPERIMENT_DIR>/min_max_scaling.json`.

        td_train_data : object, optional
            Training dataset used to compute scaling statistics. If not provided,
            the method will ensure or construct it via `_ensure_dataset(data=..., is_val=False)`.

        Behavior
        --------
        - If `use_existing=True`, loads the JSON file containing previously saved 
          min–max values and stores it in `self.minmax_dict`.
        - If `use_existing=False`, computes a new scaling dictionary using 
          `td_train_data.compute_scaling()` and stores the result in 
          `self.minmax_dict`.
        - Optionally writes the computed dictionary to disk.

        Side Effects
        -------------
        - Populates `self.minmax_dict` with scaling values.
        - Writes or loads the file `min_max_scaling.json` under 
          `<EXPERIMENT_DIR>`.
        - Prints diagnostic output if `self.debug` or `self.verbose` is True.

        Raises
        ------
        FileNotFoundError
            If `use_existing=True` but the min–max file does not exist.

        RuntimeError
            If an existing min–max file cannot be read or parsed.

        Notes
        -----
        The computed min–max dictionary is expected to contain scaling statistics 
        per feature, typically in the form:
            {
                "feature_name": {"min": float, "max": float},
                ...
            }
        """
        EXPERIMENT_DIR = self.cfg.conf_dict["PATHS"]["EXPERIMENT_DIR"]
        minmax_path = os.path.join(EXPERIMENT_DIR, "min_max_scaling.json")

        # laod exisitng if possible
        if use_existing:
            if not os.path.exists(minmax_path):
                raise FileNotFoundError(f"MinMax file not found: {minmax_path}")
            try:
                with open(minmax_path, 'r') as f:
                    self.minmax_dict = json.load(f)
                if self.debug or self.verbose:
                    print(f"[INFO] Loaded existing minmax dict from {minmax_path}")
                return
            except Exception as e:
                raise RuntimeError(f"Could not load existing minmax dict: {e}")

        # 
        if self.debug or self.verbose:
            print("[INFO] Computing new minmax dict from training data...")
            
        td_train_data=self._ensure_dataset( data=td_train_data, is_val=False)    
            
        self.minmax_dict = td_train_data.compute_scaling()

        if write:
            os.makedirs(EXPERIMENT_DIR, exist_ok=True)
            with open(minmax_path, 'w') as f:
                json.dump(self.minmax_dict, f, indent=4)
            if self.debug or self.verbose:
                print(f"[INFO] Saved new minmax dict to {minmax_path}")

    ## FIT METHODS
    @staticmethod
    def _fit_single_node(node, self_ref, settings, td_train_data, td_val_data, device_str):
        """
        Train a single node model (used by Joblib workers).
        Runs in a separate process, so all arguments must be picklable.
        """
        torch.set_num_threads(1)  # prevent thread oversubscription

        model = self_ref.models[node]

        # Resolve per-node settings
        def _resolve(key):
            val = settings[key]
            return val[node] if isinstance(val, dict) else val

        node_epochs = _resolve("epochs")
        node_lr = _resolve("learning_rate")
        node_debug = _resolve("debug")
        node_save_linear_shifts = _resolve("save_linear_shifts")
        save_simple_intercepts  = _resolve("save_simple_intercepts")
        node_verbose = _resolve("verbose")

        # Optimizer & scheduler
        if settings["optimizers"] and node in settings["optimizers"]:
            optimizer = settings["optimizers"][node]
        else:
            optimizer = Adam(model.parameters(), lr=node_lr)

        scheduler = settings["schedulers"].get(node, None) if settings["schedulers"] else None

        # Data loaders
        train_loader = td_train_data.loaders[node]
        val_loader = td_val_data.loaders[node] if td_val_data else None

        # Min-max scaling tensors
        min_vals = torch.tensor(self_ref.minmax_dict[node][0], dtype=torch.float32)
        max_vals = torch.tensor(self_ref.minmax_dict[node][1], dtype=torch.float32)
        min_max = torch.stack([min_vals, max_vals], dim=0)

        # Node directory
        try:
            EXPERIMENT_DIR = self_ref.cfg.conf_dict["PATHS"]["EXPERIMENT_DIR"]
            NODE_DIR = os.path.join(EXPERIMENT_DIR, f"{node}")
        except Exception:
            NODE_DIR = os.path.join("models", node)
            print("[WARNING] No log directory specified in config, saving to default location.")
        os.makedirs(NODE_DIR, exist_ok=True)

        if node_verbose:
            print(f"\n[INFO] Training node '{node}' for {node_epochs} epochs on {device_str} (pid={os.getpid()})")

        # --- train ---
        history = train_val_loop(
            node=node,
            target_nodes=self_ref.nodes_dict,
            NODE_DIR=NODE_DIR,
            tram_model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=node_epochs,
            optimizer=optimizer,
            use_scheduler=(scheduler is not None),
            scheduler=scheduler,
            save_linear_shifts=node_save_linear_shifts,
            save_simple_intercepts=save_simple_intercepts,
            verbose=node_verbose,
            device=torch.device(device_str),
            debug=node_debug,
            min_max=min_max)
        return node, history

    def fit(self, train_data, val_data=None, **kwargs):
        """
        Train TRAM models for all nodes defined in the DAG configuration.

        This method coordinates training across all node models, either sequentially
        or in parallel (CPU-only), handling dataset preparation, scaling, and 
        process-safe execution of node-level training loops.

        Parameters
        ----------
        train_data : pd.DataFrame or TramDagDataset
            Training dataset. Can be a pre-processed TramDagDataset or a raw DataFrame
            that will be internally converted using `_ensure_dataset()`.

        val_data : pd.DataFrame or TramDagDataset, optional
            Validation dataset, structured similarly to `train_data`.

        **kwargs : dict, optional
            Overrides or extends default training settings (`DEFAULTS_FIT`).
            Common keys include:
                - train_mode : {"sequential", "parallel"}, default="sequential"
                    "parallel" uses joblib-based multiprocessing (CPU only).
                - device : {"auto", "cpu", "cuda"}, default="auto"
                - num_workers : int, DataLoader workers (ignored in parallel mode)
                - prefetch_factor : int, DataLoader prefetch (ignored in parallel mode)
                - persistent_workers : bool, DataLoader persistence flag (ignored in parallel mode)
                - epochs : int, training epochs per node
                - learning_rate : float, optimizer learning rate
                - debug : bool, print diagnostic information
                - verbose : bool, print training progress
                - return_history : bool, if True returns training history
                - train_list : list[str], subset of nodes to train (default: all nodes)

        Returns
        -------
        results : dict, optional
            If `return_history=True`, returns a dictionary mapping each node name
            to its training history (as returned by `_fit_single_node()`).

        Behavior
        --------
        - Validates and merges keyword arguments with `DEFAULTS_FIT`.
        - Determines execution device via `get_device()`.
        - Prepares `TramDagDataset` objects for training and validation.
        - Computes or loads min–max normalization parameters.
        - Executes training:
            * **Sequential mode** — runs node models one after another (default and required for CUDA).
            * **Parallel mode** — spawns CPU workers using joblib for independent node training.
        - Each node’s results are aggregated into a dictionary if requested.

        Safety Logic
        -------------
        - In parallel mode, DataLoader multiprocessing parameters
          (`num_workers`, `prefetch_factor`, `persistent_workers`) are disabled
          to prevent nested multiprocessing errors.
        - GPU devices automatically force sequential mode.

        Raises
        ------
        ValueError
            If `train_mode` is not "sequential" or "parallel".

        Notes
        -----
        - Each node model’s training artifacts and checkpoints are stored in:
              `<EXPERIMENT_DIR>/<node>/`
        - For parallel mode, the number of workers is automatically limited to 
          half of available CPU cores.
        - Verbose and debug flags control console logging granularity.
        """

        self._validate_kwargs(kwargs, defaults_attr='DEFAULTS_FIT', context="fit")
        
        # --- merge defaults ---
        settings = dict(self.DEFAULTS_FIT)
        settings.update(kwargs)
        
        
        self.debug = settings.get("debug", False)
        self.verbose = settings.get("verbose", False)

        # --- resolve device ---
        device_str=self.get_device(settings)
        self.device = torch.device(device_str)

        # --- training mode ---
        train_mode = settings.get("train_mode", "sequential").lower()
        if train_mode not in ("sequential", "parallel"):
            raise ValueError("train_mode must be 'sequential' or 'parallel'")

        # --- DataLoader safety logic ---
        if train_mode == "parallel":
            # if user passed loader paralleling params, warn and override
            for flag in ("num_workers", "persistent_workers", "prefetch_factor"):
                if flag in kwargs:
                    print(f"[WARNING] '{flag}' is ignored in parallel mode "
                        f"(disabled to prevent nested multiprocessing).")
            # disable unsafe loader multiprocessing options
            settings["num_workers"] = 0
            settings["persistent_workers"] = False
            settings["prefetch_factor"] = None
        else:
            # sequential mode → respect user DataLoader settings
            if self.debug:
                print("[DEBUG] Sequential mode: using DataLoader kwargs as provided.")

        # --- which nodes to train ---
        train_list = settings.get("train_list") or list(self.models.keys())


        # --- dataset prep (receives adjusted settings) ---
        td_train_data = self._ensure_dataset(train_data, is_val=False, **settings)
        td_val_data = self._ensure_dataset(val_data, is_val=True, **settings)

        # --- normalization ---
        self.load_or_compute_minmax(use_existing=False, write=True, td_train_data=td_train_data)

        # --- print header ---
        if self.verbose or self.debug:
            print(f"[INFO] Training {len(train_list)} nodes ({train_mode}) on {device_str}")

        # ======================================================================
        # Sequential mode  safe for GPU or debugging)
        # ======================================================================
        if train_mode == "sequential" or "cuda" in device_str:
            if "cuda" in device_str and train_mode == "parallel":
                print("[WARNING] GPU device detected — forcing sequential mode.")
            results = {}
            for node in train_list:
                node, history = self._fit_single_node(
                    node, self, settings, td_train_data, td_val_data, device_str
                )
                results[node] = history
        

        # ======================================================================
        # parallel mode (CPU only)
        # ======================================================================
        if train_mode == "parallel":

            n_jobs = min(len(train_list), os.cpu_count() // 2 or 1)
            if self.verbose or self.debug:
                print(f"[INFO] Using {n_jobs} CPU workers for parallel node training")
            parallel_outputs = Parallel(
                n_jobs=n_jobs,
                backend="loky",#loky, multiprocessing
                verbose=10,
                prefer="processes"
            )(delayed(self._fit_single_node)(node, self, settings, td_train_data, td_val_data, device_str) for node in train_list )

            results = {node: hist for node, hist in parallel_outputs}
        
        if settings.get("return_history", False):
            return results

    ## FIT-DIAGNOSTICS
    def loss_history(self):
        """
        Load training and validation loss histories for all nodes.

        Looks for JSON files in:
            EXPERIMENT_DIR/{node}/train_loss_hist.json
            EXPERIMENT_DIR/{node}/val_loss_hist.json

        Returns
        -------
        dict
            {node: {"train": train_hist, "validation": val_hist}}
        """
        try:
            EXPERIMENT_DIR = self.cfg.conf_dict["PATHS"]["EXPERIMENT_DIR"]
        except KeyError:
            raise ValueError(
                "[ERROR] Missing 'EXPERIMENT_DIR' in cfg.conf_dict['PATHS']. "
                "History retrieval requires experiment logs."
            )

        all_histories = {}
        for node in self.nodes_dict.keys():
            node_dir = os.path.join(EXPERIMENT_DIR, node)
            train_path = os.path.join(node_dir, "train_loss_hist.json")
            val_path = os.path.join(node_dir, "val_loss_hist.json")

            node_hist = {}

            # --- load train history ---
            if os.path.exists(train_path):
                try:
                    with open(train_path, "r") as f:
                        node_hist["train"] = json.load(f)
                except Exception as e:
                    print(f"[WARNING] Could not load {train_path}: {e}")
                    node_hist["train"] = None
            else:
                node_hist["train"] = None

            # --- load val history ---
            if os.path.exists(val_path):
                try:
                    with open(val_path, "r") as f:
                        node_hist["validation"] = json.load(f)
                except Exception as e:
                    print(f"[WARNING] Could not load {val_path}: {e}")
                    node_hist["validation"] = None
            else:
                node_hist["validation"] = None

            all_histories[node] = node_hist

        if self.verbose or self.debug:
            print(f"[INFO] Loaded training/validation histories for {len(all_histories)} nodes.")

        return all_histories

    def linear_shift_history(self):
        """
        Load linear shift histories for all nodes.

        Returns
        -------
        dict
            Mapping of node names to their linear shift history DataFrames.
        """
        histories = {}
        try:
            EXPERIMENT_DIR = self.cfg.conf_dict["PATHS"]["EXPERIMENT_DIR"]
        except KeyError:
            raise ValueError(
                "[ERROR] Missing 'EXPERIMENT_DIR' in cfg.conf_dict['PATHS']. "
                "Cannot load histories without experiment directory."
            )

        for node in self.nodes_dict.keys():
            node_dir = os.path.join(EXPERIMENT_DIR, node)
            history_path = os.path.join(node_dir, "linear_shifts_all_epochs.json")
            if os.path.exists(history_path):
                histories[node] = pd.read_json(history_path)
            else:
                print(f"[WARNING] No linear shift history found for node '{node}' at {history_path}")
        return histories

    def simple_intercept_history(self):
        """
        Load simple_intercept_history histories for all nodes.

        Returns
        -------
        dict
            Mapping of node names to their simple_intercept_history history DataFrames.
        """
        histories = {}
        try:
            EXPERIMENT_DIR = self.cfg.conf_dict["PATHS"]["EXPERIMENT_DIR"]
        except KeyError:
            raise ValueError(
                "[ERROR] Missing 'EXPERIMENT_DIR' in cfg.conf_dict['PATHS']. "
                "Cannot load histories without experiment directory."
            )

        for node in self.nodes_dict.keys():
            node_dir = os.path.join(EXPERIMENT_DIR, node)
            history_path = os.path.join(node_dir, "simple_intercepts_all_epochs.json")
            if os.path.exists(history_path):
                histories[node] = pd.read_json(history_path)
            else:
                print(f"[WARNING] No simple intercept history found for node '{node}' at {history_path}")
        return histories

    def get_latent(self, df, verbose=False):
        """
        Compute latent representations for the full DAG.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with columns for each node.
        verbose : bool, optional
            If True, prints [INFO] statements during processing.

        Returns
        -------
        pd.DataFrame
            DataFrame with latent variables for each node. Columns are
            [node, f"{node}_U"] for each continuous target.
        """
        try:
            EXPERIMENT_DIR = self.cfg.conf_dict["PATHS"]["EXPERIMENT_DIR"]
        except KeyError:
            raise ValueError(
                "[ERROR] Missing 'EXPERIMENT_DIR' in cfg.conf_dict['PATHS']. "
                "Latent extraction requires trained model checkpoints."
            )

        # ensure minmax_dict is available
        if not hasattr(self, "minmax_dict"):
            raise ValueError(
                "[ERROR] minmax_dict not found in the TramDagModel instance. "
                "Either call .load_or_compute_minmax(td_train_data=train_df) or .fit() first."
            )

        all_latents_df = create_latent_df_for_full_dag(
            configuration_dict=self.cfg.conf_dict,
            EXPERIMENT_DIR=EXPERIMENT_DIR,
            df=df,
            verbose=verbose,
            min_max_dict=self.minmax_dict,
        )

        return all_latents_df

    
    ## PLOTTING FIT-DIAGNOSTICS
    
    def plot_loss_history(self, variable: str = None):
        """
        Plot training and validation loss histories.

        Parameters
        ----------
        variable : str, optional
            If given, plot only this node's loss_history.
            If None, plot all nodes together.
        """

        histories = self.loss_history()
        if not histories:
            print("[WARNING] No loss histories found.")
            return

        # Select which nodes to plot
        if variable is not None:
            if variable not in histories:
                raise ValueError(f"[ERROR] Node '{variable}' not found in histories.")
            nodes_to_plot = [variable]
        else:
            nodes_to_plot = list(histories.keys())

        # Filter out nodes with no valid history
        nodes_to_plot = [
            n for n in nodes_to_plot
            if histories[n].get("train") is not None and len(histories[n]["train"]) > 0
            and histories[n].get("validation") is not None and len(histories[n]["validation"]) > 0
        ]

        if not nodes_to_plot:
            print("[WARNING] No valid histories found to plot.")
            return

        plt.figure(figsize=(14, 12))

        # --- Full history (top plot) ---
        plt.subplot(2, 1, 1)
        for node in nodes_to_plot:
            node_hist = histories[node]
            train_hist, val_hist = node_hist["train"], node_hist["validation"]

            epochs = range(1, len(train_hist) + 1)
            plt.plot(epochs, train_hist, label=f"{node} - train", linestyle="--")
            plt.plot(epochs, val_hist, label=f"{node} - val")

        plt.title("Training and Validation NLL - Full History")
        plt.xlabel("Epoch")
        plt.ylabel("NLL")
        plt.legend()
        plt.grid(True)

        # --- Last 10% of epochs (bottom plot) ---
        plt.subplot(2, 1, 2)
        for node in nodes_to_plot:
            node_hist = histories[node]
            train_hist, val_hist = node_hist["train"], node_hist["validation"]

            total_epochs = len(train_hist)
            start_idx = total_epochs - 1 if total_epochs < 5 else int(total_epochs * 0.9)

            epochs = range(start_idx + 1, total_epochs + 1)
            plt.plot(epochs, train_hist[start_idx:], label=f"{node} - train", linestyle="--")
            plt.plot(epochs, val_hist[start_idx:], label=f"{node} - val")

        plt.title("Training and Validation NLL - Last 10% of Epochs (or Last Epoch if <5)")
        plt.xlabel("Epoch")
        plt.ylabel("NLL")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_linear_shift_history(self, data_dict=None, node=None, ref_lines=None):
        """
        Plot evolution of shift terms for one or all nodes.

        Args:
            data_dict (dict, optional): {node_name: DataFrame} with shift weights.
                                        If None, uses self.linear_shift_history().
            node (str, optional): If given, plot only that node; otherwise plot all.
            ref_lines (dict, optional): {node_name: [y_values]} for reference lines.
        """


        if data_dict is None:
            data_dict = self.linear_shift_history()
            if data_dict is None:
                raise ValueError("No shift history data provided or stored in the class.")

        nodes = [node] if node else list(data_dict.keys())

        for n in nodes:
            df = data_dict[n].copy()

            # Flatten nested lists or list-like cells
            def flatten(x):
                if isinstance(x, list):
                    if len(x) == 0:
                        return np.nan
                    if all(isinstance(i, (int, float)) for i in x):
                        return np.mean(x)  # average simple list
                    if all(isinstance(i, list) for i in x):
                        # nested list -> flatten inner and average
                        flat = [v for sub in x for v in (sub if isinstance(sub, list) else [sub])]
                        return np.mean(flat) if flat else np.nan
                    return x[0] if len(x) == 1 else np.nan
                return x

            df = df.applymap(flatten)

            # Ensure numeric columns
            df = df.apply(pd.to_numeric, errors='coerce')

            # Convert epoch labels to numeric
            df.columns = [
                int(c.replace("epoch_", "")) if isinstance(c, str) and c.startswith("epoch_") else c
                for c in df.columns
            ]
            df = df.reindex(sorted(df.columns), axis=1)

            plt.figure(figsize=(10, 6))
            for idx in df.index:
                plt.plot(df.columns, df.loc[idx], lw=1.4, label=f"shift_{idx}")

            if ref_lines and n in ref_lines:
                for v in ref_lines[n]:
                    plt.axhline(y=v, color="k", linestyle="--", lw=1.0)
                    plt.text(df.columns[-1], v, f"{n}: {v}", va="bottom", ha="right", fontsize=8)

            plt.xlabel("Epoch")
            plt.ylabel("Shift Value")
            plt.title(f"Shift Term History — Node: {n}")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
            plt.tight_layout()
            plt.show()

    def plot_simple_intercepts_history(self, data_dict=None, node=None,ref_lines=None):
        """
        Plot evolution of simple intercept weights for one or all nodes.

        Args:
            data_dict (dict, optional): {node_name: DataFrame} with intercept weights.
                                        If None, uses self.simple_intercept_history().
            node (str, optional): If given, plot only that node; otherwise plot all.
        """
        if data_dict is None:
            data_dict = self.simple_intercept_history()
            if data_dict is None:
                raise ValueError("No intercept history data provided or stored in the class.")

        nodes = [node] if node else list(data_dict.keys())

        for n in nodes:
            df = data_dict[n].copy()

            def extract_scalar(x):
                if isinstance(x, list):
                    while isinstance(x, list) and len(x) > 0:
                        x = x[0]
                return float(x) if isinstance(x, (int, float, np.floating)) else np.nan

            df = df.applymap(extract_scalar)

            # Convert epoch labels → numeric
            df.columns = [
                int(c.replace("epoch_", "")) if isinstance(c, str) and c.startswith("epoch_") else c
                for c in df.columns
            ]
            df = df.reindex(sorted(df.columns), axis=1)

            plt.figure(figsize=(10, 6))
            for idx in df.index:
                plt.plot(df.columns, df.loc[idx], lw=1.4, label=f"theta_{idx}")
            
            if ref_lines and n in ref_lines:
                for v in ref_lines[n]:
                    plt.axhline(y=v, color="k", linestyle="--", lw=1.0)
                    plt.text(df.columns[-1], v, f"{n}: {v}", va="bottom", ha="right", fontsize=8)
                
            plt.xlabel("Epoch")
            plt.ylabel("Intercept Weight")
            plt.title(f"Simple Intercept Evolution — Node: {n}")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
            plt.tight_layout()
            plt.show()

    def plot_latents(self, df, variable: str = None, confidence: float = 0.95, simulations: int = 1000):
        """
        Plot latent U distributions for one node or all nodes.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with the raw data.
        variable : str, optional
            If given, only plot for this node. If None, plot all nodes.
        confidence : float, optional
            Confidence level for QQ plot bands. Default = 0.95.
        simulations : int, optional
            Number of simulations for QQ bands. Default = 1000.
        """

        # Compute latent representations
        latents_df = self.get_latent(df)

        # Select nodes
        nodes = [variable] if variable is not None else self.nodes_dict.keys()

        for node in nodes:
            if f"{node}_U" not in latents_df.columns:
                print(f"[WARNING] No latent found for node {node}, skipping.")
                continue

            sample = latents_df[f"{node}_U"].values

            # --- Create plots ---
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))

            # Histogram
            axs[0].hist(sample, bins=50, color="steelblue", alpha=0.7)
            axs[0].set_title(f"Latent Histogram ({node})")
            axs[0].set_xlabel("U")
            axs[0].set_ylabel("Frequency")

            # QQ Plot with confidence bands
            probplot(sample, dist="logistic", plot=axs[1])
            self._add_r_style_confidence_bands(axs[1], sample, dist=logistic,confidence=confidence, simulations=simulations)
            axs[1].set_title(f"Latent QQ Plot ({node})")

            plt.suptitle(f"Latent Diagnostics for Node: {node}", fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    def plot_hdag(self,df,variables=None, plot_n_rows=1,**kwargs):
        
        """
        Visualize the transformation function h() for specified nodes or all nodes

        Parameters
        ----------
        df : pandas.DataFrame
            Input data containing node values or model predictions. 
            Can include multiple rows (samples). Only the first `plot_n_rows` 
            will be visualized.

        variables : list of str, optional
            List of variable names (nodes) to visualize. 
            If None, all nodes present in `self.models.keys()` will be plotted.

        plot_n_rows : int, default=5
            Maximum number of rows from `df` to visualize. 
            If `len(df) > plot_n_rows`, a warning is printed and the plot is truncated.

        Behavior
        --------
        For each specified node:
        - If the node represents a continuous outcome (as determined by 
          `is_outcome_modelled_continous(node, self.nodes_dict)`), 
          the function calls `show_hdag_continous()` to plot the model results 
          using configuration parameters from `self.cfg.conf_dict` and 
          scaling information from `self.minmax_dict`.
        - If the node is not continuous, a warning message is printed.

        Notes
        -----
        - This method currently supports only continuous nodes.
        - Visualization is device-aware (uses `self.device`).
        - Intended for inspecting model behavior across multiple samples.

        Warnings
        --------
        Prints a warning if:
        - `df` contains more rows than `plot_n_rows`.
        - A node is not continuous and therefore not plotted.
        """
        

        if len(df)> 1:
            print("[WARNING] len(df)>1, set: plot_n_rows accordingly")
        
        variables_list=variables if variables is not None else list(self.models.keys())
        for node in variables_list:
            if is_outcome_modelled_continous(node, self.nodes_dict):
                show_hdag_continous(df,node=node,configuration_dict=self.cfg.conf_dict,minmax_dict=self.minmax_dict,device=self.device,plot_n_rows=plot_n_rows,**kwargs)
            else:
                print(f"[WARNING] Node {node} is not continuous, not implemented yet")
    
    @staticmethod
    def _add_r_style_confidence_bands(ax, sample, dist, confidence=0.95, simulations=1000):
        """
        Adds confidence bands to a QQ plot using simulation under the null hypothesis.
        """
        
        n = len(sample)
        if n == 0:
            return

        quantiles = np.linspace(0, 1, n, endpoint=False) + 0.5 / n
        theo_q = dist.ppf(quantiles)

        # Simulate order statistics from the theoretical distribution
        sim_data = dist.rvs(size=(simulations, n))
        sim_order_stats = np.sort(sim_data, axis=1)

        # Confidence bands
        lower = np.percentile(sim_order_stats, 100 * (1 - confidence) / 2, axis=0)
        upper = np.percentile(sim_order_stats, 100 * (1 + confidence) / 2, axis=0)

        # Sort empirical sample
        sample_sorted = np.sort(sample)

        # Re-draw points and CI (overwrite probplot defaults)
        ax.clear()
        ax.plot(theo_q, sample_sorted, 'o', markersize=3, alpha=0.6, label="Empirical Q-Q")
        ax.plot(theo_q, theo_q, 'b--', label="y = x")
        ax.fill_between(theo_q, lower, upper, color='gray', alpha=0.3,
                        label=f'{int(confidence*100)}% CI')
        ax.legend()
    
    ## SAMPLING METHODS
    def sample(
        self,
        do_interventions: dict = None,
        predefined_latent_samples_df: pd.DataFrame = None,
        **kwargs,
    ):
        """
        Sample from the DAG using trained TRAM models.

        Parameters
        ----------
        do_interventions : dict, optional
            Mapping of node names to fixed values. Example: {'x1': 1.0}.
        predefined_latent_samples_df : pd.DataFrame, optional
            DataFrame with predefined latent U's. Must contain columns "{node}_U".
        kwargs : dict
            Overrides for default settings (number_of_samples, batch_size, device, etc.).

        Returns
        -------
        sampled_by_node : dict
            Mapping {node: tensor of sampled values}.
        latents_by_node : dict
            Mapping {node: tensor of latent U's used}.
        """

        try:
            EXPERIMENT_DIR = self.cfg.conf_dict["PATHS"]["EXPERIMENT_DIR"]
        except KeyError:
            raise ValueError(
                "[ERROR] Missing 'EXPERIMENT_DIR' in cfg.conf_dict['PATHS']. "
                "Sampling requires trained model checkpoints."
            )

        # ---- defaults ----
        settings = {
            "number_of_samples": 10_000,
            "batch_size": 32,
            "delete_all_previously_sampled": True,
            "verbose": self.verbose if hasattr(self, "verbose") else False,
            "debug": self.debug if hasattr(self, "debug") else False,
            "device": self.device.type if hasattr(self, "device") else "auto",
            "use_initial_weights_for_sampling": False,
            
        }
        
        # TODO adjust validation
        # self._validate_kwargs( kwargs, defaults_attr= "settings", context="sample")
        
        settings.update(kwargs)

        
        if not hasattr(self, "minmax_dict"):
            raise RuntimeError(
                "[ERROR] minmax_dict not found. You must call .fit() or .load_or_compute_minmax() "
                "before sampling, so scaling info is available."
                )
            
        # ---- resolve device ----
        device_str=self.get_device(settings)
        self.device = torch.device(device_str)


        if self.debug or settings["debug"]:
            print(f"[DEBUG] sample(): device: {self.device}")

        # ---- perform sampling ----
        sampled_by_node, latents_by_node = sample_full_dag(
            configuration_dict=self.cfg.conf_dict,
            EXPERIMENT_DIR=EXPERIMENT_DIR,
            device=self.device,
            do_interventions=do_interventions or {},
            predefined_latent_samples_df=predefined_latent_samples_df,
            number_of_samples=settings["number_of_samples"],
            batch_size=settings["batch_size"],
            delete_all_previously_sampled=settings["delete_all_previously_sampled"],
            verbose=settings["verbose"],
            debug=settings["debug"],
            minmax_dict=self.minmax_dict,
            use_initial_weights_for_sampling=settings["use_initial_weights_for_sampling"]
        )

        return sampled_by_node, latents_by_node

    def load_sampled_and_latents(self, EXPERIMENT_DIR: str = None, nodes: list = None):
        """
        Load 'sampled.pt' and 'latents.pt' tensors for each node.

        Parameters
        ----------
        EXPERIMENT_DIR : str, optional
            Path to the experiment directory. If not provided, uses
            self.cfg.conf_dict["PATHS"]["EXPERIMENT_DIR"].
        nodes : list, optional
            List of node names to load. If not provided, loads all nodes in self.nodes_dict.

        Returns
        -------
        sampled_by_node : dict
            {node: sampled tensor (on CPU)}
        latents_by_node : dict
            {node: latent tensor (on CPU)}
        """
        # --- resolve paths and node list ---
        if EXPERIMENT_DIR is None:
            try:
                EXPERIMENT_DIR = self.cfg.conf_dict["PATHS"]["EXPERIMENT_DIR"]
            except (AttributeError, KeyError):
                raise ValueError(
                    "[ERROR] Could not resolve EXPERIMENT_DIR from cfg.conf_dict['PATHS']. "
                    "Please provide EXPERIMENT_DIR explicitly."
                )

        if nodes is None:
            if hasattr(self, "nodes_dict"):
                nodes = list(self.nodes_dict.keys())
            else:
                raise ValueError(
                    "[ERROR] No node list found. Please provide `nodes` or initialize model with a config."
                )

        # --- load tensors ---
        sampled_by_node = {}
        latents_by_node = {}

        for node in nodes:
            node_dir = os.path.join(EXPERIMENT_DIR, f"{node}")
            sampling_dir = os.path.join(node_dir, "sampling")

            sampled_path = os.path.join(sampling_dir, "sampled.pt")
            latents_path = os.path.join(sampling_dir, "latents.pt")

            if not os.path.exists(sampled_path) or not os.path.exists(latents_path):
                print(f"[WARNING] Missing files for node '{node}' — skipping.")
                continue

            try:
                sampled = torch.load(sampled_path, map_location="cpu")
                latent_sample = torch.load(latents_path, map_location="cpu")
            except Exception as e:
                print(f"[ERROR] Could not load sampling files for node '{node}': {e}")
                continue

            sampled_by_node[node] = sampled.detach().cpu()
            latents_by_node[node] = latent_sample.detach().cpu()

        if self.verbose or self.debug:
            print(f"[INFO] Loaded sampled and latent tensors for {len(sampled_by_node)} nodes from {EXPERIMENT_DIR}")

        return sampled_by_node, latents_by_node

    def plot_samples_vs_true(
        self,
        df,
        sampled: dict = None,
        variable: list = None,
        bins: int = 100,
        hist_true_color: str = "blue",
        hist_est_color: str = "orange",
        figsize: tuple = (14, 5),
        
    ):
        """
        Compare true vs sampled distributions for each node.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with true observed values.
        bins : int, optional
            Number of bins for histograms (continuous case). Default = 100.
        hist_true_color : str, optional
            Color for true data histogram/bar. Default = "blue".
        hist_est_color : str, optional
            Color for sampled data histogram/bar. Default = "orange".
        figsize : tuple, optional
            Figure size for each node. Default = (14, 5).
        sampled : dict, optional
            Optional dictionary with preloaded samples per node.
            If provided, no loading from disk is performed.
        """

        target_nodes = self.cfg.conf_dict["nodes"]
        experiment_dir = self.cfg.conf_dict["PATHS"]["EXPERIMENT_DIR"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        plot_list=variable if variable is not None else target_nodes

        for node in plot_list:
            # Load sampled data
            if sampled is not None and node in sampled:
                sdata = sampled[node]
                if isinstance(sdata, torch.Tensor):
                    sampled_vals = sdata.detach().cpu().numpy()
                else:
                    sampled_vals = np.asarray(sdata)
            else:
                sample_path = os.path.join(experiment_dir, f"{node}/sampling/sampled.pt")
                if not os.path.isfile(sample_path):
                    print(f"[WARNING] skip {node}: {sample_path} not found.")
                    continue

                try:
                    sampled_vals = torch.load(sample_path, map_location=device).cpu().numpy()
                except Exception as e:
                    print(f"[ERROR] Could not load {sample_path}: {e}")
                    continue

            sampled_vals = sampled_vals[np.isfinite(sampled_vals)]

            if node not in df.columns:
                print(f"[WARNING] skip {node}: column not found in DataFrame.")
                continue

            true_vals = df[node].dropna().values
            true_vals = true_vals[np.isfinite(true_vals)]

            if sampled_vals.size == 0 or true_vals.size == 0:
                print(f"[WARNING] skip {node}: empty array after NaN/Inf removal.")
                continue

            fig, axs = plt.subplots(1, 2, figsize=figsize)

            if is_outcome_modelled_continous(node, target_nodes):
                axs[0].hist(
                    true_vals,
                    bins=bins,
                    density=True,
                    alpha=0.6,
                    color=hist_true_color,
                    label=f"True {node}",
                )
                axs[0].hist(
                    sampled_vals,
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

                qqplot_2samples(true_vals, sampled_vals, line="45", ax=axs[1])
                axs[1].set_xlabel("True quantiles")
                axs[1].set_ylabel("Sampled quantiles")
                axs[1].set_title(f"QQ plot for {node}")
                axs[1].grid(True, ls="--", alpha=0.4)

            elif is_outcome_modelled_ordinal(node, target_nodes):
                unique_vals = np.union1d(np.unique(true_vals), np.unique(sampled_vals))
                unique_vals = np.sort(unique_vals)

                true_counts = np.array([(true_vals == val).sum() for val in unique_vals])
                sampled_counts = np.array([(sampled_vals == val).sum() for val in unique_vals])

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
                unique_vals = np.union1d(np.unique(true_vals), np.unique(sampled_vals))
                unique_vals = sorted(unique_vals, key=str)

                true_counts = np.array([(true_vals == val).sum() for val in unique_vals])
                sampled_counts = np.array([(sampled_vals == val).sum() for val in unique_vals])

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

    ## SUMMARY METHODS
    def nll(self,data,variables=None):
        """
        Compute the Negative Log-Likelihood (NLL) for all or selected TRAM nodes.

        This function evaluates trained TRAM models for each specified variable (node) 
        on the provided dataset. It performs forward passes only—no training, no weight 
        updates—and returns the mean NLL per node.

        Parameters
        ----------
        data : object
            Input dataset or data source compatible with `_ensure_dataset`, containing 
            both inputs and targets for each node.
        variables : list[str], optional
            List of variable (node) names to evaluate. If None, all nodes in 
            `self.models` are evaluated.

        Returns
        -------
        dict[str, float]
            Dictionary mapping each node name to its average NLL value.

        Notes
        -----
        - Each model is evaluated independently on its respective DataLoader.
        - The normalization values (`min_max`) for each node are retrieved from 
          `self.minmax_dict[node]`.
        - The function uses `evaluate_tramdag_model()` for per-node evaluation.
        - Expected directory structure:
              `<EXPERIMENT_DIR>/<node>/`
          where each node directory contains the trained model.
        """

        td_data = self._ensure_dataset(data, is_val=True)  
        variables_list = variables if variables != None else list(self.models.keys())
        nll_dict = {}
        for node in variables_list:  
                min_vals = torch.tensor(self.minmax_dict[node][0], dtype=torch.float32)
                max_vals = torch.tensor(self.minmax_dict[node][1], dtype=torch.float32)
                min_max = torch.stack([min_vals, max_vals], dim=0)
                data_loader = td_data.loaders[node]
                model = self.models[node]
                EXPERIMENT_DIR = self.cfg.conf_dict["PATHS"]["EXPERIMENT_DIR"]
                NODE_DIR = os.path.join(EXPERIMENT_DIR, f"{node}")
                nll= evaluate_tramdag_model(node=node,
                                            target_nodes=self.nodes_dict,
                                            NODE_DIR=NODE_DIR,
                                            tram_model=model,
                                            data_loader=data_loader,
                                            min_max=min_max)
                nll_dict[node]=nll
        return nll_dict
    
    def get_train_val_nll(self, node: str, mode: str) -> tuple[float, float]:
        """
        Return the training and validation NLL for the given node and mode.
        mode ∈ {'best', 'last', 'init'}:
            - 'best' → NLL at the epoch with the lowest val NLL
            - 'last' → NLL from the last epoch
            - 'init' → NLL from the first epoch (index 0)
        """
        NODE_DIR = os.path.join(self.cfg.conf_dict["PATHS"]["EXPERIMENT_DIR"], node)
        train_path = os.path.join(NODE_DIR, "train_loss_hist.json")
        val_path = os.path.join(NODE_DIR, "val_loss_hist.json")

        if not os.path.exists(train_path) or not os.path.exists(val_path):
            if getattr(self, "debug", False):
                print(f"[DEBUG] Missing loss files for node '{node}'. Returning None.")
            return None, None

        try:
            with open(train_path, "r") as f:
                train_hist = json.load(f)
            with open(val_path, "r") as f:
                val_hist = json.load(f)

            train_nlls = np.array(train_hist)
            val_nlls = np.array(val_hist)

            if mode == "init":
                idx = 0
            elif mode == "last":
                idx = len(val_nlls) - 1
            elif mode == "best":
                idx = int(np.argmin(val_nlls))
            else:
                raise ValueError(f"Invalid mode '{mode}' — must be one of 'best', 'last', 'init'.")

            train_nll = float(train_nlls[idx])
            val_nll = float(val_nlls[idx])
            return train_nll, val_nll

        except Exception as e:
            print(f"[ERROR] Failed to load NLLs for node '{node}' ({mode}): {e}")
            return None, None

    def get_thetas(self, node: str, state: str = "best"):
        """
        Return the learned theta (intercept) parameters for a given node and model state.

        If the intercept dictionary for the given state does not exist,
        compute it using get_simple_intercepts_dict(). Otherwise, directly return
        the entry from the cached dictionary.

        Parameters
        ----------
        node : str
            Node name for which to retrieve theta (intercept) parameters.
        state : str, optional
            Model state ("best", "last", or "init"). Defaults to "best".

        Returns
        -------
        float, list, or None
            Theta (intercept) parameters for the specified node and state.
        """

        state = state.lower()
        if state not in ["best", "last", "init"]:
            raise ValueError(f"[ERROR] Invalid state '{state}'. Must be one of ['best', 'last', 'init'].")

        dict_attr = "intercept_dicts"

        # If no cached intercepts exist, compute them
        if not hasattr(self, dict_attr):
            if getattr(self, "debug", False):
                print(f"[DEBUG] '{dict_attr}' not found, computing via get_simple_intercepts_dict().")
            setattr(self, dict_attr, self.get_simple_intercepts_dict())

        all_dicts = getattr(self, dict_attr)

        # If the requested state isn’t cached, recompute
        if state not in all_dicts:
            if getattr(self, "debug", False):
                print(f"[DEBUG] State '{state}' not found in cached intercepts, recomputing full dict.")
            setattr(self, dict_attr, self.get_simple_intercepts_dict())
            all_dicts = getattr(self, dict_attr)

        state_dict = all_dicts.get(state, {})

        # Return cached node intercept if present
        if node in state_dict:
            return state_dict[node]

        # If not found, recompute full dict as fallback
        if getattr(self, "debug", False):
            print(f"[DEBUG] Node '{node}' not found in state '{state}', recomputing full dict.")
        setattr(self, dict_attr, self.get_simple_intercepts_dict())
        all_dicts = getattr(self, dict_attr)
        return all_dicts.get(state, {}).get(node, None)
        
    def get_linear_shifts(self, node: str, state: str = "best"):
        """
        Return the learned linear shift terms for a given node and model state.

        If the linear shift dictionary for the given state does not exist,
        compute it using get_linear_shifts_dict(). Otherwise, directly return
        the entry from the cached dictionary.

        Parameters
        ----------
        node : str
            Node name for which to retrieve linear shift terms.
        state : str, optional
            Model state ("best", "last", or "init"). Defaults to "best".

        Returns
        -------
        dict or None
            Linear shift terms for the specified node and state.
        """

        state = state.lower()
        if state not in ["best", "last", "init"]:
            raise ValueError(f"[ERROR] Invalid state '{state}'. Must be one of ['best', 'last', 'init'].")

        dict_attr = "linear_shift_dicts"

        # If no global dicts cached, compute once
        if not hasattr(self, dict_attr):
            if getattr(self, "debug", False):
                print(f"[DEBUG] '{dict_attr}' not found, computing via get_linear_shifts_dict().")
            setattr(self, dict_attr, self.get_linear_shifts_dict())

        all_dicts = getattr(self, dict_attr)

        # If the requested state isn't cached, compute all again (covers fresh runs)
        if state not in all_dicts:
            if getattr(self, "debug", False):
                print(f"[DEBUG] State '{state}' not found in cached linear shifts, recomputing full dict.")
            setattr(self, dict_attr, self.get_linear_shifts_dict())
            all_dicts = getattr(self, dict_attr)

        # Now fetch the dictionary for this state
        state_dict = all_dicts.get(state, {})

        # If the node is available, return its entry
        if node in state_dict:
            return state_dict[node]

        # If missing, try recomputing (fallback)
        if getattr(self, "debug", False):
            print(f"[DEBUG] Node '{node}' not found in state '{state}', recomputing full dict.")
        setattr(self, dict_attr, self.get_linear_shifts_dict())
        all_dicts = getattr(self, dict_attr)
        return all_dicts.get(state, {}).get(node, None)

    def get_linear_shifts_dict(self):
        """
        Return a nested dictionary of linear shift terms for all nodes.
        Output structure:
            {
                'best': {node: {...}},
                'last': {node: {...}},
                'init': {node: {...}}
            }
        If no best or last model exists, only 'init' is returned for that node.
        """

        EXPERIMENT_DIR = self.cfg.conf_dict["PATHS"]["EXPERIMENT_DIR"]
        nodes_list = list(self.models.keys())
        all_states = ["best", "last", "init"]
        all_linear_shift_dicts = {state: {} for state in all_states}

        for node in nodes_list:
            NODE_DIR = os.path.join(EXPERIMENT_DIR, node)
            BEST_MODEL_PATH, LAST_MODEL_PATH, _, _ = model_train_val_paths(NODE_DIR)
            INIT_MODEL_PATH = os.path.join(NODE_DIR, "initial_model.pt")

            state_paths = {
                "best": BEST_MODEL_PATH,
                "last": LAST_MODEL_PATH,
                "init": INIT_MODEL_PATH,
            }

            for state, LOAD_PATH in state_paths.items():
                if not os.path.exists(LOAD_PATH):
                    if state != "init":
                        # skip best/last if unavailable
                        continue
                    else:
                        print(f"[WARNING] No models found for node '{node}'. Only initial model will be used.")
                        if not os.path.exists(LOAD_PATH):
                            if getattr(self, "debug", False):
                                print(f"[DEBUG] Initial model also missing for node '{node}'. Skipping.")
                            continue

                # Load parents and model
                _, terms_dict, _ = ordered_parents(node, self.nodes_dict)
                state_dict = torch.load(LOAD_PATH, map_location=self.device)
                tram_model = self.models[node]
                tram_model.load_state_dict(state_dict)

                epoch_weights = {}
                if hasattr(tram_model, "nn_shift") and tram_model.nn_shift is not None:
                    for i, shift_layer in enumerate(tram_model.nn_shift):
                        module_name = shift_layer.__class__.__name__
                        if (
                            hasattr(shift_layer, "fc")
                            and hasattr(shift_layer.fc, "weight")
                            and module_name == "LinearShift"
                        ):
                            term_name = list(terms_dict.keys())[i]
                            epoch_weights[f"ls({term_name})"] = (
                                shift_layer.fc.weight.detach().cpu().squeeze().tolist()
                            )
                        elif getattr(self, "debug", False):
                            term_name = list(terms_dict.keys())[i]
                            print(f"[DEBUG] ls({term_name}): missing 'fc' or 'weight' in LinearShift.")
                else:
                    if getattr(self, "debug", False):
                        print(f"[DEBUG] Tram model for node '{node}' has no nn_shift or it is None.")

                all_linear_shift_dicts[state][node] = epoch_weights

        # Remove empty states (e.g., when best/last not found for all nodes)
        all_linear_shift_dicts = {k: v for k, v in all_linear_shift_dicts.items() if v}

        return all_linear_shift_dicts

    def get_simple_intercepts_dict(self):
        """
        Return a nested dictionary of transformed simple intercept (SI) weights 
        for all nodes and model states ('best', 'last', 'init').
        Output structure:
            {
                'best': {node: [...]},
                'last': {node: [...]},
                'init': {node: [...]}
            }
        If no best or last model exists, only 'init' is returned for that node.
        """

        EXPERIMENT_DIR = self.cfg.conf_dict["PATHS"]["EXPERIMENT_DIR"]
        nodes_list = list(self.models.keys())
        all_states = ["best", "last", "init"]
        all_si_intercept_dicts = {state: {} for state in all_states}

        debug = getattr(self, "debug", False)
        verbose = getattr(self, "verbose", False)
        is_ontram = getattr(self, "is_ontram", False)

        for node in nodes_list:
            NODE_DIR = os.path.join(EXPERIMENT_DIR, node)
            BEST_MODEL_PATH, LAST_MODEL_PATH, _, _ = model_train_val_paths(NODE_DIR)
            INIT_MODEL_PATH = os.path.join(NODE_DIR, "initial_model.pt")

            state_paths = {
                "best": BEST_MODEL_PATH,
                "last": LAST_MODEL_PATH,
                "init": INIT_MODEL_PATH,
            }

            for state, LOAD_PATH in state_paths.items():
                if not os.path.exists(LOAD_PATH):
                    if state != "init":
                        continue
                    else:
                        print(f"[WARNING] No models found for node '{node}'. Only initial model will be used.")
                        if not os.path.exists(LOAD_PATH):
                            if debug:
                                print(f"[DEBUG] Initial model also missing for node '{node}'. Skipping.")
                            continue

                # Load model state
                state_dict = torch.load(LOAD_PATH, map_location=self.device)
                tram_model = self.models[node]
                tram_model.load_state_dict(state_dict)

                # Extract and transform simple intercept weights
                si_weights = None
                if hasattr(tram_model, "nn_int") and tram_model.nn_int is not None and isinstance(tram_model.nn_int, SimpleIntercept):
                    if hasattr(tram_model.nn_int, "fc") and hasattr(tram_model.nn_int.fc, "weight"):
                        weights = tram_model.nn_int.fc.weight.detach().cpu().tolist()
                        weights_tensor = torch.Tensor(weights)

                        if debug:
                            print(f"[DEBUG] Node '{node}' ({state}) theta tilde shape: {weights_tensor.shape}")

                        if is_ontram:
                            si_weights = transform_intercepts_ordinal(weights_tensor.reshape(1, -1))[:, 1:-1].reshape(-1, 1)
                        else:
                            si_weights = transform_intercepts_continous(weights_tensor.reshape(1, -1)).reshape(-1, 1)

                        si_weights = si_weights.tolist()

                        if debug:
                            print(f"[DEBUG] Node '{node}' ({state}) theta transformed: {si_weights}")
                    else:
                        if debug:
                            print(f"[DEBUG] Node '{node}' ({state}): missing 'fc' or 'weight' in SimpleIntercept.")
                else:
                    if debug:
                        print(f"[DEBUG] Tram model for node '{node}' has no nn_int or it is None.")

                all_si_intercept_dicts[state][node] = si_weights

        # Clean up empty states
        all_si_intercept_dicts = {k: v for k, v in all_si_intercept_dicts.items() if v}
        return all_si_intercept_dicts
       
    def summary(self, verbose=False):
        """
        Summarize TramDagModel training status, parameters, and checkpoints.
        """

        # ---------- SETUP ----------
        try:
            EXPERIMENT_DIR = self.cfg.conf_dict["PATHS"]["EXPERIMENT_DIR"]
        except KeyError:
            EXPERIMENT_DIR = None
            print("[WARNING] Missing EXPERIMENT_DIR in cfg.conf_dict['PATHS'].")

        print("\n" + "=" * 120)
        print(f"{'TRAM DAG MODEL SUMMARY':^120}")
        print("=" * 120)

        # ---------- METRICS OVERVIEW ----------
        summary_data = []
        for node in self.models.keys():
            node_dir = os.path.join(self.cfg.conf_dict["PATHS"]["EXPERIMENT_DIR"], node)
            train_path = os.path.join(node_dir, "train_loss_hist.json")
            val_path = os.path.join(node_dir, "val_loss_hist.json")

            if os.path.exists(train_path) and os.path.exists(val_path):
                best_train_nll, best_val_nll = self.get_train_val_nll(node, "best")
                last_train_nll, last_val_nll = self.get_train_val_nll(node, "last")
                n_epochs_total = len(json.load(open(train_path)))
            else:
                best_train_nll = best_val_nll = last_train_nll = last_val_nll = None
                n_epochs_total = 0

            summary_data.append({
                "Node": node,
                "Best Train NLL": best_train_nll,
                "Best Val NLL": best_val_nll,
                "Last Train NLL": last_train_nll,
                "Last Val NLL": last_val_nll,
                "Epochs": n_epochs_total,
            })

        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.round(4)

        print("\n[1] TRAINING METRICS OVERVIEW")
        print("-" * 120)
        if not df_summary.empty:
            print(
                df_summary.to_string(
                    index=False,
                    justify="center",
                    col_space=14,
                    float_format=lambda x: f"{x:7.4f}",
                )
            )
        else:
            print("No training history found for any node.")
        print("-" * 120)

        # ---------- NODE DETAILS ----------
        print("\n[2] NODE-SPECIFIC DETAILS")
        print("-" * 120)
        for node in self.models.keys():
            print(f"\n{f'NODE: {node}':^120}")
            print("-" * 120)

            # THETAS & SHIFTS
            for state in ["init", "last", "best"]:
                print(f"\n  [{state.upper()} STATE]")

                # ---- Thetas ----
                try:
                    thetas = getattr(self, "get_thetas", lambda n, s=None: None)(node, state)
                    if thetas is not None:
                        if isinstance(thetas, (list, np.ndarray, pd.Series)):
                            thetas_flat = np.array(thetas).flatten()
                            compact = np.round(thetas_flat, 4)
                            arr_str = np.array2string(
                                compact,
                                max_line_width=110,
                                threshold=np.inf,
                                separator=", "
                            )
                            lines = arr_str.split("\n")
                            if len(lines) > 2:
                                arr_str = "\n".join(lines[:2]) + " ..."
                            print(f"    Θ ({len(thetas_flat)}): {arr_str}")
                        elif isinstance(thetas, dict):
                            for k, v in thetas.items():
                                print(f"     Θ[{k}]: {v}")
                        else:
                            print(f"    Θ: {thetas}")
                    else:
                        print("    Θ: not available")
                except Exception as e:
                    print(f"    [Error loading thetas] {e}")

                # ---- Linear Shifts ----
                try:
                    linear_shifts = getattr(self, "get_linear_shifts", lambda n, s=None: None)(node, state)
                    if linear_shifts is not None:
                        if isinstance(linear_shifts, dict):
                            for k, v in linear_shifts.items():
                                print(f"     {k}: {np.round(v, 4)}")
                        elif isinstance(linear_shifts, (list, np.ndarray, pd.Series)):
                            arr = np.round(linear_shifts, 4)
                            print(f"    Linear shifts ({len(arr)}): {arr}")
                        else:
                            print(f"    Linear shifts: {linear_shifts}")
                    else:
                        print("    Linear shifts: not available")
                except Exception as e:
                    print(f"    [Error loading linear shifts] {e}")

            # ---- Verbose info directly below node ----
            if verbose:
                print("\n  [DETAILS]")
                node_dir = os.path.join(EXPERIMENT_DIR, node) if EXPERIMENT_DIR else None
                model = self.models[node]

                print(f"    Model Architecture:")
                arch_str = str(model).split("\n")
                for line in arch_str:
                    print(f"      {line}")
                print(f"    Parameter count: {sum(p.numel() for p in model.parameters()):,}")

                if node_dir and os.path.exists(node_dir):
                    ckpt_exists = any(f.endswith(('.pt', '.pth')) for f in os.listdir(node_dir))
                    print(f"    Checkpoints found: {ckpt_exists}")

                    sampling_dir = os.path.join(node_dir, "sampling")
                    sampling_exists = os.path.isdir(sampling_dir) and len(os.listdir(sampling_dir)) > 0
                    print(f"    Sampling results found: {sampling_exists}")

                    for label, filename in [("Train", "train_loss_hist.json"), ("Validation", "val_loss_hist.json")]:
                        path = os.path.join(node_dir, filename)
                        if os.path.exists(path):
                            try:
                                with open(path, "r") as f:
                                    hist = json.load(f)
                                print(f"    {label} history: {len(hist)} epochs")
                            except Exception as e:
                                print(f"    {label} history: failed to load ({e})")
                        else:
                            print(f"    {label} history: not found")
                else:
                    print("    [INFO] No experiment directory defined or missing for this node.")
            print("-" * 120)

        # ---------- TRAINING DATAFRAME ----------
        print("\n[3] TRAINING DATAFRAME")
        print("-" * 120)
        try:
            self.train_df.info()
        except AttributeError:
            print("No training DataFrame attached to this TramDagModel.")
        print("=" * 120 + "\n")



