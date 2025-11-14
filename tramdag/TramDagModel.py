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

from .utils.model_helpers import train_val_loop,evaluate_tramdag_model, get_fully_specified_tram_model , model_train_val_paths ,ordered_parents
from .utils.sampling import create_latent_df_for_full_dag, sample_full_dag, is_outcome_modelled_ordinal,is_outcome_modelled_continous, is_outcome_modelled_ordinal, show_hdag_continous,show_hdag_ordinal
from .utils.continous import transform_intercepts_continous
from .utils.ordinal import transform_intercepts_ordinal

from .models.tram_models import SimpleIntercept

from .TramDagConfig import TramDagConfig
from .TramDagDataset import TramDagDataset


#%pip install -i https://test.pypi.org/simple --extra-index-url https://pypi.org/simple tramdag
# # Remove previous builds
# rm -rf build dist *.egg-info

# # Build new package
# python -m build

# # Upload to TestPyPI
# python -m twine upload --repository testpypi dist/*

# documentaiton
# pdoc tramdag -o docs

## TODO ordinal cutpoints trafo plot
## TODO documentation with docusaurus
## TODO psuh latest version to pypi

## TODO check the cutpoints >= <= for correct cutoffs

## TODO what happens if parents are proba but node is continous? # has also jsut k observations like in ordinal case

## TODO solve visualizatin for probabalistic samples

class TramDagModel:
    """
    Probabilistic DAG model built from node-wise TRAMs (transformation models).

    This class manages:
    - Configuration and per-node model construction.
    - Data scaling (min–max).
    - Training (sequential or per-node parallel on CPU).
    - Diagnostics (loss history, intercepts, linear shifts, latents).
    - Sampling from the joint DAG and loading stored samples.
    - High-level summaries and plotting utilities.
    """
    
    # ---- defaults used at construction time ----
    DEFAULTS_CONFIG = {
        "set_initial_weights": False,
        "debug":False,
        "verbose": False,
        "device":'auto',
        "initial_data":None,
        "overwrite_initial_weights": True,
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
        "num_workers" : 4,
        "persistent_workers" : True,
        "prefetch_factor" : 4,
        "batch_size":1000,
        
    }

    def __init__(self):
        """
        Initialize an empty TramDagModel shell.

        Notes
        -----
        This constructor does not build any node models and does not attach a
        configuration. Use `TramDagModel.from_config` or `TramDagModel.from_directory`
        to obtain a fully configured and ready-to-use instance.
        """
        
        self.debug = False
        self.verbose = False
        self.device = 'auto'
        pass

    @staticmethod
    def get_device(settings):
        """
        Resolve the target device string from a settings dictionary.

        Parameters
        ----------
        settings : dict
            Dictionary containing at least a key ``"device"`` with one of
            {"auto", "cpu", "cuda"}. If missing, "auto" is assumed.

        Returns
        -------
        str
            Device string, either "cpu" or "cuda".

        Notes
        -----
        If ``device == "auto"``, CUDA is selected if available, otherwise CPU.
        """
        device_arg = settings.get("device", "auto")
        if device_arg == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_str = device_arg
        return device_str

    def _validate_kwargs(self, kwargs: dict, defaults_attr: str = "DEFAULTS_FIT", context: str = None):
        """
        Validate a kwargs dictionary against a class-level defaults dictionary.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments to validate.
        defaults_attr : str, optional
            Name of the attribute on this class that contains the allowed keys,
            e.g. ``"DEFAULTS_CONFIG"`` or ``"DEFAULTS_FIT"``. Default is "DEFAULTS_FIT".
        context : str or None, optional
            Optional label (e.g. caller name) to prepend in error messages.

        Raises
        ------
        AttributeError
            If the attribute named by ``defaults_attr`` does not exist.
        ValueError
            If any key in ``kwargs`` is not present in the corresponding defaults dict.
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
        Construct a TramDagModel from a TramDagConfig object.

        This builds one TRAM model per node in the DAG and optionally writes
        the initial model parameters to disk.

        Parameters
        ----------
        cfg : TramDagConfig
            Configuration wrapper holding the underlying configuration dictionary,
            including at least:
            - ``conf_dict["nodes"]``: mapping of node names to node configs.
            - ``conf_dict["PATHS"]["EXPERIMENT_DIR"]``: experiment directory.
        **kwargs
            Node-level construction options. Each key must be present in
            ``DEFAULTS_CONFIG``. Values can be:
            - scalar: applied to all nodes.
            - dict: mapping ``{node_name: value}`` for per-node overrides.

            Common keys include:
            device : {"auto", "cpu", "cuda"}, default "auto"
                Device selection (CUDA if available when "auto").
            debug : bool, default False
                If True, print debug messages.
            verbose : bool, default False
                If True, print informational messages.
            set_initial_weights : bool
                Passed to underlying TRAM model constructors.
            overwrite_initial_weights : bool, default True
                If True, overwrite any existing ``initial_model.pt`` files per node.
            initial_data : Any
                Optional object passed down to node constructors.

        Returns
        -------
        TramDagModel
            Fully initialized instance with:
            - ``cfg``
            - ``nodes_dict``
            - ``models`` (per-node TRAMs)
            - ``settings`` (resolved per-node config)

        Raises
        ------
        ValueError
            If any dict-valued kwarg does not provide values for exactly the set
            of nodes in ``cfg.conf_dict["nodes"]``.
        """
        
        self = cls()
        self.cfg = cfg
        self.cfg.update()  # ensure latest version from disk
        self.cfg._verify_completeness()
        
        
        try:
            self.cfg.save()  # persist back to disk
            if getattr(self, "debug", False):
                print("[DEBUG] Configuration updated and saved.")
        except Exception as e:
            print(f"[WARNING] Could not save configuration after update: {e}")        
            
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
            
            TEMP_DIR = "temp"
            if os.path.isdir(TEMP_DIR) and not os.listdir(TEMP_DIR):
                os.rmdir(TEMP_DIR)
                            
        return self

    @classmethod
    def from_directory(cls, EXPERIMENT_DIR: str, device: str = "auto", debug: bool = False, verbose: bool = False):
        """
        Reconstruct a TramDagModel from an experiment directory on disk.

        This method:
        1. Loads the configuration JSON.
        2. Wraps it in a TramDagConfig.
        3. Builds all node models via `from_config`.
        4. Loads the min–max scaling dictionary.

        Parameters
        ----------
        EXPERIMENT_DIR : str
            Path to an experiment directory containing:
            - ``configuration.json``
            - ``min_max_scaling.json``.
        device : {"auto", "cpu", "cuda"}, optional
            Device selection. Default is "auto".
        debug : bool, optional
            If True, enable debug messages. Default is False.
        verbose : bool, optional
            If True, enable informational messages. Default is False.

        Returns
        -------
        TramDagModel
            A TramDagModel instance with models, config, and scaling loaded.

        Raises
        ------
        FileNotFoundError
            If configuration or min–max files cannot be found.
        RuntimeError
            If the min–max file cannot be read or parsed.
        """

        # --- load config file ---
        config_path = os.path.join(EXPERIMENT_DIR, "configuration.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"[ERROR] Config file not found at {config_path}")

        with open(config_path, "r") as f:
            cfg_dict = json.load(f)

        # Create TramConfig wrapper 
        cfg = TramDagConfig(cfg_dict, CONF_DICT_PATH=config_path)

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
        Ensure that the input data is represented as a TramDagDataset.

        Parameters
        ----------
        data : pandas.DataFrame, TramDagDataset, or None
            Input data to be converted or passed through.
        is_val : bool, optional
            If True, the resulting dataset is treated as validation data
            (e.g. no shuffling). Default is False.
        **kwargs
            Additional keyword arguments passed through to
            ``TramDagDataset.from_dataframe``.

        Returns
        -------
        TramDagDataset or None
            A TramDagDataset if ``data`` is a DataFrame or TramDagDataset,
            otherwise None if ``data`` is None.

        Raises
        ------
        TypeError
            If ``data`` is not a DataFrame, TramDagDataset, or None.
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
                "node": {"min": float, "max": float},
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
        Train a single node model (helper for per-node training).

        This method is designed to be called either from the main process
        (sequential training) or from a joblib worker (parallel CPU training).

        Parameters
        ----------
        node : str
            Name of the target node to train.
        self_ref : TramDagModel
            Reference to the TramDagModel instance containing models and config.
        settings : dict
            Training settings dictionary, typically derived from ``DEFAULTS_FIT``
            plus any user overrides.
        td_train_data : TramDagDataset
            Training dataset with node-specific DataLoaders in ``.loaders``.
        td_val_data : TramDagDataset or None
            Validation dataset or None.
        device_str : str
            Device string, e.g. "cpu" or "cuda".

        Returns
        -------
        tuple
            A tuple ``(node, history)`` where:
            node : str
                Node name.
            history : dict or Any
                Training history as returned by ``train_val_loop``.
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
        Train TRAM models for all nodes in the DAG.

        Coordinates dataset preparation, min–max scaling, and per-node training,
        optionally in parallel on CPU.

        Parameters
        ----------
        train_data : pandas.DataFrame or TramDagDataset
            Training data. If a DataFrame is given, it is converted into a
            TramDagDataset using `_ensure_dataset`.
        val_data : pandas.DataFrame or TramDagDataset or None, optional
            Validation data. If a DataFrame is given, it is converted into a
            TramDagDataset. If None, no validation loss is computed.
        **kwargs
            Overrides for ``DEFAULTS_FIT``. All keys must exist in
            ``DEFAULTS_FIT``. Common options:

            epochs : int, default 100
                Number of training epochs per node.
            learning_rate : float, default 0.01
                Learning rate for the default Adam optimizer.
            train_list : list of str or None, optional
                List of node names to train. If None, all nodes are trained.
            train_mode : {"sequential", "parallel"}, default "sequential"
                Training mode. "parallel" uses joblib-based CPU multiprocessing.
                GPU forces sequential mode.
            device : {"auto", "cpu", "cuda"}, default "auto"
                Device selection.
            optimizers : dict or None
                Optional mapping ``{node_name: optimizer}``. If provided for a
                node, that optimizer is used instead of creating a new Adam.
            schedulers : dict or None
                Optional mapping ``{node_name: scheduler}``.
            use_scheduler : bool
                If True, enable scheduler usage in the training loop.
            num_workers : int
                DataLoader workers in sequential mode (ignored in parallel).
            persistent_workers : bool
                DataLoader persistence in sequential mode (ignored in parallel).
            prefetch_factor : int
                DataLoader prefetch factor (ignored in parallel).
            batch_size : int
                Batch size for all node DataLoaders.
            debug : bool
                Enable debug output.
            verbose : bool
                Enable informational logging.
            return_history : bool
                If True, return a history dict.

        Returns
        -------
        dict or None
            If ``return_history=True``, a dictionary mapping each node name
            to its training history. Otherwise, returns None.

        Raises
        ------
        ValueError
            If ``train_mode`` is not "sequential" or "parallel".
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
        Load training and validation loss history for all nodes.

        Looks for per-node JSON files:

        - ``EXPERIMENT_DIR/{node}/train_loss_hist.json``
        - ``EXPERIMENT_DIR/{node}/val_loss_hist.json``

        Returns
        -------
        dict
            A dictionary mapping node names to:

            .. code-block:: python

                {
                    "train": list or None,
                    "validation": list or None
                }

            where each list contains NLL values per epoch, or None if not found.

        Raises
        ------
        ValueError
            If the experiment directory cannot be resolved from the configuration.
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
        Load linear shift term histories for all nodes.

        Each node history is expected in a JSON file named
        ``linear_shifts_all_epochs.json`` under the node directory.

        Returns
        -------
        dict
            A mapping ``{node_name: pandas.DataFrame}``, where each DataFrame
            contains linear shift weights across epochs.

        Raises
        ------
        ValueError
            If the experiment directory cannot be resolved from the configuration.

        Notes
        -----
        If a history file is missing for a node, a warning is printed and the
        node is omitted from the returned dictionary.
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
        Load simple intercept histories for all nodes.

        Each node history is expected in a JSON file named
        ``simple_intercepts_all_epochs.json`` under the node directory.

        Returns
        -------
        dict
            A mapping ``{node_name: pandas.DataFrame}``, where each DataFrame
            contains intercept weights across epochs.

        Raises
        ------
        ValueError
            If the experiment directory cannot be resolved from the configuration.

        Notes
        -----
        If a history file is missing for a node, a warning is printed and the
        node is omitted from the returned dictionary.
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
        Compute latent representations for all nodes in the DAG.

        Parameters
        ----------
        df : pandas.DataFrame
            Input data frame with columns corresponding to nodes in the DAG.
        verbose : bool, optional
            If True, print informational messages during latent computation.
            Default is False.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the original columns plus latent variables
            for each node (e.g. columns named ``f"{node}_U"``).

        Raises
        ------
        ValueError
            If the experiment directory is missing from the configuration or
            if ``self.minmax_dict`` has not been set.
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
        Plot training and validation loss evolution per node.

        Parameters
        ----------
        variable : str or None, optional
            If provided, plot loss history for this node only. If None, plot
            histories for all nodes that have both train and validation logs.

        Returns
        -------
        None

        Notes
        -----
        Two subplots are produced:
        - Full epoch history.
        - Last 10% of epochs (or only the last epoch if fewer than 5 epochs).
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
        Plot the evolution of linear shift terms over epochs.

        Parameters
        ----------
        data_dict : dict or None, optional
            Pre-loaded mapping ``{node_name: pandas.DataFrame}`` containing shift
            weights across epochs. If None, `linear_shift_history()` is called.
        node : str or None, optional
            If provided, plot only this node. Otherwise, plot all nodes
            present in ``data_dict``.
        ref_lines : dict or None, optional
            Optional mapping ``{node_name: list of float}``. For each specified
            node, horizontal reference lines are drawn at the given values.

        Returns
        -------
        None

        Notes
        -----
        The function flattens nested list-like entries in the DataFrames to scalars,
        converts epoch labels to numeric, and then draws one line per shift term.
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
        Plot the evolution of simple intercept weights over epochs.

        Parameters
        ----------
        data_dict : dict or None, optional
            Pre-loaded mapping ``{node_name: pandas.DataFrame}`` containing intercept
            weights across epochs. If None, `simple_intercept_history()` is called.
        node : str or None, optional
            If provided, plot only this node. Otherwise, plot all nodes present
            in ``data_dict``.
        ref_lines : dict or None, optional
            Optional mapping ``{node_name: list of float}``. For each specified
            node, horizontal reference lines are drawn at the given values.

        Returns
        -------
        None

        Notes
        -----
        Nested list-like entries in the DataFrames are reduced to scalars before
        plotting. One line is drawn per intercept parameter.
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
        Visualize latent U distributions for one or all nodes.

        Parameters
        ----------
        df : pandas.DataFrame
            Input data frame with raw node values.
        variable : str or None, optional
            If provided, only this node's latents are plotted. If None, all
            nodes with latent columns are processed.
        confidence : float, optional
            Confidence level for QQ-plot bands (0 < confidence < 1).
            Default is 0.95.
        simulations : int, optional
            Number of Monte Carlo simulations for QQ-plot bands. Default is 1000.

        Returns
        -------
        None

        Notes
        -----
        For each node, two plots are produced:
        - Histogram of the latent U values.
        - QQ-plot with simulation-based confidence bands under a logistic reference.
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
        Visualize the transformation function h() for selected DAG nodes.

        Parameters
        ----------
        df : pandas.DataFrame
            Input data containing node values or model predictions.
        variables : list of str or None, optional
            Names of nodes to visualize. If None, all nodes in ``self.models``
            are considered.
        plot_n_rows : int, optional
            Maximum number of rows from ``df`` to visualize. Default is 1.
        **kwargs
            Additional keyword arguments forwarded to the underlying plotting
            helpers (`show_hdag_continous` / `show_hdag_ordinal`).

        Returns
        -------
        None

        Notes
        -----
        - For continuous outcomes, `show_hdag_continous` is called.
        - For ordinal outcomes, `show_hdag_ordinal` is called.
        - Nodes that are neither continuous nor ordinal are skipped with a warning.
        """
                

        if len(df)> 1:
            print("[WARNING] len(df)>1, set: plot_n_rows accordingly")
        
        variables_list=variables if variables is not None else list(self.models.keys())
        for node in variables_list:
            if is_outcome_modelled_continous(node, self.nodes_dict):
                show_hdag_continous(df,node=node,configuration_dict=self.cfg.conf_dict,minmax_dict=self.minmax_dict,device=self.device,plot_n_rows=plot_n_rows,**kwargs)
            
            elif is_outcome_modelled_ordinal(node, self.nodes_dict):
                show_hdag_ordinal(df,node=node,configuration_dict=self.cfg.conf_dict,device=self.device,plot_n_rows=plot_n_rows,**kwargs)
                # plot_cutpoints_with_logistic(df,node=node,configuration_dict=self.cfg.conf_dict,device=self.device,plot_n_rows=plot_n_rows,**kwargs)
                # save_cutpoints_with_logistic(df,node=node,configuration_dict=self.cfg.conf_dict,device=self.device,**kwargs)
            else:
                print(f"[WARNING] Node {node} is wheter ordinal nor continous, not implemented yet")
    
    @staticmethod
    def _add_r_style_confidence_bands(ax, sample, dist, confidence=0.95, simulations=1000):
        """
        Add simulation-based confidence bands to a QQ-plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object on which to draw the QQ-plot and bands.
        sample : array-like
            Empirical sample used in the QQ-plot.
        dist : scipy.stats distribution
            Distribution object providing ``ppf`` and ``rvs`` methods (e.g. logistic).
        confidence : float, optional
            Confidence level (0 < confidence < 1) for the bands. Default is 0.95.
        simulations : int, optional
            Number of Monte Carlo simulations used to estimate the bands. Default is 1000.

        Returns
        -------
        None

        Notes
        -----
        The axes are cleared, and a new QQ-plot is drawn with:
        - Empirical vs. theoretical quantiles.
        - 45-degree reference line.
        - Shaded confidence band region.
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
        Sample from the joint DAG using the trained TRAM models.

        Allows for:
        
        Oberservational sampling
        Interventional sampling via ``do()`` operations
        Counterfactial sampling using predefined latent draws and do()
        
        Parameters
        ----------
        do_interventions : dict or None, optional
            Mapping of node names to intervened (fixed) values. For example:
            ``{"x1": 1.0}`` represents ``do(x1 = 1.0)``. Default is None.
        predefined_latent_samples_df : pandas.DataFrame or None, optional
            DataFrame containing columns ``"{node}_U"`` with predefined latent
            draws to be used instead of sampling from the prior. Default is None.
        **kwargs
            Sampling options overriding internal defaults:

            number_of_samples : int, default 10000
                Total number of samples to draw.
            batch_size : int, default 32
                Batch size for internal sampling loops.
            delete_all_previously_sampled : bool, default True
                If True, delete old sampling files in node-specific sampling
                directories before writing new ones.
            verbose : bool
                If True, print informational messages.
            debug : bool
                If True, print debug output.
            device : {"auto", "cpu", "cuda"}
                Device selection for sampling.
            use_initial_weights_for_sampling : bool, default False
                If True, sample from initial (untrained) model parameters.

        Returns
        -------
        tuple
            A tuple ``(sampled_by_node, latents_by_node)``:

            sampled_by_node : dict
                Mapping ``{node_name: torch.Tensor}`` of sampled node values.
            latents_by_node : dict
                Mapping ``{node_name: torch.Tensor}`` of latent U values used.

        Raises
        ------
        ValueError
            If the experiment directory cannot be resolved or if scaling
            information (``self.minmax_dict``) is missing.
        RuntimeError
            If min–max scaling has not been computed before calling `sample`.
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
        Load previously stored sampled values and latents for each node.

        Parameters
        ----------
        EXPERIMENT_DIR : str or None, optional
            Experiment directory path. If None, it is taken from
            ``self.cfg.conf_dict["PATHS"]["EXPERIMENT_DIR"]``.
        nodes : list of str or None, optional
            Nodes for which to load samples. If None, use all nodes from
            ``self.nodes_dict``.

        Returns
        -------
        tuple
            A tuple ``(sampled_by_node, latents_by_node)``:

            sampled_by_node : dict
                Mapping ``{node_name: torch.Tensor}`` of sampled values (on CPU).
            latents_by_node : dict
                Mapping ``{node_name: torch.Tensor}`` of latent values (on CPU).

        Raises
        ------
        ValueError
            If the experiment directory cannot be resolved or if no node list
            is available and ``nodes`` is None.

        Notes
        -----
        Nodes without both ``sampled.pt`` and ``latents.pt`` files are skipped
        with a warning.
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
        Compare sampled vs. observed distributions for selected nodes.

        Parameters
        ----------
        df : pandas.DataFrame
            Data frame containing the observed node values.
        sampled : dict or None, optional
            Optional mapping ``{node_name: array-like or torch.Tensor}`` of sampled
            values. If None or if a node is missing, samples are loaded from
            ``EXPERIMENT_DIR/{node}/sampling/sampled.pt``.
        variable : list of str or None, optional
            Subset of nodes to plot. If None, all nodes in the configuration
            are considered.
        bins : int, optional
            Number of histogram bins for continuous variables. Default is 100.
        hist_true_color : str, optional
            Color name for the histogram of true values. Default is "blue".
        hist_est_color : str, optional
            Color name for the histogram of sampled values. Default is "orange".
        figsize : tuple, optional
            Figure size for the matplotlib plots. Default is (14, 5).

        Returns
        -------
        None

        Notes
        -----
        - Continuous outcomes: histogram overlay + QQ-plot.
        - Ordinal outcomes: side-by-side bar plot of relative frequencies.
        - Other categorical outcomes: side-by-side bar plot with category labels.
        - If samples are probabilistic (2D tensor), the argmax across classes is used.
        """
        
        target_nodes = self.cfg.conf_dict["nodes"]
        experiment_dir = self.cfg.conf_dict["PATHS"]["EXPERIMENT_DIR"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        plot_list = variable if variable is not None else target_nodes

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

            # If logits/probabilities per sample, take argmax
            if sampled_vals.ndim == 2:
                    print(f"[INFO] CAUTION! {node}: samples are probabilistic — each sample follows a probability "
                    f"distribution based on the valid latent range. "
                    f"Note that this frequency plot reflects only the distribution of the most probable "
                    f"class per sample.")
                    sampled_vals = np.argmax(sampled_vals, axis=1)

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
                axs[0].hist(true_vals, bins=bins, density=True, alpha=0.6,
                            color=hist_true_color, label=f"True {node}")
                axs[0].hist(sampled_vals, bins=bins, density=True, alpha=0.6,
                            color=hist_est_color, label="Sampled")
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
        Retrieve training and validation NLL for a node and a given model state.

        Parameters
        ----------
        node : str
            Node name.
        mode : {"best", "last", "init"}
            State of interest:
            - "best": epoch with lowest validation NLL.
            - "last": final epoch.
            - "init": first epoch (index 0).

        Returns
        -------
        tuple of (float or None, float or None)
            A tuple ``(train_nll, val_nll)`` for the requested mode.
            Returns ``(None, None)`` if loss files are missing or cannot be read.

        Notes
        -----
        This method expects per-node JSON files:

        - ``train_loss_hist.json``
        - ``val_loss_hist.json``

        in the node directory.
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
        Return transformed intercept (theta) parameters for a node and state.

        Parameters
        ----------
        node : str
            Node name.
        state : {"best", "last", "init"}, optional
            Model state for which to return parameters. Default is "best".

        Returns
        -------
        Any or None
            Transformed theta parameters for the requested node and state.
            The exact structure (scalar, list, or other) depends on the model.

        Raises
        ------
        ValueError
            If an invalid state is given (not in {"best", "last", "init"}).

        Notes
        -----
        Intercept dictionaries are cached on the instance under the attribute
        ``intercept_dicts``. If missing or incomplete, they are recomputed using
        `get_simple_intercepts_dict`.
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
        Return learned linear shift terms for a node and a given state.

        Parameters
        ----------
        node : str
            Node name.
        state : {"best", "last", "init"}, optional
            Model state for which to return linear shift terms. Default is "best".

        Returns
        -------
        dict or Any or None
            Linear shift terms for the given node and state. Usually a dict
            mapping term names to weights.

        Raises
        ------
        ValueError
            If an invalid state is given (not in {"best", "last", "init"}).

        Notes
        -----
        Linear shift dictionaries are cached on the instance under the attribute
        ``linear_shift_dicts``. If missing or incomplete, they are recomputed using
        `get_linear_shifts_dict`.
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
        Compute linear shift term dictionaries for all nodes and states.

        For each node and each available state ("best", "last", "init"), this
        method loads the corresponding model checkpoint, extracts linear shift
        weights from the TRAM model, and stores them in a nested dictionary.

        Returns
        -------
        dict
            Nested dictionary of the form:

            .. code-block:: python

                {
                    "best": {node: {...}},
                    "last": {node: {...}},
                    "init": {node: {...}},
                }

            where the innermost dict maps term labels (e.g. ``"ls(parent_name)"``)
            to their weights.

        Notes
        -----
        - If "best" or "last" checkpoints are unavailable for a node, only
        the "init" entry is populated.
        - Empty outer states (without any nodes) are removed from the result.
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
        Compute transformed simple intercept dictionaries for all nodes and states.

        For each node and each available state ("best", "last", "init"), this
        method loads the corresponding model checkpoint, extracts simple intercept
        weights, transforms them into interpretable theta parameters, and stores
        them in a nested dictionary.

        Returns
        -------
        dict
            Nested dictionary of the form:

            .. code-block:: python

                {
                    "best": {node: [[theta_1], [theta_2], ...]},
                    "last": {node: [[theta_1], [theta_2], ...]},
                    "init": {node: [[theta_1], [theta_2], ...]},
                }

        Notes
        -----
        - For ordinal models (``self.is_ontram == True``), `transform_intercepts_ordinal`
        is used.
        - For continuous models, `transform_intercepts_continous` is used.
        - Empty outer states (without any nodes) are removed from the result.
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
        Print a multi-part textual summary of the TramDagModel.

        The summary includes:
        1. Training metrics overview per node (best/last NLL, epochs).
        2. Node-specific details (thetas, linear shifts, optional architecture).
        3. Basic information about the attached training DataFrame, if present.

        Parameters
        ----------
        verbose : bool, optional
            If True, include extended per-node details such as the model
            architecture, parameter count, and availability of checkpoints
            and sampling results. Default is False.

        Returns
        -------
        None

        Notes
        -----
        This method prints to stdout and does not return structured data.
        It is intended for quick, human-readable inspection of the current
        training and model state.
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



