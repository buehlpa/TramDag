from utils.configuration import *

class TramDagConfig:
    def __init__(self, conf_dict: dict = None, CONF_DICT_PATH: str = None):
        """
        Initialize TramDagConfig.

        Args:
            conf_dict: optional dict with configuration. If None, starts empty.
            CONF_DICT_PATH: optional path to config file.
        """
        
        #TODO add verbose and debug , vebose print only infos, debug prints info + debug statements, warnings, errors are always printed
        #TODO add veryfier such that nothing is missing for later training such as experiment name 
        
        self.conf_dict = conf_dict or {}
        self.CONF_DICT_PATH = CONF_DICT_PATH


    @classmethod
    def load(cls, CONF_DICT_PATH: str):
        """
        Alternative constructor: load config directly from a file.
        """
        conf = load_configuration_dict(CONF_DICT_PATH)
        return cls(conf, CONF_DICT_PATH=CONF_DICT_PATH)

    def save(self, CONF_DICT_PATH: str = None):
        """
        Save config to file. If path is not provided, fall back to stored path.
        """
        path = CONF_DICT_PATH or self.CONF_DICT_PATH
        if path is None:
            raise ValueError("No CONF_DICT_PATH provided to save config.")
        write_configuration_dict(self.conf_dict, path)

    def compute_scaling(self, df: pd.DataFrame, write: bool = True):
        """
        Derive scaling information (min, max, levels) from data USE training data.
        """
        print("[INFO] Make sure to provide only training data to compute_scaling!")
        # calculate 5% and 95% quantiles for min and max values
        quantiles = df.quantile([0.05, 0.95])
        min_vals = quantiles.loc[0.05]
        max_vals = quantiles.loc[0.95]

        # calculate levels for categorical variables
        levels_dict = create_levels_dict(df, self.conf_dict['data_type'])

        # TODO remove outer dependency of these functions (re-loading conf dict)
        adj_matrix = read_adj_matrix_from_configuration(self.CONF_DICT_PATH)
        nn_names_matrix = read_nn_names_matrix_from_configuration(self.CONF_DICT_PATH)

        node_dict = create_node_dict(
            adj_matrix,
            nn_names_matrix,
            self.conf_dict['data_type'],
            min_vals=min_vals,
            max_vals=max_vals,
            levels_dict=levels_dict
        )
        conf_dict = load_configuration_dict(self.CONF_DICT_PATH)
        conf_dict['nodes'] = node_dict
        self.conf_dict = conf_dict  # keep it in memory too

        if write and self.CONF_DICT_PATH is not None:
            try:
                write_configuration_dict(conf_dict, self.CONF_DICT_PATH)
                print(f'[INFO] Configuration with updated scaling saved to {self.CONF_DICT_PATH}')
            except Exception as e:
                print(f'[ERROR] Failed to save configuration: {e}')
                
                
import inspect
from archive.utils.tram_data import GenericDataset
from torch.utils.data import Dataset, DataLoader

class TramDagDataset(Dataset):
    
    #TODO add docstring
    #TODO add verbose and debug , vebose print only infos, debug prints info + debug statements, warnings, errors are always printed
    #TODO add veryfier such that nothing is missing for later training such as experiment name 
    
    DEFAULTS = {
        "batch_size": 32,
        "shuffle": True,
        "num_workers": 4,
        "pin_memory": False,
        "return_intercept_shift": True,
        "debug": False,
        "transform": None,
    }

    def __init__(self):
        """Empty init. Use classmethods like .from_dataframe()."""
        pass

    @classmethod
    def from_dataframe(cls, df, cfg, **kwargs):
        self = cls()
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"[ERROR] df must be a pandas DataFrame, but got {type(df)}")

        # merge defaults with overrides
        settings = dict(cls.DEFAULTS)
        settings.update(kwargs)

        # infer variable name automatically
        callers_locals = inspect.currentframe().f_back.f_locals
        inferred = None
        for var_name, var_val in callers_locals.items():
            if var_val is df:
                inferred = var_name
                break
        df_name = inferred or "dataframe"

        if settings["shuffle"]:
            if any(x in df_name.lower() for x in ["val", "validation", "test"]):
                print(f"[WARNING] DataFrame '{df_name}' looks like a validation/test set → shuffle=True. Are you sure?")

        self.cfg = cfg
        self.df = df.copy()
        self._apply_settings(settings)
        self._build_dataloaders()
        return self

    def _apply_settings(self, settings: dict):
        """Apply settings from defaults + overrides."""
        self.batch_size = settings["batch_size"]
        self.shuffle = settings["shuffle"]
        self.num_workers = settings["num_workers"]
        self.pin_memory = settings["pin_memory"]
        self.return_intercept_shift = settings["return_intercept_shift"]
        self.debug = settings["debug"]
        self.transform = settings["transform"]

        # nodes dict
        self.nodes_dict = self.cfg.conf_dict["nodes"]

        # validate dict attributes for all configurable params
        for name, val in {
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "return_intercept_shift": self.return_intercept_shift,
            "debug": self.debug,
            "transform": self.transform,
        }.items():
            self._check_keys(name, val)

    def _build_dataloaders(self):
        """Build node-specific dataloaders from df + settings."""
        self.loaders = {}
        for node in self.nodes_dict:
            ds = GenericDataset(
                self.df,
                target_col=node,
                target_nodes=self.nodes_dict,
                transform=self.transform if not isinstance(self.transform, dict) else self.transform[node],
                return_intercept_shift=self.return_intercept_shift if not isinstance(self.return_intercept_shift, dict) else self.return_intercept_shift[node],
                debug=self.debug if not isinstance(self.debug, dict) else self.debug[node],
            )

            batch_size = self.batch_size[node] if isinstance(self.batch_size, dict) else self.batch_size
            shuffle_flag = self.shuffle[node] if isinstance(self.shuffle, dict) else bool(self.shuffle)
            num_workers = self.num_workers[node] if isinstance(self.num_workers, dict) else self.num_workers
            pin_memory = self.pin_memory[node] if isinstance(self.pin_memory, dict) else self.pin_memory

            self.loaders[node] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

    def _check_keys(self, attr_name, attr_value):
        """Check if dict keys match cfg.conf_dict['nodes'].keys()."""
        if isinstance(attr_value, dict):
            expected_keys = set(self.nodes_dict.keys())
            given_keys = set(attr_value.keys())
            if expected_keys != given_keys:
                raise ValueError(
                    f"[ERROR] the provided attribute '{attr_name}' keys are not same as in cfg.conf_dict['nodes'].keys().\n"
                    f"Expected: {expected_keys}, but got: {given_keys}\n"
                    f"Please provide values for all variables."
                )

    def summary(self):
        print("\n[TramDagDataset Summary]")
        print("=" * 60)

        # ---- DataFrame section ----
        print("\n[DataFrame]")
        print("Shape:", self.df.shape)
        print("\nHead:")
        print(self.df.head())

        print("\nDtypes:")
        print(self.df.dtypes)

        print("\nDescribe:")
        print(self.df.describe(include="all"))

        # ---- Settings per node ----
        print("\n[Node Settings]")
        for node in self.nodes_dict.keys():
            batch_size = self.batch_size[node] if isinstance(self.batch_size, dict) else self.batch_size
            shuffle_flag = self.shuffle[node] if isinstance(self.shuffle, dict) else bool(self.shuffle)
            num_workers = self.num_workers[node] if isinstance(self.num_workers, dict) else self.num_workers
            pin_memory = self.pin_memory[node] if isinstance(self.pin_memory, dict) else self.pin_memory
            rshift = self.return_intercept_shift[node] if isinstance(self.return_intercept_shift, dict) else self.return_intercept_shift
            debug_flag = self.debug[node] if isinstance(self.debug, dict) else self.debug
            transform = self.transform[node] if isinstance(self.transform, dict) else self.transform

            print(
                f" Node '{node}': "
                f"batch_size={batch_size}, "
                f"shuffle={shuffle_flag}, "
                f"num_workers={num_workers}, "
                f"pin_memory={pin_memory}, "
                f"return_intercept_shift={rshift}, "
                f"debug={debug_flag}, "
                f"transform={transform}"
            )
        print("=" * 60 + "\n")

    def __getitem__(self, idx):
        return self.df.iloc[idx].to_dict()

    def __len__(self):
        return len(self.df)
          
                

from utils.tram_model_helpers import train_val_loop, get_fully_specified_tram_model 
from archive.utils.tram_data_helpers import create_latent_df_for_full_dag, sample_full_dag
from torch.optim import Adam
import torch
import os


class TramDagModel:
    
    #TODO add docstring
    #TODO add verbose and debug , vebose print only infos, debug prints info + debug statements, warnings, errors are always printed
    # ---- defaults used at construction time ----
    DEFAULTS_CONFIG = {
        "set_initial_weights": True,
        "debug":False,
        
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
        "debug":False,
        "verbose": 1,
    }

    def __init__(self):
        """Empty init. Use classmethods like .from_config()."""
        pass

    @classmethod
    def from_config(cls, cfg, **kwargs):
        """
        Build one TramModel per node based on configuration and kwargs.
        Kwargs can be scalars (applied to all nodes) or dicts {node: value}.
        """
        self = cls()
        self.cfg = cfg
        self.nodes_dict = self.cfg.conf_dict["nodes"] 

        # merge defaults with user overrides
        settings = dict(cls.DEFAULTS_CONFIG)
        settings.update(kwargs)

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
                        f"Please provide values for all variables."
                    )

        # build one model per node
        self.models = {}
        for node in self.nodes_dict.keys():
            per_node_kwargs = {}
            for k, v in settings.items():
                resolved = v[node] if isinstance(v, dict) else v
                per_node_kwargs[k] = resolved
                self.settings[k][node] = resolved
            print(f"\n[INFO] Building model for node '{node}' with settings: {per_node_kwargs}")
            self.models[node] = get_fully_specified_tram_model(
                node=node,
                configuration_dict=self.cfg.conf_dict,
                **per_node_kwargs
            )
        return self

    def fit(self, td_train_data, td_val_data, **kwargs):
        """
        Fit TRAM models for specified nodes.
        All kwargs can be scalar (applied to all nodes) or dict {node: value}.
        """
        # merge defaults with overrides
        settings = dict(self.DEFAULTS_FIT)
        settings.update(kwargs)

        device = torch.device(
            "cuda" if (settings["device"] == "auto" and torch.cuda.is_available()) else settings["device"]
        )
        train_list = settings["train_list"] or list(self.models.keys())

        def _resolve(key, node):
            val = settings[key]
            return val[node] if isinstance(val, dict) else val

        # store resolved settings for this fit
        self.fit_settings = {k: {} for k in settings.keys()}

        results = {}
        for node in train_list:
            model = self.models[node]

            # resolve per-node settings
            node_epochs = _resolve("epochs", node)
            node_lr = _resolve("learning_rate", node)
            node_debug = _resolve("debug", node)
            node_save_linear_shifts = _resolve("save_linear_shifts", node)
            node_verbose = _resolve("verbose", node)

            # record them
            self.fit_settings["epochs"][node] = node_epochs
            self.fit_settings["learning_rate"][node] = node_lr
            self.fit_settings["debug"][node] = node_debug
            self.fit_settings["save_linear_shifts"][node] = node_save_linear_shifts
            self.fit_settings["verbose"][node] = node_verbose

            # resolve optimizer
            if settings["optimizers"] and node in settings["optimizers"]:
                optimizer = settings["optimizers"][node]
            else:
                optimizer = Adam(model.parameters(), lr=node_lr)
            self.fit_settings["optimizers"][node] = optimizer

            # resolve scheduler
            if settings["schedulers"] and node in settings["schedulers"]:
                scheduler = settings["schedulers"][node]
            else:
                scheduler = None
            self.fit_settings["schedulers"][node] = scheduler

            # grab loaders
            train_loader = td_train_data.loaders[node]
            val_loader = td_val_data.loaders[node]

            try:
                EXPERIMENT_DIR = self.cfg.conf_dict["PATHS"]["EXPERIMENT_DIR"]
                NODE_DIR = os.path.join(EXPERIMENT_DIR, f"{node}")
                # print(f"[INFO] NODE_DIR : {NODE_DIR}")
            except Exception:
                NODE_DIR = os.path.join("models", node)
                print("[WARNING] No log directory specified in config, saving to default location.")
            os.makedirs(NODE_DIR, exist_ok=True)
            self.fit_settings["NODE_DIR"] = {node: NODE_DIR}

            if node_verbose:
                print(f"\n[INFO] Training node '{node}' for {node_epochs} epochs on {device}")

            history = train_val_loop(
                node=node,
                target_nodes=self.nodes_dict,
                NODE_DIR=NODE_DIR,
                tram_model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=node_epochs,
                optimizer=optimizer,
                use_scheduler=(scheduler is not None),
                scheduler=scheduler,
                save_linear_shifts=node_save_linear_shifts,
                verbose=node_verbose,
                device=device,
                debug=node_debug,
            )
            results[node] = history

        return results


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

            all_latents_df = create_latent_df_for_full_dag(
                configuration_dict=self.cfg.conf_dict,
                EXPERIMENT_DIR=EXPERIMENT_DIR,
                df=df,
                verbose=verbose,
            )

            return all_latents_df

    def sample(
        self,
        do_interventions: dict = None,
        predefined_latent_samples_df: pd.DataFrame = None,
        number_of_samples: int = 10_000,
        batch_size: int = 32,
        delete_all_previously_sampled: bool = True,
        verbose: bool = False,
        debug: bool = False,
    ):
        """
        Sample from the DAG using trained TRAM models.

        Parameters
        ----------
        do_interventions : dict, optional
            Mapping of node names to fixed values. Example: {'x1': 1.0}.
        predefined_latent_samples_df : pd.DataFrame, optional
            DataFrame with predefined latent U's. Must contain columns "{node}_U".
        number_of_samples : int, default=10_000
            Number of samples to draw if no predefined latents are given.
        batch_size : int, default=32
            Batch size for DataLoader evaluation during sampling.
        delete_all_previously_sampled : bool, default=True
            Whether to remove existing sampled.pt/latents.pt files before resampling.
        verbose : bool, default=True
            Print high-level progress messages ([INFO]).
        debug : bool, default=False
            Print detailed debug messages ([DEBUG]) in addition to [INFO].

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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sampled_by_node, latents_by_node = sample_full_dag(
            configuration_dict=self.cfg.conf_dict,
            EXPERIMENT_DIR=EXPERIMENT_DIR,
            device=device,
            do_interventions=do_interventions or {},
            predefined_latent_samples_df=predefined_latent_samples_df,
            number_of_samples=number_of_samples,
            batch_size=batch_size,
            delete_all_previously_sampled=delete_all_previously_sampled,
            verbose=verbose,
            debug=debug,
        )
        return sampled_by_node, latents_by_node

    def summary(self):
        print("\n[TramDagModel Summary]")
        print("=" * 60)
        for node, model in self.models.items():
            print(f" Node '{node}': {model.__class__.__name__}")
            for k, v in self.settings.items():
                if node in v:
                    print(f"   - {k}: {v[node]}")
        print("=" * 60 + "\n")
