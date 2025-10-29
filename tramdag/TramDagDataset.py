import inspect
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader

from .utils.tram_data import GenericDataset, GenericDatasetPrecomputed



class TramDagDataset(Dataset):
    
    """
    TramDagDataset
    ==============

    The `TramDagDataset` class handles structured data preparation for TRAM-DAG
    models. It wraps a pandas DataFrame together with its configuration and provides
    utilities for scaling, transformation, and efficient DataLoader construction
    for each node in a DAG-based configuration.

    ---------------------------------------------------------------------
    Core Responsibilities
    ---------------------------------------------------------------------
    - Validate and store configuration metadata (`TramDagConfig`).
    - Manage per-node settings for DataLoader creation (batch size, shuffling, workers).
    - Compute scaling information (quantile-based min/max).
    - Optionally precompute and cache dataset representations.
    - Expose PyTorch Dataset and DataLoader interfaces for model training.

    ---------------------------------------------------------------------
    Key Attributes
    ---------------------------------------------------------------------
    - **df** : pandas.DataFrame  
      The dataset content used for building loaders and computing scaling.

    - **cfg** : TramDagConfig  
      Configuration object defining nodes and variable metadata.

    - **nodes_dict** : dict  
      Mapping of variable names to node specifications from the configuration.

    - **loaders** : dict  
      Mapping of node names to `torch.utils.data.DataLoader` instances or `GenericDataset` objects.

    - **DEFAULTS** : dict  
      Default DataLoader and dataset-related settings (e.g., batch_size, shuffle, num_workers, etc.).

    ---------------------------------------------------------------------
    Main Methods
    ---------------------------------------------------------------------
    - **from_dataframe(df, cfg, **kwargs)**  
      Construct the dataset directly from a pandas DataFrame.

    - **compute_scaling(df=None, write=True)**  
      Compute per-variable min/max scaling values from data.

    - **summary()**  
      Print dataset overview including shape, dtypes, statistics, and node settings.

    ---------------------------------------------------------------------
    Notes
    ---------------------------------------------------------------------
    - Intended for training data; `compute_scaling()` should use only training subsets.
    - Compatible with both CPU and GPU DataLoader options.
    - Strict validation of keyword arguments against `DEFAULTS` prevents silent misconfiguration.

    ---------------------------------------------------------------------
    Example
    ---------------------------------------------------------------------
    >>> cfg = TramDagConfig.from_json("config.json")
    >>> dataset = TramDagDataset.from_dataframe(train_df, cfg, batch_size=1024, debug=True)
    >>> dataset.summary()
    >>> minmax = dataset.compute_scaling(train_df)
    >>> loader = dataset.loaders["variable_x"]
    >>> next(iter(loader))
    """

    DEFAULTS = {
        "batch_size": 32_000,
        "shuffle": True,
        "num_workers": 4,
        "pin_memory": True,
        "return_intercept_shift": True,
        "debug": False,
        "transform": None,
        "use_dataloader": True,
        "use_precomputed": False, 
        # DataLoader extras
        "sampler": None,
        "batch_sampler": None,
        "collate_fn": None,
        "drop_last": False,
        "timeout": 0,
        "worker_init_fn": None,
        "multiprocessing_context": None,
        "generator": None,
        "prefetch_factor": 2,
        "persistent_workers": True,
        "pin_memory_device": "",
    }

    def __init__(self):
        """Empty init. Use classmethods like .from_dataframe()."""
        pass

    @classmethod
    def from_dataframe(cls, df, cfg, **kwargs):
        """Create a TramDagDataset instance directly from a pandas DataFrame.

        This class method constructs and initializes a dataset using the provided
        DataFrame and configuration object. It validates keyword arguments against
        class defaults, merges them into a resolved settings dictionary, and builds
        the internal dataloaders.

        Args:
            df: Input pandas DataFrame containing the dataset.
            cfg: TramDagConfig instance defining variable structure and metadata.
            **kwargs: Optional keyword overrides for dataset defaults. All keys
                must be defined in `cls.DEFAULTS`; otherwise, a ValueError is raised.

        Returns:
            TramDagDataset: An initialized dataset instance.

        Raises:
            TypeError: If `df` is not a pandas DataFrame.
            ValueError: If any keyword arguments are not present in `DEFAULTS`.

        Notes:
            - Validates `kwargs` via `_validate_kwargs()` to ensure strict adherence
            to supported parameters.
            - Prints full resolved settings if `debug=True`.
            - Automatically infers the variable name of the provided DataFrame
            for clearer warning messages.
            - Issues a warning if the inferred DataFrame name suggests validation/test
            data while `shuffle=True` is set.
            - Applies resolved settings and constructs internal dataloaders via
            `_apply_settings()` and `_build_dataloaders()`.
        """
        self = cls()
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"[ERROR] df must be a pandas DataFrame, but got {type(df)}")

        # validate kwargs
        # TODO adjust validation such thath it works when called from tramdagmodel
        #self._validate_kwargs(kwargs, context="from_dataframe")
        
        # merge defaults with overrides
        settings = dict(cls.DEFAULTS)
        settings.update(kwargs)

        # store config and verify
        self.cfg = cfg
        self.cfg._verify_completeness()

        # ouptu all setttings if debug
        if settings.get("debug", False):
            print("[DEBUG] TramDagDataset.from_dataframe() settings (after defaults + overrides):")
            for k, v in settings.items():
                print(f"    {k}: {v}")

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

        # call again to ensure Warning messages if ordinal vars have missing levels
        self.df = df.copy()
        self._apply_settings(settings)
        self._build_dataloaders()
        return self

    def compute_scaling(self, df: pd.DataFrame = None, write: bool = True):
        """Compute variable-wise scaling parameters from data.

        Derives approximate minimum and maximum values for each variable based on
        the 5th and 95th percentiles of the provided DataFrame. Intended for use
        with training data to establish normalization or clipping ranges.

        Args:
            df: Optional pandas DataFrame containing the data used to compute scaling.
                If not provided, uses the dataset's internal `self.df`.
            write: Unused placeholder argument for interface consistency. Reserved
                for potential future functionality (e.g., auto-saving scaling info).

        Returns:
            dict: Mapping of variable names to [min, max] values derived from quantiles.

        Notes:
            - Prints debug information when `self.debug=True`.
            - The returned min/max correspond to the 0.05 and 0.95 quantiles,
            not the absolute extrema, to reduce outlier influence.
            - Intended for normalization or scaling within the training pipeline.
        """
        if self.debug:
            print("[DEBUG] Make sure to provide only training data to compute_scaling!")     
        if df is None:
            df = self.df
            if self.debug:
                print("[DEBUG] No DataFrame provided, using internal df.")
        quantiles = df.quantile([0.05, 0.95])
        min_vals = quantiles.loc[0.05]
        max_vals = quantiles.loc[0.95]
        minmax_dict = pd.concat([min_vals, max_vals], axis=1).T.to_dict('list')
        return minmax_dict

    def summary(self):
        """Print a structured overview of the dataset and configuration.

        Displays key properties of the internal DataFrame, including shape,
        data types, and descriptive statistics, as well as node-level settings
        and DataLoader configurations.

        Sections printed:
            1. **DataFrame** — Shape, column names, preview (head), dtypes, and summary statistics.
            2. **Configuration** — Number of nodes, DataLoader usage mode, and precomputation status.
            3. **Node Settings** — Per-node DataLoader and dataset parameters (batch_size, shuffle, etc.).
            4. **DataLoaders** — Types and lengths of instantiated loaders per node.

        Notes:
            - Intended for quick inspection and debugging of dataset setup.
            - Automatically adapts to dict- or scalar-style configuration attributes.
            - Prints additional debug information when `self.debug=True`.
        """
        
        print("\n[TramDagDataset Summary]")
        print("=" * 60)

        print("\n[DataFrame]")
        print("Shape:", self.df.shape)
        print("Columns:", list(self.df.columns))
        print("\nHead:")
        print(self.df.head())

        print("\nDtypes:")
        print(self.df.dtypes)

        print("\nDescribe:")
        print(self.df.describe(include="all"))

        print("\n[Configuration]")
        print(f"Nodes: {len(self.nodes_dict)}")
        print(f"Loader mode: {'DataLoader' if self.use_dataloader else 'Direct dataset'}")
        print(f"Precomputed: {getattr(self, 'use_precomputed', False)}")

        print("\n[Node Settings]")
        for node in self.nodes_dict.keys():
            batch_size = self.batch_size[node] if isinstance(self.batch_size, dict) else self.batch_size
            shuffle_flag = self.shuffle[node] if isinstance(self.shuffle, dict) else self.shuffle
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

        if hasattr(self, "loaders"):
            print("\n[DataLoaders]")
            for node, loader in self.loaders.items():
                try:
                    length = len(loader)
                except Exception:
                    length = "?"
                print(f"  {node}: {type(loader).__name__}, len={length}")

        print("=" * 60 + "\n")

    def _validate_kwargs(self, kwargs: dict, defaults_attr: str = "DEFAULTS", context: str = None):
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

    def _apply_settings(self, settings: dict):
        # store everything into self (makes it easy to pass into DataLoader later)
        for k, v in settings.items():
            setattr(self, k, v)

        self.nodes_dict = self.cfg.conf_dict["nodes"]

        # validate only the most important ones
        for name in ["batch_size", "shuffle", "num_workers", "pin_memory",
                     "return_intercept_shift", "debug", "transform"]:
            self._check_keys(name, getattr(self, name))

    def _build_dataloaders(self):
        """Build node-specific dataloaders or raw datasets depending on settings."""
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

        ########## QUICK PATCH 
            if hasattr(self, "use_precomputed") and self.use_precomputed:
                os.makedirs("temp", exist_ok=True) 
                pth = os.path.join("temp", "precomputed.pt")

                if hasattr(ds, "save_precomputed") and callable(getattr(ds, "save_precomputed")):
                    ds.save_precomputed(pth)
                    ds = GenericDatasetPrecomputed(pth)
                else:
                    print("[WARNING] Dataset has no 'save_precomputed()' method — skipping precomputation.")


            if self.use_dataloader:
                # resolve per-node overrides
                kwargs = {
                    "batch_size": self.batch_size[node] if isinstance(self.batch_size, dict) else self.batch_size,
                    "shuffle": self.shuffle[node] if isinstance(self.shuffle, dict) else self.shuffle,
                    "num_workers": self.num_workers[node] if isinstance(self.num_workers, dict) else self.num_workers,
                    "pin_memory": self.pin_memory[node] if isinstance(self.pin_memory, dict) else self.pin_memory,
                    "sampler": self.sampler,
                    "batch_sampler": self.batch_sampler,
                    "collate_fn": self.collate_fn,
                    "drop_last": self.drop_last,
                    "timeout": self.timeout,
                    "worker_init_fn": self.worker_init_fn,
                    "multiprocessing_context": self.multiprocessing_context,
                    "generator": self.generator,
                    "prefetch_factor": self.prefetch_factor,
                    "persistent_workers": self.persistent_workers,
                    "pin_memory_device": self.pin_memory_device,
                }
                self.loaders[node] = DataLoader(ds, **kwargs)
            else:
                self.loaders[node] = ds
                
        if hasattr(self, "use_precomputed") and self.use_precomputed:
            if os.path.exists(pth):
                try:
                    os.remove(pth)
                    if self.debug:
                        print(f"[INFO] Removed existing precomputed file: {pth}")
                except Exception as e:
                    print(f"[WARNING] Could not remove {pth}: {e}")

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

    def __getitem__(self, idx):
        return self.df.iloc[idx].to_dict()

    def __len__(self):
        return len(self.df)