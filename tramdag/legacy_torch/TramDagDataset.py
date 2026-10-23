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


import inspect
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader

from .utils.data import GenericDataset, GenericDatasetPrecomputed



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
        """
        Initialize an empty TramDagDataset shell.

        Notes
        -----
        This constructor does not attach data or configuration. Use
        `TramDagDataset.from_dataframe` to obtain a ready-to-use instance.
        """
        pass

    @classmethod
    def from_dataframe(cls, df, cfg, **kwargs):
        """
        Create a TramDagDataset instance directly from a pandas DataFrame.

        This classmethod:

        1. Validates keyword arguments against `DEFAULTS`.
        2. Merges user overrides with defaults into a resolved settings dict.
        3. Stores the configuration and verifies its completeness.
        4. Applies settings to the instance.
        5. Builds per-node datasets and DataLoaders.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing the dataset.
        cfg : TramDagConfig
            Configuration object defining nodes and variable metadata.
        **kwargs
            Optional overrides for `DEFAULTS`. All keys must exist in
            `TramDagDataset.DEFAULTS`. Common keys include:

            batch_size : int
                Batch size for DataLoaders.
            shuffle : bool
                Whether to shuffle samples per epoch.
            num_workers : int
                Number of DataLoader workers.
            pin_memory : bool
                Whether to pin memory for faster host-to-device transfers.
            return_intercept_shift : bool
                Whether datasets should return intercept/shift information.
            debug : bool
                Enable debug printing.
            transform : callable or dict or None
                Optional transform(s) applied to samples.
            use_dataloader : bool
                If True, construct DataLoaders; else store raw Dataset objects.
            use_precomputed : bool
                If True, precompute dataset representation to disk and reload it.

        Returns
        -------
        TramDagDataset
            Initialized dataset instance.

        Raises
        ------
        TypeError
            If `df` is not a pandas DataFrame.
        ValueError
            If unknown keyword arguments are provided (when validation is enabled).

        Notes
        -----
        If `shuffle=True` and the inferred variable name of `df` suggests
        validation/test data (e.g. "val", "test"), a warning is printed.
        """
        self = cls()
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"[ERROR] df must be a pandas DataFrame, but got {type(df)}")

        # validate kwargs
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
        """
        Compute variable-wise scaling parameters from data.

        Per variable, this method computes approximate minimum and maximum
        values using the 5th and 95th percentiles. This is typically used
        to derive robust normalization/clipping ranges from training data.

        Parameters
        ----------
        df : pandas.DataFrame or None, optional
            DataFrame used to compute scaling. If None, `self.df` is used.
        write : bool, optional
            Unused placeholder for interface compatibility with other components.
            Kept for potential future extensions. Default is True.

        Returns
        -------
        dict
            Mapping `{column_name: [min_value, max_value]}`, where values
            are derived from the 0.05 and 0.95 quantiles.

        Notes
        -----
        If `self.debug` is True, the method emits debug messages about the
        data source. Only training data should be used to avoid leakage.
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
        """
        Print a structured overview of the dataset and configuration.

        The summary includes:

        1. DataFrame information:
        - Shape
        - Columns
        - Head (first rows)
        - Dtypes
        - Descriptive statistics

        2. Configuration overview:
        - Number of nodes
        - Loader mode (DataLoader vs. raw Dataset)
        - Precomputation status

        3. Node settings:
        - Batch size
        - Shuffle flag
        - num_workers
        - pin_memory
        - return_intercept_shift
        - debug
        - transform

        4. DataLoader overview:
        - Type and length of each loader.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        Intended for quick inspection and debugging. Uses `print` statements
        and does not return structured metadata.
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
        """
        Validate keyword arguments against a defaults dictionary.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments to validate.
        defaults_attr : str, optional
            Name of the attribute on this class containing the default keys
            (e.g. "DEFAULTS"). Default is "DEFAULTS".
        context : str or None, optional
            Optional context label (e.g. caller name) included in error messages.

        Raises
        ------
        AttributeError
            If the attribute named by `defaults_attr` does not exist.
        ValueError
            If any keys in `kwargs` are not present in the defaults dictionary.
        """
        defaults = getattr(self, defaults_attr, None)
        if defaults is None:
            raise AttributeError(f"{self.__class__.__name__} has no attribute '{defaults_attr}'")

        unknown = set(kwargs) - set(defaults)
        if unknown:
            prefix = f"[{context}] " if context else ""
            raise ValueError(f"{prefix}Unknown parameter(s): {', '.join(sorted(unknown))}")

    def _apply_settings(self, settings: dict):
        """
        Apply resolved settings to the dataset instance.

        This method:

        1. Stores all key–value pairs from `settings` as attributes on `self`.
        2. Extracts `nodes_dict` from the configuration.
        3. Validates that dict-valued core attributes (batch_size, shuffle, etc.)
        have keys matching the node set.

        Parameters
        ----------
        settings : dict
            Resolved settings dictionary, usually built from `DEFAULTS` plus
            user overrides.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any dict-valued core attribute has keys that do not match
            `cfg.conf_dict["nodes"].keys()`.
        """
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
        """
        Check that dict-valued attributes use node names as keys.

        Parameters
        ----------
        attr_name : str
            Name of the attribute being checked (for error messages).
        attr_value : Any
            Attribute value. If it is a dict, its keys are validated.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `attr_value` is a dict and its keys do not exactly match
            `cfg.conf_dict["nodes"].keys()`.

        Notes
        -----
        This check prevents partial or mismatched per-node settings such as
        batch sizes or shuffle flags.
        """
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