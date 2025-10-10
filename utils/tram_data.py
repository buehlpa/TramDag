
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

import time
from PIL import Image
import pandas as pd
from collections import OrderedDict, defaultdict
import re
import numpy as np


import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
import re

import tqdm





class GenericDataset(Dataset):
    def __init__(
        self,
        df,
        target_col,
        target_nodes=None,
        transform=None,
        return_intercept_shift=True,
        return_y=True,
        debug=False,
        verbose=False,
        **kwargs 
    ):
        """
            A flexible PyTorch Dataset for TRAM-style models, supporting continuous,
            ordinal, and image predictors, with optional intercept/shift decomposition.

            This dataset:
            - Loads samples from a pandas DataFrame.
            - Extracts predictors according to metadata in `all_nodes_dict`.
            - Encodes ordinal/continuous variables, and applies torchvision transforms to images.
            - Optionally splits predictors into intercept and shift groups for TRAM models.
            - Returns (X, y) pairs or just X, depending on configuration.

            Parameters
            ----------
            df : pd.DataFrame
                DataFrame containing predictor and target columns.
            target_col : str
                Name of the target column in `df`.
            all_nodes_dict : dict, optional
                Mapping node_name → metadata dict with keys such as:
                - 'data_type': str (e.g., "ordinal_yo", "continuous", "source")
                - 'levels': int (for ordinal variables, number of levels)
                - 'node_type': str (e.g., "source", "internal")
                - 'parents_datatype': dict mapping parent names → data types
                - 'transformation_terms_in_h()': dict of transformation terms
                - 'transformation_term_nn_models_in_h()': dict of nn model classes
                Used for determining predictors and encoding logic.
            transform : callable, optional
                A torchvision-style transform applied to image predictors.
            return_intercept_shift : bool, default=True
                If True, returns predictors split into:
                - `int_inputs`: intercept features
                - `shifts`: list of shift feature groups
                If False, returns raw feature tuple instead.
            return_y : bool, default=True
                Whether to return the target variable `y` along with the predictors.
            debug : bool, default=False
                Enables verbose logging and attribute inspection during initialization.

            Returns
            -------
            __getitem__ : tuple
                Depending on configuration:
                - If `return_intercept_shift=True` and `return_y=True`:
                ((int_inputs, shifts), y)
                - If `return_intercept_shift=True` and `return_y=False`:
                (int_inputs, shifts)
                - If `return_intercept_shift=False` and `return_y=True`:
                (features, y)
                - If `return_intercept_shift=False` and `return_y=False`:
                features

            Features
            --------
            - Continuous predictors are returned as float tensors.
            - Ordinal predictors (with 'xn' in data type) are one-hot encoded.
            - Image predictors are loaded via PIL and passed through `transform` if given.
            - Source nodes can return a simple intercept (constant 1.0).

            Notes
            -----
            - Validates target column existence, data types, and ordinal encoding.
            - Ensures ordinal predictors are zero-indexed or properly scaled.
            - Automatically inserts a simple intercept if no explicit intercept term is found.
            - Groups transformation terms into intercept and shift components.

            Examples
            --------
            >>> dataset = GenericDataset(
            ...     df=my_df,
            ...     target_col="y",
            ...     all_nodes_dict=config["nodes"],
            ...     transform=my_transform,
            ...     return_intercept_shift=True,
            ...     debug=True
            ... )
            >>> (int_in, shifts), y = dataset[0]
            >>> int_in.shape, [s.shape for s in shifts], y.shape
        """
        # set kwargs to class attributes
        for k, v in kwargs.items():
            if k in ["debug", "verbose"]:
                raise ValueError(f"{k} is an explicit argument; pass it directly.")
            setattr(self, k, v)

        
        # initialize vebosity and debug
        self._set_verbosity(verbose)
        self._set_debug(debug)

        # set attributes via dedicated setters
        self._set_df(df)
        self._set_target_col(target_col)
        self._set_all_nodes_dict(target_nodes)
        self._set_ordered_parents_datatype_and_transformation_terms()

        self._set_predictors()
        self._set_transform(transform)

        self._set_h_needs_simple_intercept()
        self._set_target_data_type()
        self._set_target_num_classes()
        self.return_intercept_shift = return_intercept_shift
        self.return_y = return_y

        # intercept and shift
        if self.return_intercept_shift:
            self._set_intercept_shift_indexes()

        # source/ordinal
        self._set_target_is_source()
        self._set_ordinal_numal_classes()

        # checks
        self._check_multiclass_predictors_of_df()
        self._check_ordinal_levels()

        # precompute tensors
        self._precompute_all()

                # constant for SI
        self._const_one = torch.tensor(1.0)
        self._timing_print_counter = 0
        

        
        if self.debug or self.verbose:
            print(f"[INFO] ------ Initalized all attributes of Genericdataset ------")
    
    
    # Setter methods
    def _set_ordered_parents_datatype_and_transformation_terms(self):
        # set correctly ordered parents_datatype_dict and transformation_terms_preprocessing such that they are aligned with the model intake order
        
        ordered_parents_datatype, ordered_transformation_terms_in_h, _ =self._ordered_parents(self.target_col, self.all_nodes_dict)
        if not isinstance(ordered_parents_datatype, dict):
            raise TypeError(f"parents_datatype_dict must be dict, got {type(ordered_parents_datatype)}")
        self.parents_datatype_dict = ordered_parents_datatype
        
        if self.debug:
            print(f"[DEBUG] Set parents_datatype_dict: type={type(self.parents_datatype_dict)}, keys={list(self.parents_datatype_dict.keys())}")

        if ordered_transformation_terms_in_h is None:
            ordered_transformation_terms_in_h = {}
        if not isinstance(ordered_transformation_terms_in_h, dict):
            raise TypeError(f"transformation_terms_in_h must be dict, got {type(ordered_transformation_terms_in_h)}")
        self.transformation_terms_preprocessing = list(ordered_transformation_terms_in_h.values())
        if self.debug:
            print(f"[DEBUG] Set transformation_terms_preprocessing: type={type(self.transformation_terms_preprocessing)}, value={self.transformation_terms_preprocessing}")
    
    
    def _set_debug(self, debug):
        if not isinstance(debug, bool):
            raise TypeError(f"debug must be bool, got {type(debug)}")
        self.debug = debug
        if self.verbose:
            print(f"[INFO] ------ Debug set to true ------")
            
    def _set_verbosity(self, verbose):
        if not isinstance(verbose, bool):
            raise TypeError(f"debug must be bool, got {type(verbose)}")
        self.verbose = verbose


    def _set_df(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a pandas DataFrame, got {type(df)}")
        self.df = df.reset_index(drop=True)
        if self.debug:
            print(f"[DEBUG] Set df: type={type(self.df)}, shape={self.df.shape}")


    def _set_target_col(self, target_col):
        if not isinstance(target_col, str):
            raise TypeError(f"target_col must be str, got {type(target_col)}")
        
        if target_col not in self.df.columns:
            print(
                f"[WARNING] target_col '{target_col}' not in DataFrame columns — is this intended to be used as a Sampler?")
            if self.debug:
                print(f"[DEBUG] target_col '{target_col}' not found in DataFrame columns")
            # Still set it in case it's needed for Sampler or other logic
            self.target_col = target_col
            return

        self.target_col = target_col
        if self.debug:
            print(f"[DEBUG] Set target_col: type={type(self.target_col)}, value={self.target_col}")


    def _set_all_nodes_dict(self, all_nodes_dict):
        if all_nodes_dict is None:
            all_nodes_dict = {}
        if not isinstance(all_nodes_dict, dict):
            raise TypeError(f"all_nodes_dict must be dict, got {type(all_nodes_dict)}")
        self.all_nodes_dict = all_nodes_dict
        if self.debug:
            print(f"[DEBUG] Set all_nodes_dict: type={type(self.all_nodes_dict)}, keys={list(self.all_nodes_dict.keys())}")


    def _set_predictors(self):
        self.predictors = list(self.parents_datatype_dict.keys())
        if self.debug:
            print(f"[DEBUG] Set predictors: type={type(self.predictors)}, value={self.predictors}")

    def _set_transform(self, transform):
        self.transform = transform
        if self.debug:
            print(f"[DEBUG] Set transform: type={type(self.transform)}, value={self.transform}")


    def _set_h_needs_simple_intercept(self):
        self.h_needs_simple_intercept = all('i' not in str(v) for v in self.transformation_terms_preprocessing)
        if self.debug:
            print(f"[DEBUG] Set h_needs_simple_intercept: type={type(self.h_needs_simple_intercept)}, value={self.h_needs_simple_intercept}")

    def _set_target_data_type(self):
        dtype = self.all_nodes_dict.get(self.target_col, {}).get('data_type', '')
        self.target_data_type = dtype.lower()
        if self.debug:
            print(f"[DEBUG] Set target_data_type: type={type(self.target_data_type)}, value={self.target_data_type}")

    def _set_target_num_classes(self):
        levels = self.all_nodes_dict.get(self.target_col, {}).get('levels')
        if levels is not None and not isinstance(levels, int):
            raise TypeError(f"levels must be int, got {type(levels)}")
        self.target_num_classes = levels
        if self.debug:
            print(f"[DEBUG] Set target_num_classes: type={type(self.target_num_classes)}, value={self.target_num_classes}")
            
    def _set_target_is_source(self):
        # determine if target node is a source
        node_type = self.all_nodes_dict.get(self.target_col, {}).get('node_type', '')
        if not isinstance(node_type, str):
            raise TypeError(f"node_type metadata must be str, got {type(node_type)}")
        self.target_is_source = node_type.lower() == 'source'
        if self.debug:
            print(f"[DEBUG] Set target_is_source: type={type(self.target_is_source)}, value={self.target_is_source}")

    def _set_ordinal_numal_classes(self):
        """
        Compute the number of classes for each ordinal-Xn predictor.
        Prefer the configured 'levels' value from all_nodes_dict, but
        compare against observed unique levels in the DataFrame.
        If they don't match, emit a warning but continue.
        """
        mapping = {}
        for v in self.predictors:
            dt = self.parents_datatype_dict[v]
            if not isinstance(dt, str):
                raise TypeError(f"datatype for predictor '{v}' must be str, got {type(dt)}")

            if 'ordinal' in dt.lower() and 'xn' in dt.lower():
                if v not in self.df.columns:
                    raise ValueError(f"Predictor column '{v}' not in DataFrame")

                # Configured number of levels
                cfg_levels = self.all_nodes_dict.get(v, {}).get('levels', None)
                # Observed unique values
                observed_levels = int(self.df[v].nunique())
                # Decide what to use
                if cfg_levels is not None:
                    mapping[v] = cfg_levels
                    if observed_levels != cfg_levels:
                        print(
                            f"[WARNING] Ordinal '{v}' has {observed_levels} unique values in data "
                            f"but is configured for {cfg_levels} levels — using configured value."
                        )
                else:
                    mapping[v] = observed_levels
                    print(
                        f"[WARNING] Ordinal '{v}' has no 'levels' specified in config — "
                        f"using observed {observed_levels} levels from data."
                    )
        self.ordinal_num_classes = mapping

        if self.debug:
            print(f"[DEBUG] Set ordinal_num_classes: type={type(self.ordinal_num_classes)}, value={self.ordinal_num_classes}")

    def get_sort_key(self, val):
            """Map each transformation value to a sorting index."""
            order = ['ci', 'ciXX', 'cs', 'csXX', 'ls']
            
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

    def _ordered_parents(self,node, target_dict) -> dict:
        
        """
        Orders the transformation terms and their corresponding data types and nn models used for the models and the dataloader
        """
        # Extract dictionaries
        transformation_terms = target_dict[node]['transformation_terms_in_h()']
        datatype_dict = target_dict[node]['parents_datatype']
        nn_models_dict = target_dict[node]['transformation_term_nn_models_in_h()']

        # Sort the items based on the transformation_terms
        sorted_keys = sorted(transformation_terms.keys(), key=lambda k: self.get_sort_key(transformation_terms[k]))

        # Create ordered dicts
        ordered_transformation_terms_in_h = OrderedDict((k, transformation_terms[k]) for k in sorted_keys)
        ordered_parents_datatype = OrderedDict((k, datatype_dict[k]) for k in sorted_keys)
        ordered_transformation_term_nn_models_in_h = OrderedDict((k, nn_models_dict[k]) for k in sorted_keys)

        return ordered_parents_datatype, ordered_transformation_terms_in_h, ordered_transformation_term_nn_models_in_h

    # Core methods
    def _set_intercept_shift_indexes(self):
        if not any('ci' in t for t in self.transformation_terms_preprocessing):
            self.transformation_terms_preprocessing.insert(0, 'si')
            if self.debug:
                print("[DEBUG] Inserted simple intercept term 'si'")
        self.intercept_indices = [i for i, term in enumerate(self.transformation_terms_preprocessing)
                                    if term.startswith(('si', 'ci'))]
        if self.debug:
            print(f"[DEBUG] Intercept indices: {self.intercept_indices}")
        self.shift_groups_indices = []
        current_key = None
        for i, term in enumerate(self.transformation_terms_preprocessing):
            if term.startswith(('cs', 'ls')):
                if len(term) > 2 and term[2].isdigit():
                    grp = term[:3]
                    if not self.shift_groups_indices or current_key != grp:
                        self.shift_groups_indices.append([i])
                        current_key = grp
                    else:
                        self.shift_groups_indices[-1].append(i)
                else:
                    self.shift_groups_indices.append([i])
                    current_key = None
        if self.debug:
            print(f"[DEBUG] Shift group indices: {self.shift_groups_indices}")

    def _precompute_all(self):
        """
        Precompute all tensors for numerical/ordinal predictors + y.
        Store them in self.X_list (aligned with self.predictors).
        """
        self.X_list = []

        for var in self.predictors:
            dt = self.parents_datatype_dict[var].lower()

            if dt in ("continous",) or "xc" in dt:
                arr = self.df[var].to_numpy(dtype=np.float32)
                self.X_list.append(torch.from_numpy(arr).unsqueeze(1))

            elif "ordinal" in dt and "xn" in dt:
                c = self.ordinal_num_classes[var]
                vals = self.df[var].to_numpy(dtype=np.int64)
                onehot = F.one_hot(torch.from_numpy(vals), num_classes=c).float()
                self.X_list.append(onehot)

            else:
                # keep image paths as-is (handled in __getitem__)
                self.X_list.append(self.df[var].tolist())

        if self.return_y and self.target_col in self.df.columns:
            dtype = self.target_data_type
            if dtype in ("continous",) or "yc" in dtype:
                self.y_tensor = torch.tensor(self.df[self.target_col].to_numpy(dtype=np.float32))
            elif self.target_num_classes:
                vals = self.df[self.target_col].to_numpy(dtype=np.int64)
                self.y_tensor = F.one_hot(torch.from_numpy(vals), num_classes=self.target_num_classes).float()
            else:
                self.y_tensor = None
        else:
            self.y_tensor = None
    
    def _preprocess_inputs(self, x):
        # Collect intercept inputs
        if not self.intercept_indices:
            raise ValueError("No intercept tensors found!")

        # Concatenate all intercept tensors along feature dimension
        int_inputs = torch.cat(
            [x[i].reshape(x[i].shape[0], -1) for i in self.intercept_indices],
            dim=1
        )

        # Build shift groups if present
        if self.shift_groups_indices:
            shifts = [
                torch.cat([x[i].reshape(x[i].shape[0], -1) for i in grp], dim=1)
                for grp in self.shift_groups_indices
            ]
        else:
            shifts = None

        return int_inputs, shifts

    def _transform_y(self, row):
        if self.target_data_type in ('continous',) or 'yc' in self.target_data_type:
            return torch.tensor(row[self.target_col], dtype=torch.float32)
        elif self.target_num_classes:
            yi = int(row[self.target_col])
            return F.one_hot(
                torch.tensor(yi, dtype=torch.long),
                num_classes=self.target_num_classes
            ).float().squeeze()
        else:
            raise ValueError(
                f"Cannot encode target '{self.target_col}': {self.target_data_type}/{self.target_num_classes}"
            )
    #checks
    
    def _check_multiclass_predictors_of_df(self):
        for v in self.predictors:
            dt = self.parents_datatype_dict[v].lower()
            if 'ordinal' in dt and 'xn' in dt:
                vals = set(self.df[v].dropna().unique())
                if vals != set(range(len(vals))):
                    raise ValueError(
                        f"Ordinal predictor '{v}' must be zero‑indexed; got {sorted(vals)}"
                    )
        if self.debug:
            print(f"[DEBUG] _check_multiclass_predictors_of_df: checked multiclass_predicitors passed")
                

    def _check_ordinal_levels(self):
        ords = []
        # include target if it’s ordinal
        if 'ordinal' in self.all_nodes_dict.get(self.target_col, {}).get('data_type', '').lower():
            ords.append(self.target_col)
        # include any xn‐encoded predictors
        ords += [
            v for v in self.predictors
            if 'ordinal' in self.parents_datatype_dict[v].lower()
            and 'xn' in self.parents_datatype_dict[v].lower()
        ]

        for v in ords:
            if v not in self.df.columns:
                if self.debug:
                    print(f"[DEBUG] _check_ordinal_levels: Skipping '{v}' as it's not in the DataFrame")
                continue

            lvl = self.all_nodes_dict[v].get('levels')
            if lvl is None:
                raise ValueError(f"Ordinal '{v}' missing 'levels' metadata.")

            # grab unique values as floats
            uniq = np.array(sorted(self.df[v].dropna().unique()), dtype=float)

            # expected patterns
            expected_int    = np.arange(lvl, dtype=float)         # 0,1,...,n-1
            expected_scaled = np.arange(lvl, dtype=float) / lvl   # 0/n,1/n,...,(n-1)/n

            # allow either exact ints or approximate scaled floats
            if np.array_equal(uniq, expected_int):
                # perfectly integer-encoded
                if self.debug:
                    print(f"[DEBUG] Ordinal '{v}' matches expected integer encoding {expected_int.tolist()}.")
            elif np.allclose(uniq, expected_scaled, atol=1e-8):
                # scaled floats encoding
                if self.debug:
                    print(f"[DEBUG] Ordinal '{v}' matches expected scaled encoding {expected_scaled.tolist()}.")
            elif len(uniq) == 1:
                print(f"[WARNING] Ordinal '{v}' has constant value {uniq.tolist()} — may reduce model diversity.")
            else:
                print(
                    f"[WARNING] Ordinal '{v}' values {uniq.tolist()} do not match expected "
                    f"integers {expected_int.tolist()} or scaled floats {expected_scaled.tolist()}."
                )

        if self.debug:
            print(f"[DEBUG] _check_ordinal_levels: checked ordinal levels passed")


    def __len__(self):
        return len(self.df)

    ##############################################  OLD VERSION OF __getitem__ working 8.10
    def __getitem__(self, idx):
        x_data = []

        # intercept if needed
        if self.h_needs_simple_intercept:
            x_data.append(self._const_one)

        # predictors
        for _, stored in zip(self.predictors, self.X_list):
            if isinstance(stored, torch.Tensor):
                x_data.append(stored[idx])
            else:
                img = Image.open(stored[idx])
                img = img.convert("RGB")
                if self.transform:
                    img = self.transform(img)
                x_data.append(img)

        # intercept/shift decomposition
        # batched = [x.unsqueeze(0) for x in x_data] # old version
        
        batched = []
        for x in x_data:
            if isinstance(x, torch.Tensor):
                if x.ndim == 0:  # scalar (e.g., intercept)
                    batched.append(x.view(1, 1))
                elif x.ndim == 1:
                    batched.append(x.unsqueeze(0))
                else:
                    batched.append(x.unsqueeze(0) if x.ndim == 3 else x)
            else:
                batched.append(x)
        
        int_in, shifts = self._preprocess_inputs(batched)

        int_in = int_in.squeeze(0)
        shifts = [] if shifts is None else [s.squeeze(0) for s in shifts]

        out = ((int_in, shifts), self.y_tensor[idx]) if self.return_y else (int_in, shifts)
        return out

    def save_precomputed(self, path):
        items = []
        for i in range(len(self)):
            items.append(self[i])
        torch.save(items, path)
        if self.debug or self.verbose:
            print(f"[INFO] Saved {len(items)} samples to {path}")



### get Precomputed dataset class , minimal dataaset class to load precomputed data
class GenericDatasetPrecomputed(torch.utils.data.Dataset):
    def __init__(self, path):
        self.items = torch.load(path, weights_only=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
    


def get_dataloader(
    node,
    target_nodes,
    train_df=None,
    val_df=None,
    batch_size=32,
    return_intercept_shift=False,
    transform=None,
    debug=False,
    verbose=False,
    **kwargs,
):
    """
    Build train/val dataloaders for TRAM models.
    """

    if transform is None:
        transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

    train_loader, val_loader = None, None

    if train_df is not None:
        if verbose or debug:
            print("[INFO] Building training dataset...")
        train_ds = GenericDataset(
            train_df,
            target_col=node,
            target_nodes=target_nodes,
            transform=transform,
            return_intercept_shift=return_intercept_shift,
            debug=debug,
            verbose=verbose,
            **kwargs,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
        )

    if val_df is not None:
        if verbose or debug:
            print("[INFO] Building validation dataset...")
        val_ds = GenericDataset(
            val_df,
            target_col=node,
            target_nodes=target_nodes,
            transform=transform,
            return_intercept_shift=return_intercept_shift,
            debug=debug,
            verbose=verbose,
            **kwargs,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
        )

    if train_loader is None and val_loader is None:
        raise ValueError("Both train_df and val_df are None → no dataloaders created.")

    if debug:
        print(
            "[DEBUG] get_dataloader finished. "
            f"Train loader: {train_loader is not None}, "
            f"Val loader: {val_loader is not None}"
        )

    return train_loader, val_loader