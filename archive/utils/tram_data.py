
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

from utils import logger


from PIL import Image
import pandas as pd
from collections import OrderedDict, defaultdict
import re
import numpy as np

class GenericDataset(Dataset):
    def __init__(
        self,
        df,
        target_col,
        target_nodes=None,
        transform=None,
        return_intercept_shift=True,
        return_y=True,
        debug=False
    ):
        """
            A flexible PyTorch Dataset for TRAM-style models, supporting continuous,
            ordinal, and image predictors, with optional intercept/shift decomposition.

            This dataset:
            - Loads samples from a pandas DataFrame.
            - Extracts predictors according to metadata in `target_nodes`.
            - Encodes ordinal/continuous variables, and applies torchvision transforms to images.
            - Optionally splits predictors into intercept and shift groups for TRAM models.
            - Returns (X, y) pairs or just X, depending on configuration.

            Parameters
            ----------
            df : pd.DataFrame
                DataFrame containing predictor and target columns.
            target_col : str
                Name of the target column in `df`.
            target_nodes : dict, optional
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
            ...     target_nodes=config["nodes"],
            ...     transform=my_transform,
            ...     return_intercept_shift=True,
            ...     debug=True
            ... )
            >>> (int_in, shifts), y = dataset[0]
            >>> int_in.shape, [s.shape for s in shifts], y.shape
        """
        # initialize debug 
        self._set_debug(debug)

        # set attributes via dedicated setters
        self._set_df(df)
        self._set_target_col(target_col)
        self._set_target_nodes(target_nodes)
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

        if self.debug:
            print(f"[INFO] ------ Initalized all attributes of Genericdataset V6------")
            
    # Setter methods
    def _set_ordered_parents_datatype_and_transformation_terms(self):
        
        ordered_parents_datatype, ordered_transformation_terms_in_h, _ =self._ordered_parents(self.target_col, self.target_nodes)
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
        if self.debug:
            print(f"[INFO] ------ Starting Debug Mode GenericDataset_v6 ------")

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


    def _set_target_nodes(self, target_nodes):
        if target_nodes is None:
            target_nodes = {}
        if not isinstance(target_nodes, dict):
            raise TypeError(f"target_nodes must be dict, got {type(target_nodes)}")
        self.target_nodes = target_nodes
        if self.debug:
            print(f"[DEBUG] Set target_nodes: type={type(self.target_nodes)}, keys={list(self.target_nodes.keys())}")


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
        dtype = self.target_nodes.get(self.target_col, {}).get('data_type', '')
        self.target_data_type = dtype.lower()
        if self.debug:
            print(f"[DEBUG] Set target_data_type: type={type(self.target_data_type)}, value={self.target_data_type}")

    def _set_target_num_classes(self):
        levels = self.target_nodes.get(self.target_col, {}).get('levels')
        if levels is not None and not isinstance(levels, int):
            raise TypeError(f"levels must be int, got {type(levels)}")
        self.target_num_classes = levels
        if self.debug:
            print(f"[DEBUG] Set target_num_classes: type={type(self.target_num_classes)}, value={self.target_num_classes}")
            
    def _set_target_is_source(self):
        # determine if target node is a source
        node_type = self.target_nodes.get(self.target_col, {}).get('node_type', '')
        if not isinstance(node_type, str):
            raise TypeError(f"node_type metadata must be str, got {type(node_type)}")
        self.target_is_source = node_type.lower() == 'source'
        if self.debug:
            print(f"[DEBUG] Set target_is_source: type={type(self.target_is_source)}, value={self.target_is_source}")

    def _set_ordinal_numal_classes(self):
        # compute number of classes for each ordinal-xn predictor
        mapping = {}
        for v in self.predictors:
            dt = self.parents_datatype_dict[v]
            if not isinstance(dt, str):
                raise TypeError(f"datatype for predictor '{v}' must be str, got {type(dt)}")
            if 'ordinal' in dt.lower() and 'xn' in dt.lower():
                if v not in self.df.columns:
                    raise ValueError(f"Predictor column '{v}' not in DataFrame")
                mapping[v] = self.df[v].nunique()
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

    def _preprocess_inputs(self, x):
        its = [x[i] for i in self.intercept_indices]
        if not its:
            raise ValueError("No intercept tensors found!")
        int_inputs = torch.cat([t.view(t.shape[0], -1) for t in its], dim=1)
        shifts = []
        for grp in self.shift_groups_indices:
            parts = [x[i].view(x[i].shape[0], -1) for i in grp]
            shifts.append(torch.cat(parts, dim=1))
        return int_inputs, (shifts if shifts else None)

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
        if 'ordinal' in self.target_nodes.get(self.target_col, {}).get('data_type', '').lower():
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

            lvl = self.target_nodes[v].get('levels')
            if lvl is None:
                raise ValueError(f"Ordinal '{v}' missing 'levels' metadata.")

            # grab unique values as floats
            uniq = np.array(sorted(self.df[v].dropna().unique()), dtype=float)

            # expected patterns
            expected_int    = np.arange(lvl, dtype=float)         # 0,1,...,n-1
            expected_scaled = np.arange(lvl, dtype=float) / lvl   # 0/n,1/n,...,(n-1)/n

            # allow either exact ints or approximate scaled floats
            if np.array_equal(uniq, expected_int):
                # integer‐indexed: OK
                continue
            elif np.allclose(uniq, expected_scaled, atol=1e-8):
                # scaled floats: OK
                continue
            else:
                raise ValueError(
                    f"Ordinal '{v}' values {uniq.tolist()} do not match expected "
                    f"integers {expected_int.tolist()} or scaled floats {expected_scaled.tolist()}."
                )

        if self.debug:
            print(f"[DEBUG] _check_ordinal_levels: checked ordinal levels passed")


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x_data = []

        # for source nodes the dataset gets just a 1.0
        if self.h_needs_simple_intercept:
            x_data.append(torch.tensor(1.0))
        
        # id target is a source node, return the x_data and y
        if self.target_is_source:
            
            if not self.return_intercept_shift:
                if self.return_y:
                    y = self._transform_y(row)
                    return (tuple(x_data), y)
                else:
                    return tuple(x_data)
            
            batched = [x.unsqueeze(0) for x in x_data]
            int_in, shifts = self._preprocess_inputs(batched)
            int_in = int_in.squeeze(0)
            shifts = [] if shifts is None else [s.squeeze(0) for s in shifts]
            if self.return_y:
                y = self._transform_y(row)
                return ((int_in, shifts), y)
            else:
                return (int_in, shifts)

        # predictors X 
        for var in self.predictors:
            dt = self.parents_datatype_dict[var].lower()
            if dt in ('continous',) or 'xc' in dt:
                x_data.append(torch.tensor(row[var], dtype=torch.float32))
            elif 'ordinal' in dt and 'xn' in dt:
                c = self.ordinal_num_classes[var]
                o = int(row[var])
                x_data.append(
                    F.one_hot(torch.tensor(o, dtype=torch.long), num_classes=c).float().squeeze()
                )
            else:
                img = Image.open(row[var]).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                x_data.append(img)

        if not self.return_intercept_shift:
            if self.return_y:
                y = self._transform_y(row)
                return tuple(x_data), y
            else:
                return tuple(x_data)

        batched = [x.unsqueeze(0) for x in x_data]
        int_in, shifts = self._preprocess_inputs(batched)
        int_in = int_in.squeeze(0)
        shifts = [] if shifts is None else [s.squeeze(0) for s in shifts]
        if self.return_y:
            y = self._transform_y(row)
            return (int_in, shifts), y
        return int_in, shifts


def get_dataloader(
    node,
    target_nodes,
    train_df=None,
    val_df=None,
    batch_size=32,
    return_intercept_shift=False,
    transform=None,
    **kwargs,  # absorbs legacy args like debug=True
):
    """
    Build train/val dataloaders for TRAM models.

    Parameters
    ----------
    node : str
        Target node name.
    target_nodes : dict
        Node configuration dict.
    train_df : pd.DataFrame, optional
        Training data.
    val_df : pd.DataFrame, optional
        Validation data.
    batch_size : int, default=32
        Batch size.
    return_intercept_shift : bool, default=False
        Whether datasets split predictors into intercept/shift groups.
    transform : callable, optional
        Torchvision transforms for image predictors.
    **kwargs
        Ignored. Kept for backward compatibility (e.g., legacy `debug` arg).
    """

    if "debug" in kwargs:
        logger.warning( 
            "' get_dataloader debug' argument is deprecated and ignored. "
            "Use global logger.debug instead."
        )

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    train_loader, val_loader = None, None

    if train_df is not None:
        train_ds = GenericDataset(
            train_df,
            target_col=node,
            target_nodes=target_nodes,
            transform=transform,
            return_intercept_shift=return_intercept_shift,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    else:
        logger.info("train_df is None → skipping train dataloader.")

    if val_df is not None:
        val_ds = GenericDataset(
            val_df,
            target_col=node,
            target_nodes=target_nodes,
            transform=transform,
            return_intercept_shift=return_intercept_shift,
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    else:
        logger.info("val_df is None → skipping val dataloader.")

    if train_loader is None and val_loader is None:
        logger.error("Both train_df and val_df are None → no dataloaders created.")
        raise ValueError("Both train_df and val_df are None → no dataloaders created.")

    return train_loader, val_loader