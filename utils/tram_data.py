from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.tram_model_helpers import ordered_parents
import torch.nn.functional as F
import pandas as pd
class GenericDataset(Dataset):
    def __init__(self, df, target_col, conf_dict=None, transform=None, transformation_terms_in_h=None):
        #TODO if intercept is si but shifts are ci , intercept should return 1s
        """
        Args:
            df (pd.DataFrame): The dataframe containing data.
            conf_dict (dict): Dictionary mapping variable names to their type: "cont", "other", "ord".
            target_col (str): The name of the target column.
            transform (callable, optional): Transformations for images.
        """
        self.df = df
        self.variables =None  if conf_dict== None else list(conf_dict.keys())
        self.conf_dict = conf_dict
        self.target_col = target_col
        self.transform = transform
        self.transformation_terms_in_h=transformation_terms_in_h
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x_data = []
        
        # if source node only 1 for SI and the target (node itself) is returned
        if self.conf_dict is None:
            y = torch.tensor(row[self.target_col], dtype=torch.float32)
            x = torch.tensor(1.0) # For SI on Sources CI also possible but not meaningful
            x_data.append(x)
            x = tuple(x_data)
            return x , y
        
        # if there are no Intercepts we need to add a 1 because modell assumes SI for intercepts
        if all('i' not in str(value) for value in self.transformation_terms_in_h.values()):
            x = torch.tensor(1.0) 
            x_data.append(x)
        
        # data loader if not source , differnt format for the datatypes
        for var in self.variables:
            if self.conf_dict[var] == "cont":
                x_data.append(torch.tensor(row[var], dtype=torch.float32))
            elif self.conf_dict[var] == "ord":
                x_data.append(torch.tensor(row[var], dtype=torch.long))
            elif self.conf_dict[var] == "other":  
                img_path = row[var]
                image = Image.open(img_path).convert("RGB")

                if self.transform:
                    image = self.transform(image)
                    
                x_data.append(image)  # Append instead of replacing by index
        x = tuple(x_data)
        
        y = torch.tensor(row[self.target_col], dtype=torch.float32)

        return x, y

def get_dataloader(node, conf_dict, train_df, val_df, batch_size=32,verbose=False):    
    

    # TODO move args to config file batchsize  etc.
    
    transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
    
    if conf_dict[node]['node_type'] == 'source':
        print('>>>>>>>>>>>>  source node --> x in dataloader contains just 1s ') if verbose else None
        
        train_dataset = GenericDataset(train_df, target_col=node, conf_dict=None, transform=transform)
        validation_dataset = GenericDataset(val_df, target_col=node, conf_dict=None, transform=transform)
    
    else:
        # create a datatype dictionnary for the dataloader to read the datatype --->> TODO can be passed to a args 
        # parents_dict={x[0]:x[1] for x  in  zip(conf_dict[node]['parents'],conf_dict[node]['parents_datatype'])}
        
        parents_dataype_dict,transformation_terms_in_h,_=ordered_parents(node, conf_dict)
        
        
        train_dataset = GenericDataset(train_df, target_col=node, conf_dict=parents_dataype_dict, transform=transform,transformation_terms_in_h=transformation_terms_in_h)
        validation_dataset = GenericDataset(val_df, target_col=node, conf_dict=parents_dataype_dict, transform=transform,transformation_terms_in_h=transformation_terms_in_h)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4,pin_memory=True)
    
    
    return train_loader, val_loader


class GenericDataset_v2(Dataset):
    def __init__(
        self,
        df,
        target_col,
        target_nodes=None,
        parents_dataype_dict=None,
        transform=None,
        transformation_terms_in_h=None
    ):
        """
        df: pd.DataFrame
        target_col: str
        target_nodes: dict mapping each node → metadata (including 'data_type')
        parents_dataype_dict: dict var_name → "cont"|"ord"|"other"
        transform: torchvision transform for images
        transformation_terms_in_h: dict for intercept logic
        """
        self.df = df.reset_index(drop=True)
        self.target_col = target_col
        self.target_nodes = target_nodes or {}
        self.parents_dataype_dict = parents_dataype_dict or {}
        self.variables = list(self.parents_dataype_dict.keys())
        self.transform = transform
        self.transformation_terms_in_h = transformation_terms_in_h or {}

        # If we know this target is ordinal, record #classes
        if (
            self.target_nodes
            and self.target_col in self.target_nodes
            and self.target_nodes[self.target_col].get('data_type') == "ord"
        ):
            self.num_classes = int(self.df[self.target_col].nunique())
        else:
            self.num_classes = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x_data = []

        # --- SOURCE NODE: no parents → x = [1.0] ---
        if not self.parents_dataype_dict:
            x_data = [torch.tensor(1.0)]
            # handle y
            if self.num_classes is not None:
                # ordinal source → one-hot
                raw = row[self.target_col]
                y_int = int(raw)
                y = F.one_hot(torch.tensor(y_int, dtype=torch.long), num_classes=self.num_classes).float()
                
            else:
                # continuous or other
                y = torch.tensor(row[self.target_col], dtype=torch.float32)
            return tuple(x_data), y

        # --- INTERCEPT if needed ---
        if all('i' not in str(v) for v in self.transformation_terms_in_h.values()):
            x_data.append(torch.tensor(1.0))

        # --- BUILD FEATURES ---
        for var in self.variables:
            dtype = self.parents_dataype_dict[var]
            if dtype == "cont":
                x_data.append(torch.tensor(row[var], dtype=torch.float32))
            elif dtype == "ord":
                x_data.append(torch.tensor(row[var], dtype=torch.long))
            else:  # "other"
                img = Image.open(row[var]).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                x_data.append(img)

        # --- BUILD TARGET ---
        if self.num_classes is not None:
            # ordinal → one-hot
            raw = row[self.target_col]
            y_int = int(raw)
            y = F.one_hot(torch.tensor(y_int, dtype=torch.long),num_classes=self.num_classes).float()
            
        else:
            # continuous or other
            y = torch.tensor(row[self.target_col], dtype=torch.float32)
        return tuple(x_data), y

def get_dataloader_v2(node, target_nodes, train_df, val_df, batch_size=32, verbose=False):
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    if target_nodes[node]['node_type'] == 'source':
        if verbose:
            print('Source node → features are just a constant 1.')
        train_ds = GenericDataset_v2(train_df,target_col=node,target_nodes=target_nodes,parents_dataype_dict=None,transform=transform)
        val_ds = GenericDataset_v2(val_df,target_col=node,target_nodes=target_nodes,parents_dataype_dict=None,transform=transform)
        
    else:
        parents_dataype_dict, transformation_terms_in_h, _ = ordered_parents(node, target_nodes)
        if verbose:
            print(f"Parents dtype: {parents_dataype_dict}")
        train_ds = GenericDataset_v2(train_df,target_col=node,target_nodes=target_nodes,parents_dataype_dict=parents_dataype_dict,transform=transform,transformation_terms_in_h=transformation_terms_in_h)
        val_ds = GenericDataset_v2(val_df,target_col=node,target_nodes=target_nodes,parents_dataype_dict=parents_dataype_dict,transform=transform,transformation_terms_in_h=transformation_terms_in_h)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader


class GenericDataset_v5(Dataset):
    def __init__(
        self,
        df,
        target_col,
        target_nodes=None,
        parents_dataype_dict=None,
        transformation_terms_in_h=None,
        return_intercept_shift=True,
        transform=None,
        return_y=True
    ):
        """
        df: pd.DataFrame
        target_col: str
        target_nodes: dict mapping each node → metadata (including 'data_type')
        parents_dataype_dict: dict var_name → "cont"|"ord"|"other"
        transformation_terms_in_h: dict for intercept logic
        return_intercept_shift: whether to return (int_input, shift_list) or raw features
        transform: torchvision transform for images
        """
        self.return_intercept_shift = return_intercept_shift
        self.return_y=return_y
        self.df = df.reset_index(drop=True)
        self.target_col = target_col
        self.target_nodes = target_nodes or {}
        self.parents_dataype_dict = parents_dataype_dict or {}
        self.predictors = list(self.parents_dataype_dict.keys())
        self.transform = transform
        self.transformation_terms_preprocessing = list((transformation_terms_in_h or {}).values())

        self.target_is_source = (self.target_nodes[self.target_col].get('node_type', '').lower() == "source")
        
        # do we need an explicit simple intercept?
        self.h_needs_simple_intercept = all('i' not in str(v) for v in self.transformation_terms_preprocessing)
        # count ordinal classes
        self.ordinal_num_classes = {
            v: self.df[v].nunique()
            for v in self.predictors
            if "ordinal" in self.parents_dataype_dict[v].lower()
            and "xn" in self.parents_dataype_dict[v].lower()
        }

        self.target_data_type = self.target_nodes[self.target_col].get('data_type', '').lower()
        self.target_num_classes = self.target_nodes[self.target_col].get('levels')

        # figure out the intercept and shift terms for preprocessed 
        if return_intercept_shift:
            self._set_intercept_shift_indexes()

        # checks
        self._check_multiclass_predictors_of_df()
        self._check_ordinal_levels()

    def _set_intercept_shift_indexes(self):
        # always inject a simple intercept term if no 'ci' present
        if not any('ci' in t for t in self.transformation_terms_preprocessing):
            self.transformation_terms_preprocessing.insert(0, 'si')

        self.intercept_indices = [
            i for i, term in enumerate(self.transformation_terms_preprocessing)
            if term.startswith(('si', 'ci'))
        ]

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

    def _preprocess_inputs(self, x):
        """
        x: List of tensors, each with a dummy batch‑dim at 0
        Returns:
            int_inputs: (B, sum(intercept_dims))
            shift_list: list of (B, sum(dims_per_group)) or None
        """
        # intercept
        its = [x[i] for i in self.intercept_indices]
        if not its:
            raise ValueError("No intercept tensors found!")
        int_inputs = torch.cat([t.view(t.shape[0], -1) for t in its], dim=1)

        # shift‐groups
        shifts = []
        for grp in self.shift_groups_indices:
            parts = [x[i].view(x[i].shape[0], -1) for i in grp]
            shifts.append(torch.cat(parts, dim=1))

        return int_inputs, (shifts if shifts else None)

    def _transform_y(self, row):
        if self.target_data_type == "continous" or "yc" in self.target_data_type:
            return torch.tensor(row[self.target_col], dtype=torch.float32)
        elif self.target_num_classes:
            yi = int(row[self.target_col])
            return F.one_hot(
                torch.tensor(yi, dtype=torch.long),
                num_classes=self.target_num_classes
            ).float().squeeze()
        else:
            raise ValueError(
                f"Cannot encode target '{self.target_col}': "
                f"{self.target_data_type}/{self.target_num_classes}"
            )

    def _check_multiclass_predictors_of_df(self):
        for v in self.predictors:
            dt = self.parents_dataype_dict[v].lower()
            if "ordinal" in dt and "xn" in dt:
                vals = set(self.df[v].dropna().unique())
                if vals != set(range(len(vals))):
                    raise ValueError(
                        f"Ordinal predictor '{v}' must be zero‑indexed; got {sorted(vals)}"
                    )

    def _check_ordinal_levels(self):
        ords = []
        if "ordinal" in self.target_nodes[self.target_col]['data_type']:
            ords.append(self.target_col)
        ords += [
            v for v in self.predictors
            if "ordinal" in self.parents_dataype_dict[v].lower()
            and "xn" in self.parents_dataype_dict[v].lower()
        ]
        for v in ords:
            lvl = self.target_nodes[v].get('levels')
            if lvl is None:
                raise ValueError(f"Ordinal '{v}' missing 'levels' metadata.")
            uniq = sorted(self.df[v].dropna().unique())
            if uniq != list(range(lvl)):
                raise ValueError(
                    f"Ordinal '{v}' values {uniq} != expected 0…{lvl-1}."
                )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x_data = []

        # simple intercept if needed
        if self.h_needs_simple_intercept:
            x_data.append(torch.tensor(1.0))

        # source‐only shortcut
        if self.target_is_source:
            y = self._transform_y(row)
            if not self.return_intercept_shift:
                return tuple(x_data), y

            # batchify & preprocess
            batched = [x.unsqueeze(0) for x in x_data]
            int_in, shifts = self._preprocess_inputs(batched)
            # **squeeze off** our dummy batch‐dim:
            int_in = int_in.squeeze(0)              # -> (n_intercept,)
            shifts = [] if shifts is None else [s.squeeze(0) for s in shifts]
            return (int_in, shifts), y

        # build predictors
        for var in self.predictors:
            dt = self.parents_dataype_dict[var].lower()
            if dt == "continous" or "xc" in dt:
                x_data.append(torch.tensor(row[var], dtype=torch.float32))
            elif "ordinal" in dt and "xn" in dt:
                c = self.ordinal_num_classes[var]
                o = int(row[var])
                x_data.append(
                    F.one_hot(torch.tensor(o, dtype=torch.long), num_classes=c).float().squeeze()
                )
            else:
                img = Image.open(row[var]).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                x_data.append(img)

        # only return the original data without separating them for int and shifts
        if not self.return_intercept_shift:
            if self.return_y:
                y = self._transform_y(row)
                return tuple(x_data), y
            else:
                return tuple(x_data)
        
        # returning already splitted int and shifts
        else:    
            batched = [x.unsqueeze(0) for x in x_data]
            int_in, shifts = self._preprocess_inputs(batched)
            int_in = int_in.squeeze(0)
            shifts = [] if shifts is None else [s.squeeze(0) for s in shifts]
            
            if self.return_y:
                y = self._transform_y(row)
                return (int_in, shifts), y
            else:
                return (int_in, shifts)


def get_dataloader_v5(node, target_nodes, train_df, val_df, batch_size=32,return_intercept_shift=False, verbose=False):
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    ordered_parents_dataype_dict, ordered_transformation_terms_in_h, _ = ordered_parents(node, target_nodes)
    if verbose:
        print(f"Parents dtype: {ordered_parents_dataype_dict}")
    train_ds = GenericDataset_v5(train_df,target_col=node,target_nodes=target_nodes,parents_dataype_dict=ordered_parents_dataype_dict,transform=transform,transformation_terms_in_h=ordered_transformation_terms_in_h,return_intercept_shift=return_intercept_shift)
    val_ds = GenericDataset_v5(val_df,target_col=node,target_nodes=target_nodes,parents_dataype_dict=ordered_parents_dataype_dict,transform=transform,transformation_terms_in_h=ordered_transformation_terms_in_h,return_intercept_shift=return_intercept_shift)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader



class GenericDataset_v6(Dataset):
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
        df: pd.DataFrame
        target_col: str
        target_nodes: dict mapping each node → metadata (including 'data_type', 'levels', 'node_type')
        parents_datatype_dict: dict var_name → "cont"|"ord"|"other"
        transformation_terms_in_h: dict for intercept logic
        return_intercept_shift: whether to return (int_input, shift_list) or raw features
        transform: torchvision transform for images
        return_y: whether to return target y
        debug: bool to enable debug logging and attribute printouts
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

    # Setter methods
    def _set_ordered_parents_datatype_and_transformation_terms(self):
        
        ordered_parents_datatype, ordered_transformation_terms_in_h, _ =ordered_parents(self.target_col, self.target_nodes)
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
            print(f"[DEBUG] Set debug: type={type(self.debug)}, value={self.debug}")

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
                f"[WARNING] target_col '{target_col}' not in DataFrame columns — is this intended for use as a Sampler?")
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

    def ordered_parents(self,node, target_dict) -> dict:
        
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

    def _check_ordinal_levels(self):
        ords = []
        if 'ordinal' in self.target_nodes.get(self.target_col, {}).get('data_type', '').lower():
            ords.append(self.target_col)
        ords += [
            v for v in self.predictors
            if 'ordinal' in self.parents_datatype_dict[v].lower()
            and 'xn' in self.parents_datatype_dict[v].lower()
        ]
        for v in ords:
            lvl = self.target_nodes[v].get('levels')
            if lvl is None:
                raise ValueError(f"Ordinal '{v}' missing 'levels' metadata.")
            uniq = sorted(self.df[v].dropna().unique())
            if uniq != list(range(lvl)):
                raise ValueError(
                    f"Ordinal '{v}' values {uniq} != expected 0…{lvl-1}."
                )



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



def get_dataloader_v6(node, target_nodes, train_df, val_df, batch_size=32,return_intercept_shift=False, verbose=False):
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    train_ds = GenericDataset_v6(train_df,target_col=node,target_nodes=target_nodes,transform=transform,return_intercept_shift=return_intercept_shift)
    val_ds = GenericDataset_v6(val_df,target_col=node,target_nodes=target_nodes,transform=transform,return_intercept_shift=return_intercept_shift)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader