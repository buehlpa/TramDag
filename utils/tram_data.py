from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.tram_model_helpers import ordered_parents
import torch.nn.functional as F

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
