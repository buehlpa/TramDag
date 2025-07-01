from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.tram_model_helpers import ordered_parents

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