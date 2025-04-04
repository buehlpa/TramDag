from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.tram_model_utils import ordered_parents

class GenericDataset(Dataset):
    def __init__(self, df, target_col, data_type=None, transform=None):
        
        
        #TODO if intercept is si but shifts are ci , intercept should return 1s
        
        """
        Args:
            df (pd.DataFrame): The dataframe containing data.
            data_type (dict): Dictionary mapping variable names to their type: "cont", "other", "ord".
            target_col (str): The name of the target column.
            transform (callable, optional): Transformations for images.
        """
        self.df = df
        self.variables =None  if data_type== None else list(data_type.keys())
        self.data_type = data_type
        self.target_col = target_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        row = self.df.iloc[idx]
        
        # if source node
        if self.data_type is None:
            y = torch.tensor(row[self.target_col], dtype=torch.float32)
            x = (torch.tensor(1.0),) # dummy input
            return x , y
        
        # data loader if not source
        x_data = []
        for var in self.variables:
            if self.data_type[var] == "cont":
                x_data.append(torch.tensor(row[var], dtype=torch.float32))
            elif self.data_type[var] == "ord":
                x_data.append(torch.tensor(row[var], dtype=torch.long))
            elif self.data_type[var] == "other":  
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
    
    # TODO amove transforms to the config file  
    transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
    
    
    
    if conf_dict[node]['node_type'] == 'source':
        print('>>>>>>>>>>>>  source node --> x in dataloader contains just 1s ') if verbose else None
        
        train_dataset = GenericDataset(train_df, target_col=node, data_type=None, transform=transform)
        validation_dataset = GenericDataset(train_df, target_col=node, data_type=None, transform=transform)
        
    
    else:
        # create a datatype dictionnary for the dataloader to read the datatype --->> TODO can be passed to a args 
        # parents_dict={x[0]:x[1] for x  in  zip(conf_dict[node]['parents'],conf_dict[node]['parents_datatype'])}
        
        parents_dataype_dict,_,_=ordered_parents(node, conf_dict)
        
        
        train_dataset = GenericDataset(train_df, target_col=node, data_type=parents_dataype_dict, transform=transform)
        validation_dataset = GenericDataset(val_df, target_col=node, data_type=parents_dataype_dict, transform=transform)
     
     
    # TODO add args to the datloader via config file    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    
    return train_loader, val_loader