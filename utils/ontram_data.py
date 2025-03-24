import torch
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class GenericDataset(Dataset):
    def __init__(self, df, target_col, data_type=None, transform=None):
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
            x = torch.tensor(1.0) # dummy input
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