from torch.utils.data import Dataset

class AugmentedDataset3D(Dataset):
    """
    The function returns an augmented dataset.
    """
    def __init__(self, x_im_data, y_data, x_tab_data=None, x_im_data2=None, augment=None):
        self.x_im_data = x_im_data
        self.y_data = y_data
        self.x_tab_data = x_tab_data
        self.x_im_data2 = x_im_data2
        self.augment = augment

    # has to be implemented for any custom Dataset function
    def __len__(self): 
        return len(self.x_im_data)

    def __getitem__(self, idx):
        x_im = self.x_im_data[idx]
        y = self.y_data[idx]
        if self.x_tab_data is not None:
            x_tab = self.x_tab_data[idx]
        if self.x_im_data2 is not None:
            x_im2 = self.x_im_data2[idx]

        if self.augment:
            x_im = self.augment(x_im)
            if self.x_im_data2 is not None:
                x_im2 = self.augment(x_im2)

        if self.x_tab_data is None and self.x_im_data2 is None:
            return x_im, y
        elif self.x_tab_data is not None and self.x_im_data2 is None:
            return x_im, x_tab, y
        elif self.x_tab_data is None and self.x_im_data2 is not None:
            return x_im, x_im2, y
        elif self.x_tab_data is not None and self.x_im_data2 is not None:
            return x_im, x_tab, x_im2, y