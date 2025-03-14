import torch
from torch.utils.data import Dataset
import numpy as np
from scipy import ndimage

      
def zoom(volume: torch.Tensor) -> torch.Tensor:
    """
    Zoom the volume by a random factor between 0.9 and 1.1.
    """
    # Move tensor to CPU and convert to numpy
    volume_np = volume.cpu().numpy()
    
    # Sample a random zoom factor
    min_zoom, max_zoom = 0.8, 1.2
    z = np.random.uniform(min_zoom, max_zoom)
    
    # Define the zoom matrix (4x4, assuming the last dimension is channels)
    zoom_matrix = np.array([[1, 0, 0, 0],
                            [0, z, 0, 0],
                            [0, 0, z, 0],
                            [0, 0, 0, z]])
    
    # Apply the affine transform
    augmented_np = ndimage.affine_transform(volume_np, zoom_matrix, mode="nearest", order=1)
    
    # Convert back to torch.Tensor and send to the original device
    return torch.from_numpy(augmented_np).to(volume.device)

# Rotate changes the pixel values when we change the order for ndimage.rotate
def rotate(volume: torch.Tensor) -> torch.Tensor:
    """
    Rotate the volume by a random angle chosen from [-20, -10, -5, 5, 10, 20] degrees.
    Rotation is applied along axes (0,1).
    """
    volume_np = volume.cpu().numpy()
    
    # Choose a random angle
    angles = [-20, -10, -5, 5, 10, 20]
    angle = np.random.choice(angles)
    
    # Rotate the volume; reshape=False keeps the original dimensions.
    augmented_np = ndimage.rotate(volume_np, angle, axes=(2, 3), reshape=False, order=1, mode="nearest")
    
    return torch.from_numpy(augmented_np).to(volume.device)

def shift(volume: torch.Tensor) -> torch.Tensor:
    """
    Shift the volume along x, y, and z axes.
    For x and y the shift is between -20 and 20 pixels, for z between -5 and 5.
    """
    volume_np = volume.cpu().numpy()
    
    # Sample random shifts for x, y, and z.
    x_shift = np.random.uniform(-20, 20)
    y_shift = np.random.uniform(-20, 20)
    z_shift = np.random.uniform(-5, 5)
    
    # Note: the last dimension (assumed to be channels) is not shifted.
    augmented_np = ndimage.shift(volume_np, shift=[0, z_shift, x_shift, y_shift], mode="nearest", order=0)
    
    return torch.from_numpy(augmented_np).to(volume.device)

def flip(volume: torch.Tensor) -> torch.Tensor:
    """
    Randomly flip the volume along the vertical axis.
    With a random choice, if axis==0, the volume is flipped vertically.
    """
    volume_np = volume.cpu().numpy()
    
    # Randomly decide whether to flip
    axis = np.random.choice([0, 1])
    if axis == 0:
        augmented_np = np.flip(volume_np, axis=2).copy() # upside down, axis=3 left/right
    else:
        augmented_np = volume_np
    
    return torch.from_numpy(augmented_np).to(volume.device)

def gauss_filter(volume: torch.Tensor) -> torch.Tensor:
    """
    Randomly apply a Gaussian filter to the volume.
    Sigma is chosen randomly from [0, 1.2].
    """
    volume_np = volume.cpu().numpy()
    
    # Choose sigma: if sigma is 0, the filter does nothing.
    sigma = np.random.choice([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35])
    augmented_np = ndimage.gaussian_filter(volume_np, sigma=sigma)
    
    # Ensure the output is float32, similar to the original tf.float32.
    return torch.from_numpy(augmented_np.astype(np.float32)).to(volume.device)


class AugmentedDataset3D(Dataset):
    """
    The function returns an augmented dataset.
    """
    def __init__(self, x_im_data, y_data, x_tab_data=None, x_im_data2=None, augment=False):
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
            x_im = zoom(x_im)
            x_im = rotate(x_im)
            x_im = gauss_filter(x_im)
            x_im = shift(x_im)

            if self.x_im_data2 is not None:
                x_im2 = zoom(x_im2)
                x_im2 = rotate(x_im2)
                x_im2 = gauss_filter(x_im2)
                x_im2 = shift(x_im2)

        if self.x_tab_data is None and self.x_im_data2 is None:
            return x_im, y
        elif self.x_tab_data is not None and self.x_im_data2 is None:
            return x_im, x_tab, y
        elif self.x_tab_data is None and self.x_im_data2 is not None:
            return x_im, x_im2, y
        elif self.x_tab_data is not None and self.x_im_data2 is not None:
            return x_im, x_tab, x_im2, y
            