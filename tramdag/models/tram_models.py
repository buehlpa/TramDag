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


import torch
import torch.nn as nn
import torch.nn.functional as F


class TramModel(nn.Module):
    def __init__(self, nn_int, nn_shift=None,device='cpu'):
        super(TramModel, self).__init__()
        """
        Combine the intercept and shift models into a single model for ONTRAM
        
        Attributes:
            nn_int: PyTorch model for the intercept term
            nn_shift: List of PyTorch models for the shift terms (or None)
        """
        self.nn_int = nn_int
        self.device = device
        
        # if there is no shift model
        if nn_shift is None or nn_shift==[]:
            self.nn_shift = None
        else:
            # If there are shift models make sure that they are provided as list
            if isinstance(nn_shift, list):
                self.nn_shift = nn.ModuleList(nn_shift)
            else:
                raise ValueError("Input to nn_shift must be a list.")

    def forward(self, int_input, shift_input = None):
        # Forward pass for the intercept
        self.nn_int = self.nn_int.to(self.device)
        int_out = self.nn_int(int_input)
        
        if self.nn_shift is None or shift_input is None or self.nn_shift==[] or shift_input== []:
            return {'int_out': int_out, 'shift_out': None}
        
        if len(self.nn_shift) != len(shift_input):
            raise AttributeError("Number of pytorch models (nn_shift) is not equal to number of provided data (shift_inputs).")
        
        shift_out = []
        for i, shift_model in enumerate(self.nn_shift):
            shift_model = shift_model.to(self.device)
            out = shift_model(shift_input[i])
            shift_out.append(out)
        
        return {'int_out': int_out, 'shift_out': shift_out}


#### Default Neural Network Models  ####

class SimpleIntercept(nn.Module):
    """
    Intercept term , hI()
    Attributes:
        n_thetas (int): how many output thetas, for ordinal target this is the number of classes - 1, thetas are order of bernsteinpol() in continous case
    """
    def __init__(self, n_thetas=20):
        super(SimpleIntercept, self).__init__()  
        self.fc = nn.Linear(1,n_thetas, bias=False)

    def forward(self, x):
        return self.fc(x)
    


class LinearShift(nn.Module):
    """
    Linear shift term, hS()

    Args:
        n_features (int): Number of input features
        init_weight (float or list or 1D torch.Tensor, optional): 
            Initial weight(s) for the linear layer. 
            - If n_features=1, must be a float or tensor/list of shape (1,)
            - If n_features>1, must be a list or 1D tensor of shape (n_features,)

    Raises:
        ValueError: if init_weight shape/type does not match n_features
    """
    def __init__(self, n_features=1, init_weight=None):
        super(LinearShift, self).__init__()
        self.fc = nn.Linear(n_features, 1, bias=False)


        # if weight is initalized set weights to init weight
        if init_weight is not None:
            if isinstance(init_weight, (float, int)):  # scalar
                if n_features != 1:
                    raise ValueError("Scalar init_weight only allowed if n_features=1.")
                weight_tensor = torch.tensor([[float(init_weight)]], dtype=torch.float32)

            elif isinstance(init_weight, (list, torch.Tensor)):
                weight_tensor = torch.tensor(init_weight, dtype=torch.float32).view(1, -1)
                if weight_tensor.shape != (1, n_features):
                    raise ValueError(f"init_weight must have shape (1, {n_features}) but got {weight_tensor.shape}.")

            else:
                raise TypeError("init_weight must be a float, list, or 1D torch.Tensor.")

            with torch.no_grad():
                self.fc.weight.copy_(weight_tensor)

    def forward(self, x):
        return self.fc(x)



######################################################################################################
#Tabular
######################################################################################################


############################################### Default neural network for complex shift term and intercept


class ComplexShiftDefaultTabular(nn.Module):
    """
    A neural network module to compute complex shift terms for tabular data.
    
    Architecture:
        Input layer -> Linear(64) -> ReLU -> Dropout
                    -> Linear(128) -> ReLU -> Dropout
                    -> Linear(64) -> ReLU -> Dropout
                    -> Linear(1, no bias)
    
    Attributes:
        n_features (int): Number of input features (predictors)
    """
    def __init__(self, n_features=1):
        super(ComplexShiftDefaultTabular, self).__init__()
        self.fc1 = nn.Linear(n_features, 64)     # Input layer: n_features -> 64
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 128)            # Hidden layer: 64 -> 128
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)            # Hidden layer: 128 -> 64
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 1, bias=False)  # Output layer: 64 -> 1, no bias

    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x (Tensor): Input tensor of shape (batch_size, n_features)

        Returns:
            Tensor: Output tensor of shape (batch_size, 1)
        """
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

    

class ComplexInterceptDefaultTabular(nn.Module):
    """
    Complex shift term for tabular data. Can be any neural network architecture
    Attributes:
        n_thetas (int): number of features/predictors
    """
    def __init__(self, n_features=1,n_thetas=20):
        super(ComplexInterceptDefaultTabular, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(n_features, 8)  # First hidden layer (X_i -> 8)
        self.relu1 = nn.ReLU()               # ReLU activation
        self.fc2 = nn.Linear(8, 8)           # Second hidden layer (8 -> 8)
        self.relu2 = nn.ReLU()               # ReLU activation
        self.fc3 = nn.Linear(8, n_thetas, bias=False)  # Output layer (8 -> n_thetas, no bias)
        
    def forward(self, x):
        # Forward pass through the network
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    
############################################### Deeper networks
class ComplexShiftCustomTabular(nn.Module):
    """
    Deeper shift network for tabular data, without any manual flattening.
    Linear layers will be applied to the last dimension, whatever its shape.
    """
    def __init__(self, n_features: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1, bias=False)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ComplexInterceptCustomTabular(nn.Module):
    """
    Deeper intercept network for tabular data, without any manual flattening.
    """
    def __init__(self, n_thetas: int = 20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_thetas, bias=False)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



######################################################################################################
#Iamges
######################################################################################################

class ComplexInterceptDefaultImage(nn.Module):
    """
    input : eg  torch.randn(1, 3, 28, 28)
    output: n_thetas
    """
    def __init__(self, n_thetas=20):
        super(ComplexInterceptDefaultImage, self).__init__()
        
        # Adjusted convolutional layers for 28x28 input images
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.pool = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)  # 14x14 -> 14x14
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_thetas, bias=False)  # Output layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool2(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = x.view(-1, 16 * 7 * 7)  # Flatten for fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer
        return x


class ComplexShiftDefaultImage(nn.Module):
    """
    Accepts input of shape (batch_size, 3, 128, 128).
    """
    def __init__(self):
        super(ComplexShiftDefaultImage, self).__init__()

        # Adjusted convolutional layers for 128x128 input images
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, padding=1)  # 128x128 -> 128x128
        self.pool = nn.MaxPool2d(2, 2)  # 128x128 -> 64x64
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)  # 64x64 -> 64x64
        self.pool2 = nn.MaxPool2d(2, 2)  # 64x64 -> 32x32

        # Fully connected layers (adjusted for 128x128 input size)
        self.fc1 = nn.Linear(16 * 32 * 32, 120)  # Previously 16 * 7 * 7
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1, bias=False)  # Output layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool2(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = x.view(-1, 16 * 32 * 32)  # Flatten for fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer
        return x