import torch
import torch.nn as nn
import torch.nn.functional as F



class OntramModel(nn.Module):
    def __init__(self, nn_int, nn_shift=None):
        super(OntramModel, self).__init__()
        """
        Combine the intercept and shift models into a single model for ONTRAM
        
        Attributes:
            nn_int: PyTorch model for the intercept term
            nn_shift: List of PyTorch models for the shift terms (or None)
        """
        self.nn_int = nn_int
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # if there is no shift model
        if nn_shift is None:
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
        
        if self.nn_shift is None:
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

### SI an LS are always the same
class SimpleIntercept(nn.Module):
    """
    Intercept term , hI()
    Attributes:
        n_thetas (int): how many output thetas, for ordinal target this is the number of classes - 1
    """
    def __init__(self, n_thetas):
        super(SimpleIntercept, self).__init__()  
        self.fc = nn.Linear(1,n_thetas, bias=False)

    def forward(self, x):
        return self.fc(x)
    
class LinearShift(nn.Module):
    """
    Linear shift term,  hS()
    Attributes:
        n_features (int): number of features/predictors
    """
    def __init__(self, n_features):
        super(LinearShift, self).__init__() 
        self.fc = nn.Linear(n_features, 1, bias=False)

    def forward(self, x):
        return self.fc(x)




#################
#Tabular
#################


# Default neural network for complex shift term 
class ComplexShiftDefaultTabular(nn.Module):
    """
    Complex shift term for tabular data. Can be any neural network architecture
    Attributes:
        n_features (int): number of features/predictors
    """
    def __init__(self, n_features):
        super(ComplexShiftDefaultTabular, self).__init__()
        
        # Define the layers
        self.fc1 = nn.Linear(n_features, 8)  # First hidden layer (n_features -> 8)
        self.relu1 = nn.ReLU()               # ReLU activation
        self.fc2 = nn.Linear(8, 8)           # Second hidden layer (8 -> 8)
        self.relu2 = nn.ReLU()               # ReLU activation
        self.fc3 = nn.Linear(8, 1, bias=False)  # Output layer (8 -> 1, no bias)
        
    def forward(self, x):
        # Forward pass through the network
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    

class ComplexInterceptDefaultTabular(nn.Module):
    """
    Complex shift term for tabular data. Can be any neural network architecture
    Attributes:
        n_thetas (int): number of features/predictors
    """
    def __init__(self, n_thetas):
        super(ComplexInterceptDefaultTabular, self).__init__()
        
        # Define the layers
        self.fc1 = nn.Linear(1, 8)  # First hidden layer (X_i -> 8)
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
    


#################
#Iamges
#################


class ComplexInterceptDefaultImage(nn.Module):
    """
    input : eg  torch.randn(1, 3, 28, 28)
    output: n_thetas
    """
    def __init__(self, n_thetas):
        super(ComplexShiftDefaultImage, self).__init__()
        
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
    input : eg  torch.randn(1, 3, 28, 28)
    """
    def __init__(self):
        super(ComplexShiftDefaultImage, self).__init__()
        
        # Adjusted convolutional layers for 28x28 input images
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.pool = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)  # 14x14 -> 14x14
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1, bias=False)  # Output layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool2(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = x.view(-1, 16 * 7 * 7)  # Flatten for fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer
        return x