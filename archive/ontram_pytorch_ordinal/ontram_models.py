import torch
import torch.nn as nn

class OntramModel(nn.Module):
    def __init__(self, nn_int, nn_shift=None):
        super(OntramModel, self).__init__()
        """
        Function to define the Ontram Model
        
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

class InterceptNeuralNetwork(nn.Module):
    """
    Intercept term in an ordinal neural network model.

    Attributes:
        C (int): number of classes
    """
    def __init__(self, C):
        super(InterceptNeuralNetwork, self).__init__()  # Initialize the base class
        # fully connected layer with 1 as input and C-1 output nodes
        self.fc = nn.Linear(1, C-1, bias=False)

    def forward(self, x):
        # Forward pass through the network
        return self.fc(x)
    
class LinearShiftNeuralNetwork(nn.Module):
    """
    Linear shift term for tabular data

    Attributes:
        n_features (int): number of features/predictors
    """
    def __init__(self, n_features):
        super(LinearShiftNeuralNetwork, self).__init__() # Initialize base class
        self.fc = nn.Linear(n_features, 1, bias=False)

    def forward(self, x):
        # Forward pass through the network
        return self.fc(x)

class ComplexShiftNeuralNetwork(nn.Module):
    """
    Complex shift term for tabular data. Can be any neural network architecture

    Attributes:
        n_features (int): number of features/predictors
    """
    def __init__(self, n_features):
        super(ComplexShiftNeuralNetwork, self).__init__()
        
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