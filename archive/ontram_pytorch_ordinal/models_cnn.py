import torch
import torch.nn as nn
from lighter_zoo import SegResNet
   
class Custom3DCNN(nn.Module):
    def __init__(self, input_shape, n_output_nodes):
        super(Custom3DCNN, self).__init__()
        
        # Convolutional layers
        self.convblock1 = nn.Sequential(
            nn.Conv3d(input_shape[0], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)))
        self.convblock2 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)))
        self.convblock3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)))
        self.convblock4 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)))
        self.convblock5 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)))
        self.convblock6 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)))
        
        # Get flatten size
        self.flatten_size = self._get_flatten_size(input_shape)

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(self.flatten_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4))
        self.fc2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4))
        self.fcout = nn.Sequential(
            nn.Linear(64, n_output_nodes, bias=False))
        
    def _get_flatten_size(self, input_shape):
        dummy_input = torch.zeros(1, *input_shape)
        x = self.convblock1(dummy_input)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fcout(x)
        return x
    

class CTFoundation(nn.Module):
    def __init__(self, input_shape, n_output_nodes):
        super(CTFoundation, self).__init__()
        
        self.encoder = SegResNet.from_pretrained("project-lighter/ct_fm_segresnet")
        
        # Classification Head
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten_size = self._get_flatten_size(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 64),
            nn.ReLU())
        self.fcout = nn.Sequential(
            nn.Linear(64, n_output_nodes, bias=False))
    
    def _get_flatten_size(self, input_shape):
        dummy_input = torch.zeros(1, *input_shape)
        x = self.encoder(dummy_input)
        x = self.avg_pool(x)
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.fcout(x)
        return x