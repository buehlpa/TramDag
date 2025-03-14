import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class SwinUNETR_Classifier(nn.Module):
    def __init__(self, input_shape, n_output_nodes):
        super(SwinUNETR_Classifier, self).__init__()
        
        self.encoder = SwinUNETR(
            img_size=input_shape[1:],
            in_channels=input_shape[0],
            out_channels=2,
            feature_size=48
        )
        
        # Load pretrained weights from: https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt
        # stored in the data folder
        self.encoder.load_state_dict(torch.load('./weights/model_swinvit.pt', weights_only=True), strict=False)
        
        # Classification Head
        self.avg_pool = nn.AdaptiveAvgPool3d((1,1,1))  # Global Pooling Layer
        
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