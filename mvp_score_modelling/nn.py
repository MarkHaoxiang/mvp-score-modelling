from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# TODO: Add RFF

class ClassificationNet(nn.Module):
    """Network for classification based on mobilenet
    """

    def __init__(self, n_classes: int=2, rff_dim: Optional[int] = None) -> None:
        """ Create a network for classification based on mobilenet

        Args:
            n_classes (int, optional): Number of target classes. Defaults to 2.
            rff_dim (Optional[int], optional): Random fourier features. Defaults to None.
        """
        super().__init__()

        if rff_dim is None:
            rff_dim = 0
        if rff_dim > 0:
            raise NotImplementedError("RFF not yet transferred from notebook")
            
        mobile_net = mobilenet_v3_small(MobileNet_V3_Small_Weights.DEFAULT)
        self.cnn = nn.Sequential(
            mobile_net.features,
            mobile_net.avgpool,
            nn.Flatten(),
            nn.Dropout(0.2, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=576 + rff_dim, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = self.fc(x)
        return x
