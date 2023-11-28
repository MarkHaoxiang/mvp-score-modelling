from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class ClassificationNet(nn.Module):
    """Network for classification based on mobilenet

    Unconditioned on noise. No fourier features.
    """
    def generate_rff(self, x, dimension):
        random_matrix = torch.randn((1, dimension), device=x.device)
        transformed = torch.matmul(x.unsqueeze(-1), random_matrix)
        return torch.cos(transformed)

    def __init__(self, n_classes: int=2) -> None:
        """ Create a network for classification based on mobilenet

        Args:
            n_classes (int, optional): Number of target classes. Defaults to 2.
        """
        super().__init__()

        mobile_net = mobilenet_v3_small(MobileNet_V3_Small_Weights.DEFAULT)
        self.cnn = nn.Sequential(
            mobile_net.features,
            mobile_net.avgpool,
            nn.Flatten(),
            nn.Dropout(0.2, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=576, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [b, c, h, w]
        """
        x = self.cnn(x)
        x = self.fc(x)
        return x

class ClassificationNetCNN(nn.Module):
    def generate_rff(self, x, dimension):
        random_matrix = torch.randn((1, dimension), device=x.device)
        transformed = torch.matmul(x.unsqueeze(-1), random_matrix)
        return torch.cos(transformed)

    def __init__(self, n_classes: int=2, rff_dim: Optional[int] = None) -> None:
        """ Create a network for classification based on mobilenet

        Args:
            n_classes (int, optional): Number of target classes. Defaults to 2.
            rff_dim (Optional[int], optional): Random fourier features. Defaults to None.
        """
        super().__init__()

        if rff_dim is None:
            self.rff_dim = 0
        if rff_dim > 0:
            self.rff_dim = rff_dim

        kernel_size = 5
        padding = 2
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(0.2, inplace=True),  
        )
        linear = 512
        self.fc = nn.Sequential(
            nn.Linear(64 * 32 * 32 + self.rff_dim, linear),
            nn.LeakyReLU(),
            nn.Linear(linear, n_classes)  # Output for 2 classes
        )

    def forward(self, x, sigmas):
        '''
        x: [b, c, h, w]
        sigmas: [b]
        '''
        x = self.cnn(x)

        log_sigma = torch.log(sigmas + 1e-8)  # To ensure numerical stability
        rff = self.generate_rff(log_sigma, self.rff_dim)
        x = torch.cat([x, rff], dim=1)  # Concatenate RFF at the last linear layer

        x = self.fc(x)
        return x