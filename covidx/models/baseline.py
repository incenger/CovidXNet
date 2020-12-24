import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from efficientnet_pytorch import EfficientNet

class ResnetCovidX(nn.Module):
    """
    Covid X-ray classification using Resnet-34 backbone

    Args:
        num_class: number of output classes
    """
    def __init__(self, num_class=3):
        super().__init__()
        self.num_class = num_class
        self.model = models.resnet34(pretrained=True)
        # Replace first conv layer for training grayscale images
        self.model.conv1 = nn.Conv2d(in_channels=1,
                                     out_channels=64,
                                     kernel_size=7,
                                     stride=2,
                                     padding=3,
                                     bias=False)
        # Replace fully connected layers with an embedding layer
        self.model.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=num_class))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input image, shape (B, C, H, W)

        Returns:
            torch.Tensor: output logits tensor, shape (B, num_class)
        """
        out: torch.Tensor = self.model(x)
        return out


class EfficientNetCovidXray(nn.Module):
    """
    Covid X-ray classification using EfficientNet-b3 backbone

    Args:
        num_class: number of output classes
    """

    def __init__(self, num_class=3):
        super().__init__()
        self.num_class = num_class
        self.model = EfficientNet.from_pretrained('efficientnet-b3', in_channels=1)
        self.fc = nn.Linear(in_features=1000, out_features=self.num_class)

    def forward(self, x):
        out = self.model(x)
        out = self.fc(out)
        return out
