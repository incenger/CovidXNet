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
        self.model = EfficientNet.from_pretrained('efficientnet-b2',
                                                  in_channels=1)
        self.fc = nn.Linear(in_features=1000, out_features=self.num_class)

    def forward(self, x):
        out = self.model(x)
        out = self.fc(out)
        return out


class ConvNetXray(nn.Module):
    """
    Simple ConvNet

    Input image size should be (224, 224)

    Reference: https://cs231n.github.io/convolutional-networks/
    """
    def __init__(self, num_class=3):
        super().__init__()

        self.num_class = num_class
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3,
                      padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d((2, 2)))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d((2, 2)))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.MaxPool2d((2, 2)))

        # 7*7*512
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=7 * 7 * 256, out_features=1024), nn.ReLU())

        self.fc2 = nn.Linear(in_features=1024, out_features=self.num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out = out.reshape(out.shape[0], -1)

        out = self.fc1(out)
        out = self.fc2(out)

        return out


class DenseNetCovidX(nn.Module):
    """
    Covid X-ray classification using Resnet-34 backbone

    Args:
        num_class: number of output classes
    """
    def __init__(self, num_class=3):
        super().__init__()
        self.num_class = num_class
        self.model = models.densenet161(pretrained=True)

        num_ftrs = self.model.classifier.in_features

        self.model.classifier = nn.Linear(num_ftrs, num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input image, shape (B, C, H, W)

        Returns:
            torch.Tensor: output logits tensor, shape (B, num_class)
        """
        out: torch.Tensor = self.model(x)
        return out
