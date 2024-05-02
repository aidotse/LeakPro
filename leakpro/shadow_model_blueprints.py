"""Models for the datasets."""
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torchvision import models

from leakpro.import_helper import Self


class NN(nn.Module):
    """NN for Adult dataset."""

    def __init__(self:Self, in_shape:int, num_classes:int=10) -> None:
        """Initialize the model.

        Args:
        ----
            in_shape (int): The input shape.
            num_classes (int, optional): The number of classes. Defaults to 10.

        """
        super().__init__()

        # Store the initialization parameters to be saved in the metadata
        self.init_params = {
            "in_shape": in_shape,
            "num_classes": num_classes
        }

        self.fc1 = nn.Linear(in_shape, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, num_classes)

    def forward(self:Self, inputs:torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        inputs = inputs.flatten(1)
        outputs = F.relu(self.fc1(inputs))
        outputs = F.relu(self.fc2(outputs))
        return F.relu(self.fc3(outputs))

class ConvNet(nn.Module):
    """Convolutional Neural Network model."""

    def __init__(self:Self) -> None:
        """Initialize the ConvNet model."""
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self:Self, x:torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
        ----
            x (torch.Tensor): The input tensor.

        Returns:
        -------
            torch.Tensor: The output tensor.

        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SmallerSingleLayerConvNet(nn.Module):
    """Smaller Convolutional Neural Network model with only one convolutional layer."""

    def __init__(self:Self) -> None:
        """Initialize the SmallerSingleLayerConvNet model."""
        super().__init__()
        # Only one convolutional layer
        self.conv1 = nn.Conv2d(3, 4, 5)
        self.pool = nn.MaxPool2d(2, 2)

        # Adjusting the linear layers since we now have only one conv layer
        # Assuming the input image size is 32x32, after one convolution and pooling, the size is reduced to 14x14 (32 -> 28 -> 14)
        self.fc1 = nn.Linear(4 * 14 * 14, 120)  # Adjusted to match the output of the pooling layer
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self:Self, x:torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ResNet18(nn.Module):
    """ResNet-18 model from torchvision."""

    def __init__(self:Self, num_classes:int = 10) -> None:  # noqa: D417
        """Initialize the ResNet-18 model.
        Args:
        ----
            num_classes (int, optional): The number of classes. Defaults to 1000.
        """
        super().__init__()
        self.init_params = {
            "num_classes": num_classes
        }
        self.model = models.resnet18(pretrained=False, num_classes=num_classes)

    def forward(self:Self, x:torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        Args:
        ----
            x (torch.Tensor): The input tensor.
        Returns:
        -------
            torch.Tensor: The output tensor.
        """
        return self.model(x)