"""ResNet model."""
from collections import OrderedDict

import torch

from leakpro.utils.import_helper import Self


class ConvNet(torch.nn.Module):
    """ConvNetBN."""

    def __init__(self: Self, width: int=32, num_classes: int=10, num_channels: int=3):
        """Init with width and num classes."""
        super().__init__()
        self.model = torch.nn.Sequential(OrderedDict([
            ("conv0", torch.nn.Conv2d(num_channels, 1 * width, kernel_size=3, padding=1)),
            ("bn0", torch.nn.BatchNorm2d(1 * width)),
            ("relu0", torch.nn.ReLU()),

            ("conv1", torch.nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
            ("bn1", torch.nn.BatchNorm2d(2 * width)),
            ("relu1", torch.nn.ReLU()),

            ("conv2", torch.nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)),
            ("bn2", torch.nn.BatchNorm2d(2 * width)),
            ("relu2", torch.nn.ReLU()),

            ("conv3", torch.nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)),
            ("bn3", torch.nn.BatchNorm2d(4 * width)),
            ("relu3", torch.nn.ReLU()),

            ("conv4", torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ("bn4", torch.nn.BatchNorm2d(4 * width)),
            ("relu4", torch.nn.ReLU()),

            ("conv5", torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ("bn5", torch.nn.BatchNorm2d(4 * width)),
            ("relu5", torch.nn.ReLU()),

            ("pool0", torch.nn.MaxPool2d(3)),

            ("conv6", torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ("bn6", torch.nn.BatchNorm2d(4 * width)),
            ("relu6", torch.nn.ReLU()),

            ("conv7", torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ("bn7", torch.nn.BatchNorm2d(4 * width)),
            ("relu7", torch.nn.ReLU()),

            ("pool1", torch.nn.MaxPool2d(3)),
            ("flatten", torch.nn.Flatten()),
            ("linear", torch.nn.Linear(36 * width, num_classes))
        ]))

    def forward(self: Self, input: torch.Tensor) -> None:
        """Model forward implementation."""
        return self.model(input)
