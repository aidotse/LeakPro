"""Models for the datasets."""

from typing import Callable, Optional

import torch
import torchvision
from torch import Tensor, flatten, nn
from torch.nn import Module, functional
from torchvision.models import resnet18
from torchvision.models.resnet import Bottleneck

from leakpro.import_helper import Self


class NN(Module):
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

    def forward(self:Self, inputs:Tensor) -> Tensor:
        """Forward pass of the model."""
        inputs = inputs.flatten(1)
        outputs = functional.relu(self.fc1(inputs))
        outputs = functional.relu(self.fc2(outputs))
        return functional.relu(self.fc3(outputs))

class ConvNet(Module):
    """Convolutional Neural Network model."""

    def __init__(self:Self, num_classes: int = 10) -> None:
        """Initialize the ConvNet model."""
        super().__init__()
        self.init_params = {
            "num_classes": num_classes
        }

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self:Self, x:Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
        ----
            x (torch.Tensor): The input tensor.

        Returns:
        -------
            torch.Tensor: The output tensor.

        """
        x = self.pool(functional.relu(self.conv1(x)))
        x = self.pool(functional.relu(self.conv2(x)))
        x = flatten(x, 1) # flatten all dimensions except batch
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
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

    def forward(self:Self, x:Tensor) -> Tensor:
        """Forward pass of the model."""
        x = self.pool(functional.relu(self.conv1(x)))
        x = flatten(x, 1)  # flatten all dimensions except the batch
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
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
        self.model = resnet18(pretrained=False, num_classes=num_classes)

    def forward(self:Self, x:Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
        ----
            x (torch.Tensor): The input tensor.

        Returns:
        -------
            torch.Tensor: The output tensor.

        """
        return self.model(x)

class ResNet(torchvision.models.ResNet):
    """ResNet generalization for CIFAR thingies."""

    def __init__(self: Self, block: Module = torchvision.models.resnet.BasicBlock, layers: list=[5,5,5], num_classes: int=10,  # noqa: B006, C901
                 zero_init_residual: bool=False, groups: int = 1, base_width : int =160,
                 replace_stride_with_dilation: Optional[list[bool]] =None, norm_layer: Optional[Callable[..., nn.Module]] =None,
                 strides: list = [1, 2, 2, 2], pool: str = "avg") -> None:  # noqa: B006
        """Initialize as usual. Layers and strides are scriptable."""
        super(torchvision.models.ResNet, self).__init__()  # nn.Module
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer


        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False, False]
        if len(replace_stride_with_dilation) != 4:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 4-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups

        self.inplanes = base_width
        self.base_width = 64  # Do this to circumvent BasicBlock errors. The value is not actually used.
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layers = torch.nn.ModuleList()
        width = self.inplanes
        for idx, layer in enumerate(layers):
            self.layers.append(self._make_layer(block, width, layer, stride=strides[idx],
                                                dilate=replace_stride_with_dilation[idx]))
            width *= 2

        self.pool = nn.AdaptiveAvgPool2d((1, 1)) if pool == "avg" else nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(width // 2 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, torchvision.models.resnet.BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _forward_impl(self: Self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for layer in self.layers:
            x = layer(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
