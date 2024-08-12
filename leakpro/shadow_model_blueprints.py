"""Models for the datasets."""

from torch import Tensor, flatten, nn
from torch.nn import Module, functional
from torchvision.models import resnet18

from leakpro.import_helper import Self


class NN(Module):
    """NN for Adult dataset."""

    def __init__(self:Self, in_shape:int, num_classes:int) -> None:
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

    def __init__(self:Self,  num_classes:int) -> None:
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
        self.fc3 = nn.Linear(84, self.init_params["num_classes"])

    def forward(self:Self, x:Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
        ----
            x (torch.Tensor): The input tensor.

        Returns:
        -------
            torch.Tensor: The output tensor.s

        """
        x = self.pool(functional.relu(self.conv1(x)))
        x = self.pool(functional.relu(self.conv2(x)))
        x = flatten(x, 1) # flatten all dimensions except batch
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        return self.fc3(x)

class SimpleCNN(Module):
    """Convolutional Neural Network model from Label only attack.

    This CNN has four convolution layers with ReLU activations. The first two 3 * 3 convolutions have 32 filters
    and the second two have 64 filters, with a max-pool in between the two. To compute logits we feed the output
    through a fully-connected layer with 512 neurons. This model has 1.2 million parameters. Write it in pytorch.

    """

    def __init__(self: Self, num_classes: int = 10) -> None:
        super().__init__()
        self.init_params = {
            "num_classes": num_classes
        }

        # First two convolutional layers with 32 filters each
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)

        # Max-pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Second two convolutional layers with 64 filters each
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        # Fully connected layer with 512 neurons
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self: Self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
        ----
            x (torch.Tensor): The input tensor.

        Returns:
        -------
            torch.Tensor: The output tensor.

        """
        # First conv block
        x = functional.relu(self.conv1(x))
        x = functional.relu(self.conv2(x))

        # Max pooling
        x = self.pool(x)

        # Second conv block
        x = functional.relu(self.conv3(x))
        x = functional.relu(self.conv4(x))

        # # Flatten the output from conv layers
        x = x.reshape(-1, 64 * 16 * 16)

        # Fully connected layers
        x = functional.relu(self.fc1(x))
        return self.fc2(x)


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
