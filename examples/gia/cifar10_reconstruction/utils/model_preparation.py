"""Models for the datasets."""
import pickle
from typing import Callable, Optional
import os
import torch
import torchvision
from torch import Tensor, nn, save
from torch.utils.data import DataLoader
from torch.nn import Module, CrossEntropyLoss
from torchvision.models import resnet18
from torchvision.models.resnet import Bottleneck, BasicBlock
from tqdm import tqdm

class ResNet18(nn.Module):
    """ResNet-18 model from torchvision."""

    def __init__(self, num_classes:int = 10) -> None:  # noqa: D417
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

    def forward(self, x:Tensor) -> Tensor:
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

    def __init__(self, block: Module=BasicBlock, layers: list=[5, 5, 5], num_classes: int=10,
                 zero_init_residual: bool=False, groups: int = 1, base_width : int =160,
                 replace_stride_with_dilation: Optional[list[bool]] =None, norm_layer: Optional[Callable[..., nn.Module]] =None,
                 strides: list = [1, 2, 2, 2], pool: str = "avg") -> None:  # noqa: B006
        """Initialize as usual. Layers and strides are scriptable."""
        
        self.init_params = {
            "block": block,
            "layers": layers,
            "num_classes": num_classes,
            "zero_init_residual": zero_init_residual,
            "groups": groups,
            "base_width": base_width,
            "replace_stride_with_dilation": replace_stride_with_dilation,
            "norm_layer": norm_layer,
            "strides": strides,
            "pool": pool
        }
        
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


    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for layer in self.layers:
            x = layer(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

def evaluate_model(model: Module, dataloader: DataLoader) -> float:
    """Evaluate the model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def train_model(model: Module, 
                pretrain_loader: DataLoader,
                train_loader: DataLoader, 
                test_loader:DataLoader, 
                epochs:int) -> None:
    """Train the model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()
    model.train()
    model.to(device)
    train_accuracy = 0
    for _ in tqdm(range(epochs), desc="Training Progress"):
        for inputs, labels in pretrain_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            train_accuracy += (outputs.argmax(1) == labels).sum().item()
            loss =  criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        train_accuracy /= len(pretrain_loader.dataset)

        test_accuracy = evaluate_model(model, test_loader)
        print(f"Train accuracy: {train_accuracy}, Test accuracy: {test_accuracy}")
    
    # Store the model and metadata
    # Move the model back to the CPU
    model.to("cpu")
    
    if not os.path.exists("./target"):
        os.mkdir("./target")
    with open("./target/target_model.pkl", "wb") as f:
        save(model.state_dict(), f)
    
    # Store the metadata needed to train the local models
    epochs = 1
    batch_size = 1
    
    meta_data = {}
    meta_data["train_indices"] = train_loader.dataset.indices
    meta_data["test_indices"] = test_loader.dataset.indices
    meta_data["num_train"] = len(meta_data["train_indices"])

    # Write init params
    meta_data["init_params"] = {}
    for key, value in model.init_params.items():
        meta_data["init_params"][key] = value

    # read out optimizer parameters
    meta_data["optimizer"] = {}
    meta_data["optimizer"]["name"] = "sgd"
    meta_data["optimizer"]["lr"] = 1e-3

    # read out criterion parameters
    meta_data["loss"] = {}
    meta_data["loss"]["name"] = "crossentropyloss"

    meta_data["batch_size"] = batch_size
    meta_data["epochs"] = epochs
    meta_data["train_acc"] = train_accuracy
    meta_data["test_acc"] = test_accuracy
    meta_data["train_loss"] = None
    meta_data["test_loss"] = None
    meta_data["dataset"] = "cifar10"

    with open("./target/model_metadata.pkl", "wb") as f:
        pickle.dump(meta_data, f)