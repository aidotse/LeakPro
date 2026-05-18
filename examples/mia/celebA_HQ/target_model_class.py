import torch.nn as nn
import torch.nn.functional as F
from opacus.validators import ModuleValidator
from torchvision.models.resnet import BasicBlock, ResNet


class BasicBlockNoInplaceAdd(BasicBlock):
    """Torchvision BasicBlock variant without in-place residual addition."""

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Avoid in-place op (`out += identity`) for Opacus compatibility.
        out = out + identity
        out = self.relu(out)

        return out


def _build_resnet18(num_classes: int) -> ResNet:
    model = ResNet(block=BasicBlockNoInplaceAdd, layers=[2, 2, 2, 2])
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.num_classes = num_classes
        self.model = _build_resnet18(self.num_classes)

    def forward(self, x):
        return self.model(x)


class CNN(nn.Module):
    def __init__(self, num_classes=10, dpsgd=True):
        super(CNN, self).__init__()
        self.dpsgd = dpsgd
        self.num_classes = num_classes

        self.init_model()

    def init_model(self):
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(25088, 256),
            nn.Linear(256, self.num_classes),
        )
        
        self.model = nn.Sequential(
            self.features,
            nn.Flatten(),
            self.classifier
        )

    def validate_model(self):
        """Compatibility no-op; DP validation/fixing is handled in the input handler."""
        return None

    def reset_validation(self):
        self.init_model()

    def forward(self, x):
        return self.model(x)


class SampleConvNet(nn.Module):
    def __init__(self, num_classes=10, dpsgd=True):
        super().__init__()
        self.dpsgd = dpsgd
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, self.num_classes)

    def forward(self, x):
        # x of shape [B, 3, 224, 224]
        x = F.relu(self.conv1(x))  # -> [B, 16, 112, 112]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 111, 111]
        x = F.relu(self.conv2(x))  # -> [B, 32, 54, 54]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 53, 53]
        x = F.adaptive_avg_pool2d(x, (4, 4))  # -> [B, 32, 4, 4]
        x = x.view(x.size(0), -1)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x
    
class ResNet18_DPsgd(nn.Module):
    def __init__(self, num_classes=10, dpsgd=False, validate=True):
        super(ResNet18_DPsgd, self).__init__()

        self.dpsgd = dpsgd
        self.validate = validate
        self.num_classes = num_classes

        self.init_model()
        if self.dpsgd or self.validate:
            self.validate_model()

    def forward(self, x):
        return self.model(x)

    def init_model(self,):
        self.model = _build_resnet18(self.num_classes)

    def validate_model(self,):
        self.model = ModuleValidator.fix(self.model)

    def reset_validation(self,):
        self.init_model()
