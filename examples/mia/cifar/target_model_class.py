import torch.nn as nn
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

class ResNet18_DPsgd(nn.Module):
    def __init__(self, num_classes=10, dpsgd=True):
        super(ResNet18_DPsgd, self).__init__()

        self.dpsgd = dpsgd
        self.num_classes = num_classes

        self.init_model()
        if self.dpsgd:
            self.validate()

    def forward(self, x):
        return self.model(x)

    def init_model(self,):
        self.model = _build_resnet18(self.num_classes)

    def validate(self,):
        self.model = ModuleValidator.fix(self.model)

    def reset_validation(self,):
        self.init_model()
