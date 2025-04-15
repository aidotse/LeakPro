import torchvision.models as models
import torch.nn as nn

from opacus.validators import ModuleValidator

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.num_classes = num_classes
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)

    def forward(self, x):
        return self.model(x)

class ResNet18_DPsgd(nn.Module):
    def __init__(self, num_classes=10, dpsgd=True, **kwargs):
        super(ResNet18_DPsgd, self).__init__()
        self.num_classes = num_classes
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)

        # Run validation 
        if dpsgd:
            self.validate()

    def forward(self, x):
        return self.model(x)

    def validate(self,):
        self.model = ModuleValidator.fix(self.model)