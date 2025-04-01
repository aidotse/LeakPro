import torchvision.models as models
import torch.nn as nn

class ResNet18(nn.Module):
    def __init__(self, **kwargs):
        super(ResNet18, self).__init__()
        # self.num_classes = num_classes
        self.num_classes = 10
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
    
    def forward(self, x):
        return self.model(x)

    def validator(self,):
        pass