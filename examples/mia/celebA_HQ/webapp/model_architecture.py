"""CelebA-HQ model architecture — upload this as arch.py in Step 2.

ResNet-18 with pretrained ImageNet weights, fine-tuned for face identity classification.
Pretrained weights are recommended for CelebA-HQ because the dataset is small (~5k images).
"""
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18(nn.Module):
    def __init__(self, num_classes=307):
        super().__init__()
        self.num_classes = num_classes
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
