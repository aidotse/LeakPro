import torch
import torch.nn as nn
import torch.nn.functional as F


class NN(nn.Module):
    """NN for Purchase dataset."""

    def __init__(self, in_shape, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_shape, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, num_classes)

    def forward(self, inputs):
        """Forward pass of the model."""
        inputs = inputs.flatten(1)
        outputs = F.relu(self.fc1(inputs))
        outputs = F.relu(self.fc2(outputs))
        outputs = F.relu(self.fc3(outputs))
        return outputs

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
    
    
    
    