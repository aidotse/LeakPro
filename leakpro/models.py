import torch
import torch.nn as nn

class NN(nn.Module):
    """NN for Purchase dataset."""

    def __init__(self, in_shape, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_shape, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, inputs):
        """Forward pass of the model."""
        inputs = inputs.flatten(1)
        y = torch.tanh(self.fc1(inputs))
        outputs = torch.tanh(self.fc2(y))
        return outputs
    