import torch
import torch.nn as nn


class NN(nn.Module):
    """NN for Purchase dataset."""

    def __init__(self, in_shape, num_classes=10, last_activation=True):
        super().__init__()
        self.fc1 = nn.Linear(in_shape, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

        # Enable the last activaton layer
        self.last_activation = last_activation

    def forward(self, inputs):
        """Forward pass of the model."""
        inputs = inputs.flatten(1)
        y = torch.tanh(self.fc1(inputs))
        if self.last_activation:
         return torch.tanh(self.fc2(y))
        return self.fc2(y)
