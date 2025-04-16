import torch.nn as nn
from torch import sigmoid

class LR(nn.Module):
    def __init__(self, input_dim: int):
        """Initialize the logistic regression model with a single linear layer.

        Args:
        ----
            input_dim (int): The size of the input feature vector.

        """
        super(LR, self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(self.input_dim, 1)  # Binary classification (1 output)
        # Metadata initialization
        self.init_params = {"input_dim": self.input_dim}

    def forward(self, x):
        """Forward pass through the model."""
        return sigmoid(self.linear(x))  # Sigmoid to produce probabilities for binary classification