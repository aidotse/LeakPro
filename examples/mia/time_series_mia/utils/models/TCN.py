import torch
from torch import nn
import pytorch_tcn

class TCN(nn.Module):
    """Full Temporal Convolutional Network with multiple residual blocks."""
    def __init__(self, input_dim, horizon, num_channels=None, kernel_size=2, dropout=0.1):
        super(TCN, self).__init__()
        if num_channels is None:
            num_channels = [32, 32]
        self.init_params = {"input_dim": input_dim,
                            "horizon": horizon,
                            "num_channels": num_channels,
                            "kernel_size": kernel_size,
                            "dropout": dropout}
        self.input_dim = input_dim
        self.horizon = horizon

        self.tcn = pytorch_tcn.TCN(input_dim, num_channels, kernel_size=kernel_size, 
                                   dropout=dropout, causal=False, input_shape="NLC",
                                   dilation_reset = 8)
        self.fc = nn.Linear(num_channels[-1], horizon * input_dim)

    def forward(self, x):
        y = self.tcn(x)
        y_last = y[:, -1, :]
        out = self.fc(y_last)
        out = out.view(x.size(0), self.horizon, self.input_dim)
        return out

