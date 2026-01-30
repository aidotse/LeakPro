import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(CausalConv1d, self).__init__()
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              dilation=dilation, padding=self.padding)
    
    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding]  # Remove future time steps

class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, dilation_channels, skip_channels, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        self.conv_filter = CausalConv1d(residual_channels, dilation_channels, kernel_size, dilation)
        self.conv_gate = CausalConv1d(residual_channels, dilation_channels, kernel_size, dilation)
        self.conv_residual = nn.Conv1d(dilation_channels, residual_channels, 1)
        self.conv_skip = nn.Conv1d(dilation_channels, skip_channels, 1)
    
    def forward(self, x, skip_sum):
        filter_out = torch.tanh(self.conv_filter(x))
        gate_out = torch.sigmoid(self.conv_gate(x))
        output = filter_out * gate_out
        
        residual_out = self.conv_residual(output) + x  # Residual connection
        skip_out = self.conv_skip(output)
        
        if skip_sum is None:
            skip_sum = skip_out
        else:
            skip_sum = skip_sum + skip_out
        
        return residual_out, skip_sum

class WaveNet(nn.Module):
    def __init__(self, input_dim, horizon, residual_channels=32, dilation_channels=32, skip_channels=32, kernel_size=2, num_layers=4):
        super(WaveNet, self).__init__()
        self.horizon = horizon
        self.input_conv = nn.Conv1d(input_dim, residual_channels, kernel_size=1)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(residual_channels, dilation_channels, skip_channels, kernel_size, 2**i)
            for i in range(num_layers)
        ])
        self.output_conv1 = nn.Conv1d(skip_channels, skip_channels, 1)
        self.output_conv2 = nn.Conv1d(skip_channels, input_dim, 1)

        self.init_params = {"input_dim": input_dim,
                            "horizon": horizon,
                            "residual_channels": residual_channels,
                            "dilation_channels": dilation_channels,
                            "skip_channels": skip_channels,
                            "kernel_size": kernel_size,
                            "num_layers": num_layers}

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Conv expects channel dim before time dim
        x = self.input_conv(x)
        skip_sum = None
        
        for block in self.residual_blocks:
            x, skip_sum = block(x, skip_sum)
        
        x = F.relu(skip_sum)
        x = F.relu(self.output_conv1(x))
        x = self.output_conv2(x)
        x = x[:, :, -self.horizon:] 
        x = x.permute(0, 2, 1)  # Permute back to (samples, timesteps, variables) 
        return x