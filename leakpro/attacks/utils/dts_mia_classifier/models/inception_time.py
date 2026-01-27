"""PyTorch implementation of the (single predictor) InceptionTime architecture.

For the original implementation, see https://github.com/hfawaz/InceptionTime.
Paper reference: Fawaz et al. InceptionTime: Finding AlexNet for Time Series Classification. 2019.
"""

import torch
from torch import nn


def default_kernel_sizes(seq_length: int) -> list[int]:
    """Gets the default convolutional kernel sizes.

    Based on results from Fawaz et al., best filter length depending on the length of the input time series.
    """
    return [2, 4, 8] if seq_length < 81 else [10, 20, 40]

class GlobalAveragePooling1D(nn.Module):
    """1D Global Average Pooling layer."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return torch.mean(x, dim=2)

class ShortcutLayer(nn.Module):
    """A Shortcut layer."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding="same", bias=False)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # noqa: D102
        shortcut = self.conv(x)
        shortcut = self.bnorm(shortcut)
        x = shortcut + y
        return self.relu(x)

class InceptionModule(nn.Module):
    """An Inception Module."""

    def __init__(
            self,
            use_bottleneck: bool,
            bottleneck_size: int,
            in_channels: int,
            num_filters: int,
            kernel_sizes: list[int]
        ) -> None:
        super().__init__()

        if use_bottleneck and in_channels > 1:
            self.input_layer = nn.Conv1d(in_channels, bottleneck_size, kernel_size=1, stride=1, padding="same", bias=False)
        else:
            self.input_layer = nn.Identity()

        self.convs = nn.ModuleList()
        inception_in_channels = bottleneck_size if use_bottleneck else in_channels
        for kernel_size in kernel_sizes:
            self.convs.append(
                nn.Conv1d(inception_in_channels, num_filters, kernel_size=kernel_size, stride=1, padding="same", bias=False)
            )

        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)    # padding=1 results in 'same' padding for kernel_size=3, stride=1  # noqa: E501
        self.mp_conv = nn.Conv1d(in_channels, num_filters , kernel_size=1, stride=1, padding="same", bias=False)

        num_convs = len(kernel_sizes) + 1
        num_channels_after_concat = num_filters * num_convs
        self.bnorm = nn.BatchNorm1d(num_channels_after_concat)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        inception_input = self.input_layer(x)
        conv_list_out = [conv(inception_input) for conv in self.convs]
        mp_conv_out = self.mp_conv(self.max_pool(x))
        conv_list_out.append(mp_conv_out)
        x = torch.cat(conv_list_out, dim=1)
        x = self.bnorm(x)
        return self.relu(x)

class InceptionTime(nn.Module):
    """The InceptionTime model."""

    def __init__(
            self,
            in_channels: int,
            seq_len: int,
            num_filters: int = 32,
            use_residual: bool = True,
            use_bottleneck: bool = True,
            bottleneck_size: int = 32,
            depth: int = 6,
            kernel_sizes: list[int] = None
        ) -> None:
        super().__init__()

        self.use_residual = use_residual

        if kernel_sizes is None:
            kernel_sizes = default_kernel_sizes(seq_len)

        num_inception_module_convs = len(kernel_sizes) + 1
        num_channels_after_concat = num_filters * num_inception_module_convs

        self.inception_modules = nn.ModuleList([
            InceptionModule(use_bottleneck, bottleneck_size, in_channels, num_filters, kernel_sizes),
            *[InceptionModule(use_bottleneck, bottleneck_size, num_channels_after_concat, num_filters, kernel_sizes) for _ in range(1, depth)]  # noqa: E501
        ])

        if use_residual:
            num_shortcuts = len([d for d in range(depth) if d % 3 == 2])
            self.shortcuts = nn.ModuleList([
                ShortcutLayer(in_channels, num_channels_after_concat),
                *[ShortcutLayer(num_channels_after_concat, num_channels_after_concat) for _ in range(1, num_shortcuts)]
            ])

        self.gap = GlobalAveragePooling1D()
        self.fc = nn.Linear(num_channels_after_concat, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        x = x.permute(0, 2, 1)  # channels first
        input_residual = x
        for d, inception_module in enumerate(self.inception_modules):
            x = inception_module(x)
            if self.use_residual and d % 3 == 2:
                x = self.shortcuts[d // 3](input_residual, x)
                input_residual = x

        x = self.gap(x)
        x = self.fc(x)
        return self.sigmoid(x) # use sigmoid to model membership probability (original InceptionTime uses softmax to support arbitrary amount of classes)  # noqa: E501
