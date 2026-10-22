#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""CIFAR model architecture for the webapp: upload this as the architecture file in Step 2.

WideResNet-28-2, the architecture ``cifar_main.ipynb`` audits. Three constraints come from how the
webapp reads this file, so keep them if you edit it:

- The webapp trains the *first* ``nn.Module`` subclass in this module's namespace. ``WideResNet`` is
  therefore defined first, and nothing here imports a module class (``from torch import nn`` binds a
  module, not a class). The original ``target_model_class.py`` broke this by importing torchvision's
  ``BasicBlock`` at the top.
- Every constructor argument has a default, because the webapp passes only ``num_classes``.
- Every constructor argument is stored under its own name, because LeakPro rebuilds shadow models
  from ``model_metadata.init_params``, which it reads back from those attributes.

The residual sum is out-of-place and the ReLUs are not in-place, so Opacus can wrap the model for
DP-SGD once its BatchNorm layers are replaced (the webapp applies ``ModuleValidator.fix``).
"""

from torch import Tensor, nn


class WideResNet(nn.Module):
    """WideResNet for 32x32 RGB inputs (Zagoruyko and Komodakis, BMVC 2016)."""

    def __init__(self, num_classes: int = 10, depth: int = 28, widen_factor: int = 2,
                 drop_rate: float = 0.0) -> None:
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError(f"depth must be 6n + 4, got {depth}")
        self.num_classes = num_classes
        self.depth = depth
        self.widen_factor = widen_factor
        self.drop_rate = drop_rate

        blocks_per_group = (depth - 4) // 6
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.group1 = _make_group(widths[0], widths[1], blocks_per_group, stride=1, drop_rate=drop_rate)
        self.group2 = _make_group(widths[1], widths[2], blocks_per_group, stride=2, drop_rate=drop_rate)
        self.group3 = _make_group(widths[2], widths[3], blocks_per_group, stride=2, drop_rate=drop_rate)
        self.bn = nn.BatchNorm2d(widths[3])
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[3], num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Return class logits for a batch of images."""
        out = self.conv1(x)
        out = self.group3(self.group2(self.group1(out)))
        out = self.relu(self.bn(out))
        return self.fc(self.pool(out).flatten(1))


class _WideBasicBlock(nn.Module):
    """Pre-activation residual block. Defined after WideResNet on purpose: see the module docstring."""

    def __init__(self, in_planes: int, out_planes: int, stride: int, drop_rate: float) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.shortcut = (
            nn.Identity() if in_planes == out_planes and stride == 1
            else nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        pre = self.relu1(self.bn1(x))
        # Projection shortcuts take the pre-activated input, identity shortcuts the raw one.
        shortcut = x if isinstance(self.shortcut, nn.Identity) else self.shortcut(pre)
        out = self.conv1(pre)
        out = self.conv2(self.dropout(self.relu2(self.bn2(out))))
        return out + shortcut


def _make_group(in_planes: int, out_planes: int, num_blocks: int, stride: int,
                drop_rate: float) -> nn.Sequential:
    """Stack ``num_blocks`` residual blocks; only the first changes width and resolution."""
    layers = [_WideBasicBlock(in_planes, out_planes, stride, drop_rate)]
    layers += [_WideBasicBlock(out_planes, out_planes, 1, drop_rate) for _ in range(num_blocks - 1)]
    return nn.Sequential(*layers)
