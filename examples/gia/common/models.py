#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Models for the image examples: a pre-activation ResNet and a tiny CNN for ``--quick``.

The ResNet uses the pre-activation block of He et al., Identity Mappings in Deep Residual Networks, arXiv:1603.05027.
It is adapted from GradAttack: https://github.com/Princeton-SysML/GradAttack.
"""
import torch.nn.functional as F
from torch import nn

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, width_factor=1):
        super().__init__()
        self.width_factor = width_factor
        base_width = 64 * width_factor

        self.in_planes = base_width

        self.conv1 = conv3x3(3, base_width)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.layer1 = self._make_layer(block, 64 * width_factor, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128 * width_factor, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256 * width_factor, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512 * width_factor, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * width_factor * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.avg_pool2d(out, 4)
        return self.linear(out.view(out.size(0), -1))

class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


def cifar_resnet(num_classes: int = 10) -> nn.Module:
    """The PreAct-ResNet-18 (width 2) the benchmark attacks."""
    return ResNet(PreActBlock, [2, 2, 2, 2], num_classes, width_factor=2)


def tiny_cnn(num_classes: int = 10) -> nn.Module:
    """A small BatchNorm CNN for 16x16 inputs; random weights, enough for a smoke run."""
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.AvgPool2d(2),
        nn.Conv2d(8, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.AdaptiveAvgPool2d(2),
        nn.Flatten(), nn.Linear(32, num_classes),
    )


class _TinyDCGAN(nn.Module):
    """Latent ``[B, latent_dim]`` to a 3x16x16 image in the tanh range, as a ``main`` Sequential of ConvTranspose blocks."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 16, 4, 1, 0, bias=False), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 4, 2, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 4, 2, 1, bias=False), nn.Tanh(),
        )

    def forward(self, z, labels=None):  # noqa: ANN001, ANN201, ARG002
        return self.main(z.view(z.shape[0], self.latent_dim, 1, 1))


def tiny_dcgan_generator(latent_dim: int = 16) -> nn.Module:
    """A random-weight DCGAN-style generator for 16x16 images (``--quick``); usable with GIAS, GGL and, wrapped, GIFD."""
    return _TinyDCGAN(latent_dim).eval()


def tiny_unet():  # noqa: ANN201
    """A random-weight one-block diffusers UNet for 16x16 images, built from its config (no download)."""
    from diffusers import UNet2DModel  # noqa: PLC0415

    return UNet2DModel(sample_size=16, in_channels=3, out_channels=3, layers_per_block=1, block_out_channels=(8,),
                       down_block_types=("DownBlock2D",), up_block_types=("UpBlock2D",), norm_num_groups=4).eval()
