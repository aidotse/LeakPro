"""ResNet in PyTorch (with configurable normalization layer)."""

from torch import nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes=10,
        width_factor=1,
        norm_layer=None,            # NEW
    ):
        super(ResNet, self).__init__()

        # NEW: default norm = BatchNorm2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.norm_layer = norm_layer  # NEW
        self.width_factor = width_factor
        base_width = 64 * width_factor
        self.in_planes = base_width

        self.conv1 = conv3x3(3, base_width)
        self.bn1 = self.norm_layer(base_width)  # CHANGED
        self.layer1 = self._make_layer(block, 64 * width_factor, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128 * width_factor, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256 * width_factor, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512 * width_factor, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * width_factor * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            # CHANGED: pass norm_layer into block
            layers.append(block(self.in_planes, planes, stride, norm_layer=self.norm_layer))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, lin=0, lout=5):
        out = x
        if lin < 1 and lout > -1:
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)
        if lin < 2 and lout > 0:
            out = self.layer1(out)
        if lin < 3 and lout > 1:
            out = self.layer2(out)
        if lin < 4 and lout > 2:
            out = self.layer3(out)
        if lin < 5 and lout > 3:
            out = self.layer4(out)
        if lout > 4:
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        return out


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock with configurable norm."""
    expansion = 1

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        norm_layer=None,
    ):
        super(PreActBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.bn1 = norm_layer(in_planes)     # CHANGED
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = norm_layer(planes)        # CHANGED
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
                )
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Callable, List
from functools import partial


# -----------------------------
# Utils
# -----------------------------
class StochasticDepth(nn.Module):
    """DropPath / StochasticDepth (per-sample)"""
    def __init__(self, p: float, mode: str = "row"):
        super().__init__()
        if mode != "row":
            raise ValueError("Only mode='row' supported in this minimal impl.")
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        # per-sample mask
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class LayerNorm2d(nn.Module):
    """LayerNorm over channels for NCHW tensors via channels-last transform."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NCHW -> NHWC
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        # NHWC -> NCHW
        x = x.permute(0, 3, 1, 2)
        return x


@dataclass
class CNBlockConfig:
    input_channels: int
    out_channels: Optional[int]
    num_layers: int


# -----------------------------
# ConvNeXt blocks
# -----------------------------
class CNBlockPreLN(nn.Module):
    """
    Reference (torchvision-like):
      DWConv -> (to NHWC) -> LN -> Linear -> GELU -> Linear -> (to NCHW)
    """
    def __init__(self, dim: int, layer_scale: float, sd_prob: float,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)
        self.to_nhwc = Permute([0, 2, 3, 1])
        self.norm = norm_layer(dim)
        self.pw1 = nn.Linear(dim, 4 * dim, bias=True)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(4 * dim, dim, bias=True)
        self.to_nchw = Permute([0, 3, 1, 2])

        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.drop_path = StochasticDepth(sd_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.to_nhwc(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.to_nchw(x)

        x = self.layer_scale * x
        x = self.drop_path(x)
        return residual + x


class CNBlockPostActLN(nn.Module):
    """
    Your requested variant:
      DWConv -> (to NHWC) -> Linear -> GELU -> LN -> Linear -> (to NCHW)

    Note: LN now normalizes the *activated* hidden (4*dim).
    This is "post-activation norm" inside the MLP.
    """
    def __init__(self, dim: int, layer_scale: float, sd_prob: float,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)
        self.to_nhwc = Permute([0, 2, 3, 1])

        self.pw1 = nn.Linear(dim, 4 * dim, bias=True)
        self.act = nn.GELU()
        self.norm = norm_layer(4 * dim)          # <-- moved after activation, on hidden width
        self.pw2 = nn.Linear(4 * dim, dim, bias=True)

        self.to_nchw = Permute([0, 3, 1, 2])

        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.drop_path = StochasticDepth(sd_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.to_nhwc(x)

        x = self.pw1(x)
        x = self.act(x)
        x = self.norm(x)                         # <-- post-act LN
        x = self.pw2(x)

        x = self.to_nchw(x)

        x = self.layer_scale * x
        x = self.drop_path(x)
        return residual + x


class ConvNeXt(nn.Module):
    def __init__(
        self,
        block_setting: List[CNBlockConfig],
        num_classes: int = 10,
        stochastic_depth_prob: float = 0.1,
        layer_scale: float = 1e-6,
        block: Callable[..., nn.Module] = CNBlockPreLN,
        norm_layer_2d: Optional[Callable[..., nn.Module]] = None,
        downsample_bn: bool = False,
        downsample_norm_input: bool = True,
        small_stem = False,
    ):
        super().__init__()
        if norm_layer_2d is None:
            norm_layer_2d = partial(LayerNorm2d, eps=1e-6)


        first_dim = block_setting[0].input_channels

        if small_stem:
            # CIFAR-friendly: preserve 32x32 resolution initially
            stem_kernel, stem_stride, stem_pad = 3, 1, 1
        else:
            # ImageNet-style ConvNeXt stem
            stem_kernel, stem_stride, stem_pad = 4, 4, 0

        self.stem = nn.Sequential(
            nn.Conv2d(3, first_dim, kernel_size=stem_kernel, stride=stem_stride, padding=stem_pad, bias=True),
            norm_layer_2d(first_dim),
        )

        total_blocks = sum(c.num_layers for c in block_setting)
        block_id = 0
        layers: List[nn.Module] = []

        for cfg in block_setting:
            stage = []
            for _ in range(cfg.num_layers):
                sd = 0.0 if total_blocks == 1 else stochastic_depth_prob * block_id / (total_blocks - 1)
                stage.append(block(cfg.input_channels, layer_scale, sd))
                block_id += 1
            layers.append(nn.Sequential(*stage))

            if cfg.out_channels is not None:
                layers.append(
                    Downsample(
                        in_ch=cfg.input_channels,
                        out_ch=cfg.out_channels,
                        norm2d=norm_layer_2d,
                        downsample_norm_input=downsample_norm_input,
                        downsample_bn=downsample_bn,
                    )
                )

        self.stages = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        last = block_setting[-1]
        last_dim = last.out_channels if last.out_channels is not None else last.input_channels
        self.head = nn.Sequential(
            norm_layer_2d(last_dim),
            nn.Flatten(1),
            nn.Linear(last_dim, num_classes, bias=True),
        )

        self._init_weights()


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.avgpool(x)
        x = self.head(x)
        return x

class CNBlockPostActBN(nn.Module):
    """
    ConvNeXt-style block with post-activation BatchNorm:
      DWConv -> PWConv -> GELU -> BN -> PWConv
    """
    def __init__(self, dim: int, layer_scale: float, sd_prob: float):
        super().__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)

        self.pw1 = nn.Conv2d(dim, 4 * dim, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.bn = nn.BatchNorm2d(4 * dim)   # <-- post-activation BN
        self.pw2 = nn.Conv2d(4 * dim, dim, kernel_size=1, bias=True)

        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.drop_path = StochasticDepth(sd_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.dwconv(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.pw2(x)

        x = self.layer_scale * x
        x = self.drop_path(x)
        return residual + x


class Downsample(nn.Module):
    """
    Stage-transition projection (Conv stride=2) with ablation flags:
      - downsample_norm_input: apply norm BEFORE conv (ConvNeXt-ish) vs not
      - downsample_bn: apply BN AFTER conv (ResNet-ish) vs not
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        norm2d: Callable[[int], nn.Module],
        downsample_norm_input: bool = True,
        downsample_bn: bool = False,
    ):
        super().__init__()
        self.downsample_norm_input = downsample_norm_input

        self.pre_norm = norm2d(in_ch) if downsample_norm_input else nn.Identity()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2, bias=True)

        # BN is defined on OUT channels (like torchvision ResNet downsample)
        self.post_bn = nn.BatchNorm2d(out_ch) if downsample_bn else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x)
        x = self.conv(x)
        x = self.post_bn(x)
        return x

def convnext_tiny_cifar10(
    num_classes: int = 10,
    stochastic_depth_prob: float = 0.1,
    block: Callable[..., nn.Module] = CNBlockPreLN,
    downsample_bn: bool = False,
    downsample_norm_input: bool = True,
    small_stem: bool = False,
) -> ConvNeXt:
    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 9),
        CNBlockConfig(768, None, 3),
    ]
    return ConvNeXt(
        block_setting=block_setting,
        num_classes=num_classes,
        stochastic_depth_prob=stochastic_depth_prob,
        block=block,
        downsample_bn=downsample_bn,
        downsample_norm_input=downsample_norm_input,
        small_stem=small_stem,
    )
