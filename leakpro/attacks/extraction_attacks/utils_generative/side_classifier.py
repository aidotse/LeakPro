#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Time-dependent ResNet classifier used for SIDE surrogate guidance."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SinusoidalTimestepEmbedding(nn.Module):
    """Encode scalar diffusion timesteps with sinusoidal features."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 2:
            raise ValueError("dimension must be at least 2.")
        self.dimension = dimension

    def forward(self, timesteps: Tensor) -> Tensor:
        """Return one sinusoidal embedding per timestep."""
        if timesteps.ndim != 1:
            raise ValueError("timesteps must be one-dimensional.")
        half = self.dimension // 2
        exponent = -math.log(10_000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        exponent = exponent / max(half - 1, 1)
        phases = timesteps.float().unsqueeze(1) * exponent.exp().unsqueeze(0)
        embedding = torch.cat([phases.sin(), phases.cos()], dim=1)
        if self.dimension % 2:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding


class TimeResidualBlock(nn.Module):
    """Basic ResNet block with timestep injection after the first normalization."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int, time_dimension: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.time_projection = nn.Linear(time_dimension, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=False)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, inputs: Tensor, time_embedding: Tensor) -> Tensor:
        """Apply the residual block with additive timestep conditioning."""
        residual = self.skip(inputs)
        output = self.bn1(self.conv1(inputs))
        output = output + self.time_projection(time_embedding).unsqueeze(-1).unsqueeze(-1)
        output = self.activation(output)
        output = self.bn2(self.conv2(output))
        return self.activation(output + residual)


class TimeConditionedResNet(nn.Module):
    """ResNet34-shaped pseudo-label classifier with SIDE timestep modules."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        base_width: int = 64,
        blocks: tuple[int, int, int, int] = (3, 4, 6, 3),
        timestep_embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        self.time_embedding = nn.Sequential(
            SinusoidalTimestepEmbedding(timestep_embedding_dim),
            nn.Linear(timestep_embedding_dim, timestep_embedding_dim),
            nn.SiLU(),
            nn.Linear(timestep_embedding_dim, timestep_embedding_dim),
        )
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=False),
        )
        widths = (base_width, base_width * 2, base_width * 4, base_width * 8)
        layers: list[nn.ModuleList] = []
        current_channels = base_width
        for stage, (width, block_count) in enumerate(zip(widths, blocks)):
            stage_blocks = nn.ModuleList()
            for block_index in range(block_count):
                stride = 2 if stage > 0 and block_index == 0 else 1
                stage_blocks.append(TimeResidualBlock(current_channels, width, stride, timestep_embedding_dim))
                current_channels = width
            layers.append(stage_blocks)
        self.layers = nn.ModuleList(layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(current_channels, num_classes)

    def forward(self, images: Tensor, timesteps: Tensor) -> Tensor:
        """Predict surrogate-cluster logits for noisy images and timesteps."""
        if images.ndim != 4:
            raise ValueError("images must be BCHW.")
        if timesteps.shape != (images.shape[0],):
            raise ValueError("timesteps must contain one value per image.")
        time_embedding = self.time_embedding(timesteps)
        output = self.stem(images)
        for stage in self.layers:
            for block in stage:
                output = block(output, time_embedding)
        return self.classifier(self.pool(output).flatten(start_dim=1))
