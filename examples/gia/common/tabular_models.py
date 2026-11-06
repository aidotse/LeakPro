#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""MLP builder over the architecture factors (width, depth, norm, dropout, activation)."""
from __future__ import annotations

import torch.nn as nn

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU}
NORMS = {"layernorm": nn.LayerNorm, "batchnorm": nn.BatchNorm1d, "none": None}

# "small": d_hidden=32, 2 hidden layers, LayerNorm, GELU, no dropout.
# "medium"/"large": wider and deeper variants.
PRESETS = {
    "small": dict(d_hidden=32, n_layers=2, norm="layernorm", activation="gelu", dropout=0.0),
    "medium": dict(d_hidden=64, n_layers=3, norm="layernorm", activation="gelu", dropout=0.0),
    "large": dict(d_hidden=128, n_layers=3, norm="layernorm", activation="gelu", dropout=0.0),
}


def build_mlp(in_dim: int, out_dim: int, *, d_hidden: int = 32, n_layers: int = 2, norm: str = "layernorm",
              activation: str = "gelu", dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(n_layers):
        layers.append(nn.Linear(d, d_hidden))
        if NORMS[norm] is not None:
            layers.append(NORMS[norm](d_hidden))
        layers.append(ACTIVATIONS[activation]())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = d_hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)
