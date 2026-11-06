#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Euclidean projection onto an L1 ball.

Used by GIFD's intermediate feature stages to keep each stage's feature update bounded (Duchi et al. 2008).
"""
from __future__ import annotations

import torch
from torch import Tensor


def project_l1_ball(x: Tensor, radius: float) -> Tensor:
    """Project each sample in ``x`` onto the L1 ball of the given ``radius``.

    Samples already inside the ball (``sum(abs(x)) <= radius``) are returned
    unchanged. Operates per-sample on a ``[B, *dims]`` tensor (dim 0 is the
    sample dimension; the rest is flattened for the projection).

    """
    orig_shape = x.shape
    flat = x.flatten(1)
    abs_flat = flat.abs()
    over_budget = abs_flat.sum(dim=1) > radius

    if not over_budget.any():
        return x

    out = flat.clone()
    selected = abs_flat[over_budget]
    sorted_abs, _ = torch.sort(selected, dim=1, descending=True)
    cssv = sorted_abs.cumsum(dim=1) - radius
    idx = torch.arange(1, sorted_abs.shape[1] + 1, device=x.device, dtype=x.dtype)
    cond = sorted_abs - cssv / idx > 0
    rho = cond.float().sum(dim=1).clamp(min=1).long() - 1
    theta = cssv.gather(1, rho.unsqueeze(1)).squeeze(1) / (rho.float() + 1)
    theta = theta.clamp(min=0)

    projected = torch.sign(flat[over_budget]) * (abs_flat[over_budget] - theta.unsqueeze(1)).clamp(min=0)
    out[over_budget] = projected
    return out.view(orig_shape)
