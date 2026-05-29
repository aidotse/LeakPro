#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Shared normalisation helpers for diffusion-based gradient inversion.

These functions convert tensors between the DDPM image space (``[-1, 1]``)
and the dataset-normalised model space used by the FL target model.

Both helpers are imported by:

* :mod:`~leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.mean_strategies`
* :mod:`~leakpro.attacks.gia_attacks.modular.components.diffusion_optimizer`

Centralising them here breaks the previous circular dependency where
``mean_strategies.py`` copy-pasted the function to avoid importing from
``diffusion_optimizer.py``.
"""

from __future__ import annotations

import torch


def denorm_ddpm_to_model_space(
    x: torch.Tensor,
    data_mean: torch.Tensor | None,
    data_std: torch.Tensor | None,
) -> torch.Tensor:
    """Convert DDPM output ``[-1, 1]`` to dataset-normalised model space.

    Pipeline: ``[-1, 1] → [0, 1] → (x - mean) / std``.
    If *data_mean* / *data_std* are ``None`` the tensor is returned as-is.

    Args:
        x: Input tensor with values in ``[-1, 1]``.
        data_mean: Per-channel mean ``[C, 1, 1]`` (broadcast-compatible).
        data_std:  Per-channel std  ``[C, 1, 1]`` (broadcast-compatible).

    Returns:
        Tensor in the same normalised space as the client's training data.
    """
    if data_mean is None or data_std is None:
        return x
    x01 = (x + 1.0) / 2.0
    mean = data_mean.to(x.device)
    std = data_std.to(x.device)
    while mean.ndim < x.ndim:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    return (x01 - mean) / std


def norm_model_space_to_ddpm(
    x: torch.Tensor,
    data_mean: torch.Tensor | None,
    data_std: torch.Tensor | None,
) -> torch.Tensor:
    """Convert dataset-normalised model space back to DDPM ``[-1, 1]``.

    Inverse of :func:`denorm_ddpm_to_model_space`::

        x_01   = x_model * std + mean   # un-normalise → [0, 1]
        x_ddpm = 2 * x_01 - 1          # [0, 1] → DDPM [-1, 1]

    If *data_mean* / *data_std* are ``None`` the tensor is returned as-is.

    Args:
        x: Input tensor in dataset-normalised space.
        data_mean: Per-channel mean ``[C, 1, 1]`` (broadcast-compatible).
        data_std:  Per-channel std  ``[C, 1, 1]`` (broadcast-compatible).

    Returns:
        Tensor with values in ``[-1, 1]``.
    """
    if data_mean is None or data_std is None:
        return x
    mean = data_mean.to(x.device)
    std = data_std.to(x.device)
    while mean.ndim < x.ndim:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    x01 = x * std + mean
    return x01 * 2.0 - 1.0


__all__ = ["denorm_ddpm_to_model_space", "norm_model_space_to_ddpm"]
