#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Integer-translation alignment of reconstructions to a reference image."""
from __future__ import annotations

import torch
from torch import Tensor


def align_translation(reference: Tensor, moving: Tensor, max_shift: int) -> Tensor:
    """Shift each image of ``moving`` to best match the same image of ``reference``.

    Both tensors are ``[..., C, H, W]``. For every image (all leading dimensions flattened) the
    circular shift ``(dy, dx)`` with ``|dy|, |dx| <= max_shift`` that minimises the mean squared
    error to the reference is applied with ``torch.roll``. ``max_shift == 0`` returns ``moving``.

    Raises:
        ValueError: If ``max_shift`` is negative.

    """
    if max_shift < 0:
        raise ValueError(f"max_shift must be non-negative, got {max_shift}.")
    if max_shift == 0:
        return moving
    ref = reference.reshape(-1, *reference.shape[-3:])
    mov = moving.reshape(-1, *moving.shape[-3:])
    candidates = torch.stack([
        mov.roll((dy, dx), dims=(-2, -1))
        for dy in range(-max_shift, max_shift + 1)
        for dx in range(-max_shift, max_shift + 1)
    ])                                                                     # [K, B, C, H, W]
    mse = ((candidates - ref) ** 2).mean(dim=(-3, -2, -1))                 # [K, B]
    best = mse.argmin(dim=0)                                               # [B]
    chosen = candidates[best, torch.arange(mov.shape[0], device=mov.device)]
    return chosen.reshape(moving.shape)
