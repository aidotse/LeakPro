#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Per-sample radius-preserving projection, used by GIFD's stage-0 latent search.

Keep the optimized latent code on its initial hypersphere.
"""
from __future__ import annotations

from torch import Tensor


def project_to_radius(z: Tensor, initial_radius: Tensor) -> Tensor:
    """Rescale each sample in ``z`` back onto its initial L2-norm radius.

    Args:
        z: ``[B, *dims]`` tensor to project (dim 0 is the per-sample dimension).
        initial_radius: ``[B, 1]`` (or broadcastable) target radius per sample,
            captured once at bind time.

    Returns:
        ``z`` rescaled in place-equivalent (a new tensor) so
        ``out.flatten(1).norm(dim=1) == initial_radius`` (up to float precision).

    """
    flat = z.flatten(1)
    current_radius = flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
    scale = initial_radius.reshape(current_radius.shape) / current_radius
    scaled_flat = flat * scale
    return scaled_flat.view(z.shape)
