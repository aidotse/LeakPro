#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Per-image reconstruction metrics: PSNR, SSIM and (optionally) LPIPS."""
from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

_LPIPS_MODELS: dict[tuple[str, str], object] = {}


def psnr_per_image(recon: Tensor, target: Tensor, data_range: float = 1.0) -> Tensor:
    """PSNR of each image of two ``[N, C, H, W]`` batches, ``inf`` for identical images."""
    mse = ((recon - target) ** 2).flatten(1).mean(dim=1)
    return 10 * torch.log10(data_range ** 2 / mse)


def ssim_per_image(recon: Tensor, target: Tensor, data_range: float = 1.0) -> Tensor:
    """SSIM of each image of two ``[N, C, H, W]`` batches.

    Raises:
        ImportError: If ``torchmetrics`` (part of the ``federated`` extra) is not installed.

    """
    try:
        from torchmetrics.functional.image import structural_similarity_index_measure  # noqa: PLC0415 -- heavy import, deferred
    except ImportError as exc:
        raise ImportError("SSIM needs the 'torchmetrics' package: pip install torchmetrics") from exc

    return structural_similarity_index_measure(recon, target, data_range=data_range, reduction="none")


def lpips_per_image(recon: Tensor, target: Tensor, net: Literal["alex", "vgg"] = "alex") -> Tensor:
    """LPIPS distance of each image pair; inputs in ``[0, 1]`` are mapped to ``[-1, 1]``.

    Raises:
        ImportError: If the optional ``lpips`` package is not installed.

    """
    try:
        import lpips  # noqa: PLC0415 -- optional dependency, imported on use
    except ImportError as exc:
        raise ImportError("LPIPS needs the optional 'lpips' package: pip install lpips") from exc
    key = (net, str(recon.device))
    if key not in _LPIPS_MODELS:
        _LPIPS_MODELS[key] = lpips.LPIPS(net=net, verbose=False).to(recon.device).eval()
    with torch.no_grad():
        return _LPIPS_MODELS[key](recon * 2 - 1, target * 2 - 1).flatten()
