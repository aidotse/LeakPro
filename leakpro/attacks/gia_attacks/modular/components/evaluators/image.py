#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Image reconstruction quality: PSNR / SSIM under optimal row matching."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from torch import Tensor

from leakpro.attacks.gia_attacks.modular.components.utils import image_metrics
from leakpro.attacks.gia_attacks.modular.components.utils.matching import hungarian_align
from leakpro.attacks.gia_attacks.modular.core.evaluation import EvalResult
from leakpro.attacks.gia_attacks.modular.core.interfaces import Evaluator

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.state import Context, State


class ImageEvaluator(Evaluator):
    """PSNR/SSIM (and optionally LPIPS) on [N, C, H, W] batches in data space ([0, 1] pixels).

    Each metric is the mean of the per-image scores, the convention of the gradient-inversion
    literature; the individual scores are returned in ``extras`` as ``<metric>_per_image``.

    A reconstructed batch has no inherent row order — the attack recovers a
    *set* of images — so batches of more than one are Hungarian-matched to the
    ground truth before scoring.  The permutation is returned in ``extras`` so
    callers can apply the same reordering when visualising.

    A pixel-exact image has an infinite PSNR, so a batch containing one has a mean PSNR of ``inf``.

    Args:
        data_range: Pixel value range of the images.
        lpips: Backbone of the LPIPS metric, or ``None`` to skip it. Needs the optional ``lpips``
            package.

    """

    def __init__(self, data_range: float = 1.0, lpips: Literal["alex", "vgg"] | None = None) -> None:
        self.data_range = data_range
        self.lpips = lpips

    def evaluate(
        self,
        recon: Tensor,
        target: Tensor,
        *,
        ctx: Context | None = None,  # noqa: ARG002 -- interface signature
        state: State | None = None,  # noqa: ARG002 -- interface signature
    ) -> EvalResult:
        """Aligns reconstructed images to the targets with the Hungarian algorithm and reports image-quality metrics."""
        if recon.shape[0] > 1:
            recon, col_ind = hungarian_align(target, recon)
            permutation = col_ind.tolist()
        else:
            permutation = [0]

        per_image = {
            "psnr": image_metrics.psnr_per_image(recon, target, self.data_range),
            "ssim": image_metrics.ssim_per_image(recon, target, self.data_range),
        }
        if self.lpips is not None:
            per_image["lpips"] = image_metrics.lpips_per_image(recon, target, self.lpips)

        return EvalResult(
            metrics={name: values.mean().item() for name, values in per_image.items()},
            extras={
                "permutation": permutation,
                "matched_recon": recon,
                **{f"{name}_per_image": values.tolist() for name, values in per_image.items()},
            },
        )
