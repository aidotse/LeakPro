#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import math
import sys

import pytest
import torch

from leakpro.attacks.gia_attacks.modular.components.evaluators import ImageEvaluator
from leakpro.attacks.gia_attacks.modular.components.utils import image_metrics


def _pair() -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.rand(2, 3, 8, 8, generator=torch.Generator().manual_seed(0)) * 0.8 + 0.1
    recon = target.clone()
    recon[0] += 0.1
    recon[1] += 0.01
    return recon, target


def test_psnr_per_image_matches_the_definition() -> None:
    recon, target = _pair()
    expected = [10 * math.log10(1.0 / ((recon[i] - target[i]) ** 2).mean().item()) for i in range(2)]
    assert image_metrics.psnr_per_image(recon, target).tolist() == pytest.approx(expected, rel=1e-5)


def test_the_evaluator_reports_the_mean_of_per_image_scores_not_the_batch_mse_score() -> None:
    recon, target = _pair()
    result = ImageEvaluator().evaluate(recon, target)
    per_image = image_metrics.psnr_per_image(recon, target)
    assert result.metrics["psnr"] == pytest.approx(per_image.mean().item(), rel=1e-5)
    batch_psnr = 10 * math.log10(1.0 / ((recon - target) ** 2).mean().item())
    assert result.metrics["psnr"] != pytest.approx(batch_psnr, rel=1e-3)
    assert result.extras["psnr_per_image"] == pytest.approx(per_image.tolist(), rel=1e-5)
    assert len(result.extras["ssim_per_image"]) == 2


def test_lpips_without_the_package_raises_with_an_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "lpips", None)
    recon, target = _pair()
    with pytest.raises(ImportError, match="pip install lpips"):
        ImageEvaluator(lpips="alex").evaluate(recon, target)


def test_lpips_scores_identical_images_as_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    lpips = pytest.importorskip("lpips")
    # Build the network with a randomly initialised backbone so no ImageNet weights are downloaded; the wrapper under
    # test (input mapping, caching, flattening) is the same either way.
    real = lpips.LPIPS
    monkeypatch.setattr(lpips, "LPIPS", lambda **kw: real(**{**kw, "pnet_rand": True}))
    monkeypatch.setattr(image_metrics, "_LPIPS_MODELS", {})
    # AlexNet's pooling stack needs at least 32x32 pixels; smaller images fail inside torch.
    target = torch.rand(2, 3, 32, 32, generator=torch.Generator().manual_seed(0))
    same = image_metrics.lpips_per_image(target, target)
    assert same.shape == (2,)
    assert same.abs().max() < 1e-6
    other = image_metrics.lpips_per_image(1.0 - target, target)
    assert (other > same).all()
