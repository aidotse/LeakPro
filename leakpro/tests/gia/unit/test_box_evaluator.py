#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""BoxEvaluator: ground truth by constructor or artifact, graceful and loud when absent, device-safe."""
from __future__ import annotations

import logging

import pytest
import torch

pytest.importorskip("torchvision")

from leakpro.attacks.gia_attacks.modular.components.evaluators.box import BoxEvaluator
from leakpro.attacks.gia_attacks.modular.components.utils.boxes import pad_boxes
from leakpro.attacks.gia_attacks.modular.core.state import State, TargetInfo


def _target() -> TargetInfo:
    boxes, labels, mask = pad_boxes([torch.tensor([[10.0, 10.0, 30.0, 30.0]])], [torch.tensor([1])], k_max=1)
    return TargetInfo(boxes=boxes, box_labels=labels, box_mask=mask)


def _image() -> torch.Tensor:
    return torch.zeros(1, 3, 64, 64)


def test_ground_truth_by_constructor_argument() -> None:
    t = _target()
    result = BoxEvaluator(ground_truth=t).evaluate(_image(), _image(), state=State(target=t))
    assert result.metrics["f1_at_iou"] == pytest.approx(1.0)


def test_ground_truth_mapping_without_mask_scores_every_box() -> None:
    t = _target()
    gt = {"boxes": t.boxes, "labels": t.box_labels}
    result = BoxEvaluator(ground_truth=gt).evaluate(_image(), _image(), state=State(target=t))
    assert result.metrics["recall_at_iou"] == pytest.approx(1.0)


def test_artifact_fallback_still_works() -> None:
    t = _target()
    state = State(target=t, artifacts={BoxEvaluator.GT_KEY: {"boxes": t.boxes, "labels": t.box_labels, "mask": t.box_mask}})
    assert BoxEvaluator().evaluate(_image(), _image(), state=state).metrics["f1_at_iou"] == pytest.approx(1.0)


def test_missing_ground_truth_is_empty_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        result = BoxEvaluator().evaluate(_image(), _image(), state=State(target=_target()))
    assert result.metrics == {}
    assert "ground truth" in caplog.text.lower()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a second device")
def test_ground_truth_on_another_device_than_the_reconstruction() -> None:
    t = _target()
    recon_t = t.to("cuda")
    result = BoxEvaluator(ground_truth={"boxes": t.boxes, "labels": t.box_labels}).evaluate(
        _image().cuda(), _image().cuda(), state=State(target=recon_t))
    assert result.metrics["f1_at_iou"] == pytest.approx(1.0)
