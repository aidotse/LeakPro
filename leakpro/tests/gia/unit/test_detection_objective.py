#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""TorchvisionDetection: an objective whose model call needs the target at forward time.

No existing example exercises a detector through the framework, so this is
the smoke test for it. Uses a from-scratch (unpretrained) torchvision Faster
R-CNN with min_size == max_size pinned to the reconstruction's height — the
thing that had to be verified before writing this objective at all, since a
mismatched internal resize would silently break gradient attribution back to
fixed pixel positions.
"""
from __future__ import annotations

import pytest
import torch

torchvision = pytest.importorskip("torchvision")
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn  # noqa: E402

from leakpro.attacks.gia_attacks.modular.components.objectives import TorchvisionDetection  # noqa: E402
from leakpro.attacks.gia_attacks.modular.components.utils.boxes import pad_boxes  # noqa: E402
from leakpro.attacks.gia_attacks.modular.core.axes import AxisManager  # noqa: E402
from leakpro.attacks.gia_attacks.modular.core.protocol import FedAvg  # noqa: E402
from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo  # noqa: E402
from leakpro.attacks.gia_attacks.modular.core.task import Task  # noqa: E402

IMG_SIZE = 64


def _tiny_detector() -> torch.nn.Module:
    # weights=None / weights_backbone=None avoids any network download; the
    # test only needs a real detector forward/backward, not accuracy.
    return fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, min_size=IMG_SIZE, max_size=IMG_SIZE)


def _tiny_single_stage_detector() -> torch.nn.Module:
    # RetinaNet has no RoIAlign, so unlike Faster R-CNN it supports the double
    # backward simulate_dummy_signal needs (create_graph=True) — see
    # components/objectives/detection.py for why Faster R-CNN can't be used here.
    return retinanet_resnet50_fpn(weights=None, weights_backbone=None, min_size=IMG_SIZE, max_size=IMG_SIZE)


def _toy_batch() -> tuple[torch.Tensor, TargetInfo]:
    images = torch.rand(2, 3, IMG_SIZE, IMG_SIZE)
    boxes_list = [
        torch.tensor([[10.0, 10.0, 30.0, 30.0]]),
        torch.tensor([[5.0, 5.0, 20.0, 20.0], [30.0, 30.0, 50.0, 50.0]]),
    ]
    labels_list = [torch.tensor([1]), torch.tensor([2, 1])]
    boxes, labels, mask = pad_boxes(boxes_list, labels_list, k_max=2)
    target = TargetInfo(boxes=boxes, box_labels=labels, box_mask=mask)
    return images, target


def _make_ctx(model: torch.nn.Module, task: Task) -> object:
    import types

    from leakpro.attacks.gia_attacks.modular.core.session import RunSession
    return types.SimpleNamespace(
        model=model,
        task=task,
        device=torch.device("cpu"),
        axes=AxisManager(),
        input_shape=(2, 3, IMG_SIZE, IMG_SIZE),
        session=RunSession(),
    )


class TestTorchvisionDetectionProtocols:
    def test_fedavg_constructs(self):
        Task(TorchvisionDetection(), FedAvg())


class TestTorchvisionDetectionVictimSignal:
    def test_victim_signal_returns_one_gradient_per_param(self):
        model = _tiny_detector()
        model.train()
        images, target = _toy_batch()
        task = Task(TorchvisionDetection())

        obs = task.victim_signal(model, images, target, input_shape=(2, 3, IMG_SIZE, IMG_SIZE))

        assert obs.gradients is not None
        assert len(obs.gradients) == len(list(model.parameters()))
        assert any(g.abs().sum() > 0 for g in obs.gradients)

    def test_empty_box_image_does_not_crash(self):
        """An image with zero real boxes (all-False mask) must not break the forward call."""
        model = _tiny_detector()
        model.train()
        images = torch.rand(2, 3, IMG_SIZE, IMG_SIZE)
        boxes_list = [torch.zeros((0, 4)), torch.tensor([[5.0, 5.0, 20.0, 20.0]])]
        labels_list = [torch.zeros((0,), dtype=torch.long), torch.tensor([2])]
        boxes, labels, mask = pad_boxes(boxes_list, labels_list, k_max=1)
        target = TargetInfo(boxes=boxes, box_labels=labels, box_mask=mask)
        task = Task(TorchvisionDetection())

        obs = task.victim_signal(model, images, target, input_shape=(2, 3, IMG_SIZE, IMG_SIZE))
        assert len(obs.gradients) == len(list(model.parameters()))


class TestTorchvisionDetectionFasterRCNNDoubleBackwardLimitation:
    def test_dummy_signal_raises_on_two_stage_detector(self):
        """Pins the RoIAlign double-backward limitation documented in components/objectives/detection.py —
        so this failure mode is caught by CI, not rediscovered by a preset author."""
        model = _tiny_detector()  # Faster R-CNN
        images, target = _toy_batch()
        images.requires_grad_(True)

        task = Task(TorchvisionDetection())
        ctx = _make_ctx(model, task)
        ctx.session.begin_stage(ctx)

        obs = task.simulate_dummy_signal(images, target, ctx)
        loss = sum(g.pow(2).sum() for g in obs.gradients)
        with pytest.raises(RuntimeError, match="double backward"):
            torch.autograd.grad(loss, images)


class TestTorchvisionDetectionDummySignal:
    def test_simulate_dummy_signal_builds_graph_to_input(self):
        """The attacker path must differentiate through the detector back to pixels —
        this is the actual thing GIA needs, not just that the forward call succeeds.
        Uses RetinaNet (single-stage, no RoIAlign) — see the double-backward
        limitation test above for why Faster R-CNN can't be used here."""
        model = _tiny_single_stage_detector()
        images, target = _toy_batch()
        images.requires_grad_(True)

        task = Task(TorchvisionDetection())
        ctx = _make_ctx(model, task)
        ctx.session.begin_stage(ctx)

        obs = task.simulate_dummy_signal(images, target, ctx)
        loss = sum(g.pow(2).sum() for g in obs.gradients)
        grad_images = torch.autograd.grad(loss, images)[0]

        assert grad_images.shape == images.shape
        assert grad_images.abs().sum() > 0
