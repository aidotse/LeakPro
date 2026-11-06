#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: AGPL-3.0-only
#
"""CriterionDetection with the custom YOLOv8-style detector (examples/gia/coco_yolo_1_image/model.py).

Pins two findings discovered while wiring this model up, both surprising
enough to regress silently otherwise:

1. This model freezes DFL.conv (requires_grad_(False) by construction) — FedSGD
   must filter to trainable parameters before calling torch.autograd.grad
   (which, unlike .backward(), errors if any tensor in its second argument
   doesn't require grad).
2. ComputeLoss.assign() is wrapped in @torch.no_grad(), which detaches the
   ground-truth box values themselves — box coordinates get no gradient at
   all (None, not zero) through this loss, unlike RetinaNet's TorchvisionDetection.

Run against the real model + real ComputeLoss criterion, loaded by file path
from the example (skipped if its dependencies aren't importable).
"""
from __future__ import annotations

import pytest
import torch

from leakpro.attacks.gia_attacks.modular.components.representations.pixel import PixelRepresentation
from leakpro.attacks.gia_attacks.modular.components.targets.box import JointBoxTarget
from leakpro.attacks.gia_attacks.modular.components.utils.boxes import pad_boxes
from leakpro.attacks.gia_attacks.modular.core.axes import AxisManager
from leakpro.attacks.gia_attacks.modular.core.protocol import FedAvg
from leakpro.attacks.gia_attacks.modular.core.state import Context, State, TargetInfo
from leakpro.attacks.gia_attacks.modular.core.task import Task
from yolo_example import load_yolo_example

pytest.importorskip("yaml")

yolo = load_yolo_example()
ComputeLoss, yolo_v8_n_basicblock = yolo.model.ComputeLoss, yolo.model.yolo_v8_n_basicblock

IMG_SIZE = 64


def _yolo_task(compute_loss_fn, **kwargs) -> Task:
    return Task(yolo.attack.yolo_objective(compute_loss_fn, IMG_SIZE), **kwargs)


def _build_compute_loss(model: torch.nn.Module) -> ComputeLoss:
    return ComputeLoss(model)


def _toy_target(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    boxes_list = [torch.tensor([[10.0, 10.0, 30.0, 30.0], [20.0, 20.0, 40.0, 40.0]], device=device)]
    labels_list = [torch.tensor([1, 2], device=device)]
    return pad_boxes(boxes_list, labels_list, k_max=2)


class TestYoloCriterionVictimSignal:
    def test_only_trainable_params_get_gradients(self):
        """DFL.conv is frozen — gradients must be filtered to trainable params,
        not raise, and victim/dummy signals must have matching (filtered) length."""
        device = torch.device("cpu")
        model = yolo_v8_n_basicblock().to(device).train()
        compute_loss_fn = _build_compute_loss(model)

        boxes, labels, mask = _toy_target(device)
        target = TargetInfo(boxes=boxes, box_labels=labels, box_mask=mask)
        images = torch.rand(1, 3, IMG_SIZE, IMG_SIZE, device=device)

        task = _yolo_task(compute_loss_fn)
        obs = task.victim_signal(model, images, target, input_shape=images.shape)

        n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        assert n_frozen > 0, "test assumes this model has a frozen parameter (DFL.conv)"
        assert len(obs.gradients) == n_trainable

    def test_fedavg_is_supported(self):
        # The criterion call needs no target at forward time, so the functional FedAvg path applies.
        assert isinstance(_yolo_task(object(), protocol=FedAvg()).protocol, FedAvg)


class TestYoloCriterionGradientFlow:
    def test_pixel_gradient_flows_but_box_gradient_does_not(self):
        """Pins both the working path (pixels) and the documented limitation
        (boxes) — see CriterionDetection for why assign()'s
        @torch.no_grad() blocks box gradients specifically."""
        device = torch.device("cpu")
        model = yolo_v8_n_basicblock().to(device).train()
        compute_loss_fn = _build_compute_loss(model)

        boxes, labels, mask = _toy_target(device)
        target = TargetInfo(boxes=boxes, box_labels=labels, box_mask=mask)
        images = torch.rand(1, 3, IMG_SIZE, IMG_SIZE, device=device)

        task = _yolo_task(compute_loss_fn)
        ctx = Context(
            model=model, observation=None, modality=None, task=task,
            input_shape=images.shape, device=device, axes=AxisManager(),
        )
        ctx.session.begin_stage(ctx)

        rep = PixelRepresentation(0.0, 1.0)
        state = State()
        rep = rep.bind(state, ctx)
        box_target = JointBoxTarget(oracle_labels=labels, k_max=2)
        box_target = box_target.bind(state, ctx)

        recon = rep.decode()
        current_target = box_target.current()
        dummy_obs = task.simulate_dummy_signal(recon, current_target, ctx)

        loss = sum(g.pow(2).sum() for g in dummy_obs.gradients)
        loss.backward()

        assert rep._param.grad is not None
        assert rep._param.grad.norm().item() > 0
        assert box_target._raw_boxes.grad is None
