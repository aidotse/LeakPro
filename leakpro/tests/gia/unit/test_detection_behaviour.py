#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Behaviour of every detection objective: boxes receive gradient, loss is minimal at the ground truth, both protocols run."""
from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from leakpro.attacks.gia_attacks.modular.components.objectives import (
    CriterionDetection, HuggingFaceDetection, TorchvisionDetection,
)
from leakpro.attacks.gia_attacks.modular.core.protocol import FedAvg, FedSGD
from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo
from leakpro.attacks.gia_attacks.modular.core.task import Task
from leakpro.tests.gia.contract import registry_detection as reg

pytest.importorskip("torchvision")
pytest.importorskip("transformers")

# Tiny DETR runs in the default tier; RetinaNet (ResNet-50 FPN) is slow on CPU.
FAMILIES = [
    pytest.param("detr", id="detr"),
    pytest.param("retinanet", id="retinanet", marks=pytest.mark.gia_slow),
]


def _objective_and_model(family: str):
    if family == "detr":
        return HuggingFaceDetection(img_size=reg.DET_SIZE), copy.deepcopy(reg._detr()).train()
    return TorchvisionDetection(), copy.deepcopy(reg._retinanet()).train()


def _with_boxes(t: TargetInfo, boxes: torch.Tensor) -> TargetInfo:
    return TargetInfo(boxes=boxes, box_labels=t.box_labels, box_mask=t.box_mask)


@pytest.mark.parametrize("family", FAMILIES)
def test_box_gradient_flows_to_the_box_target(family: str) -> None:
    objective, model = _objective_and_model(family)
    t = reg._box_target()
    boxes = t.boxes.clone().requires_grad_(True)
    objective.loss(model, reg._det_input(), _with_boxes(t, boxes)).backward()
    assert boxes.grad is not None and boxes.grad.abs().sum() > 0


@pytest.mark.parametrize("family", FAMILIES)
def test_loss_is_lower_at_the_ground_truth_than_at_displaced_boxes(family: str) -> None:
    objective, model = _objective_and_model(family)
    x, t = reg._det_input(), reg._box_target()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(60 if family == "detr" else 25):
        opt.zero_grad()
        objective.loss(model, x, t).backward()
        opt.step()
    with torch.no_grad():
        at_gt = objective.loss(model, x, t)
        displaced = objective.loss(model, x, _with_boxes(t, t.boxes + 25.0))
    assert at_gt < displaced


def _echo_task() -> tuple[Task, nn.Module, torch.Tensor, TargetInfo]:
    """A 'detector' that outputs exactly the encoded ground truth: the criterion must then be zero."""
    t = reg._box_target()
    encoded = reg._box_rows(t)

    class Echo(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.ones(1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return encoded * self.w

    objective = CriterionDetection(reg._criterion, reg._box_rows, sum_output=True)
    return Task(objective), Echo(), reg._det_input(), t


def test_criterion_loss_is_zero_when_the_model_outputs_the_ground_truth() -> None:
    task, model, x, t = _echo_task()
    assert float(task.objective.loss(model, x, t)) == pytest.approx(0.0, abs=1e-9)


def test_criterion_loss_grows_with_the_distance_from_the_ground_truth() -> None:
    task, model, x, t = _echo_task()
    near = float(task.objective.loss(model, x, _with_boxes(t, t.boxes + 1.0)))
    far = float(task.objective.loss(model, x, _with_boxes(t, t.boxes + 10.0)))
    assert 0.0 < near < far


@pytest.mark.parametrize("family", FAMILIES)
def test_fedavg_and_fedsgd_payloads_cover_the_trainable_parameters(family: str) -> None:
    objective, model = _objective_and_model(family)
    n_trainable = sum(p.requires_grad for p in model.parameters())
    x, t = reg._det_input(), reg._box_target()
    for protocol in (FedSGD(), FedAvg(lr=0.01)):
        obs = Task(objective, protocol).victim_signal(model, x, t, input_shape=tuple(x.shape))
        assert len(obs.gradients) == n_trainable
        assert any(g.abs().sum() > 0 for g in obs.gradients)


@pytest.mark.parametrize("family", FAMILIES)
def test_fedavg_dummy_signal_is_differentiable_to_pixels_and_leaves_the_model_alone(family: str) -> None:
    from leakpro.attacks.gia_attacks.modular.components.modalities.detection import DetectionImageModality
    from leakpro.tests.gia.contract.conftest import make_ctx

    objective, model = _objective_and_model(family)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    task = Task(objective, FedAvg(lr=0.01))
    x, t = reg._det_input(), reg._box_target()
    obs = task.victim_signal(model, x, t, input_shape=tuple(x.shape))
    x = x.clone().requires_grad_(True)
    ctx = make_ctx(model=model, task=task, modality=DetectionImageModality(), input_shape=tuple(x.shape), observation=obs)
    ctx.session.begin_stage(ctx)
    obs = task.simulate_dummy_signal(x, t, ctx)
    g = torch.autograd.grad(sum(d.pow(2).sum() for d in obs.gradients), x)[0]
    assert g.abs().sum() > 0
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
