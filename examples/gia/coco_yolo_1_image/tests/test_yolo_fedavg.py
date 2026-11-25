#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: AGPL-3.0-only
#
"""Task(CriterionDetection, FedAvg) with the YOLO example's detector (examples/gia/coco_yolo_1_image).

The first test pins the client update against plain SGD; the rest pin the task's behaviour
(payload shape, multi-epoch, gradient flow, model mode).
"""
from __future__ import annotations

import pytest
import torch

from leakpro.attacks.gia_attacks.modular.components.representations.pixel import PixelRepresentation
from leakpro.attacks.gia_attacks.modular.components.targets.box import JointBoxTarget
from leakpro.attacks.gia_attacks.modular.components.targets.fixed import FixedTarget
from leakpro.attacks.gia_attacks.modular.components.utils.boxes import pad_boxes
from leakpro.attacks.gia_attacks.modular.core.axes import AxisManager
from leakpro.attacks.gia_attacks.modular.core.protocol import FedAvg, FedSGD
from leakpro.attacks.gia_attacks.modular.core.state import Context, State, TargetInfo
from leakpro.attacks.gia_attacks.modular.core.task import Task
from yolo_example import load_yolo_example

pytest.importorskip("yaml")

yolo = load_yolo_example()
ComputeLoss, yolo_v8_n_basicblock = yolo.model.ComputeLoss, yolo.model.yolo_v8_n_basicblock
yolo_head_mode = yolo.attack.yolo_head_mode

IMG_SIZE = 64


DEVICE = torch.device("cpu")


def _toy_target() -> TargetInfo:
    boxes_list = [torch.tensor([[10.0, 10.0, 30.0, 30.0], [20.0, 20.0, 40.0, 40.0]], device=DEVICE)]
    labels_list = [torch.tensor([1, 2], device=DEVICE)]
    boxes, labels, mask = pad_boxes(boxes_list, labels_list, k_max=2)
    return TargetInfo(boxes=boxes, box_labels=labels, box_mask=mask)


def _task(loss_fn, *, epochs: int = 1, lr: float = 0.1, model_mode: str = "eval") -> Task:
    return Task(yolo.attack.yolo_objective(loss_fn, IMG_SIZE), FedAvg(local_epochs=epochs, lr=lr), model_mode=model_mode)


def _prepared(model: torch.nn.Module, task: Task) -> torch.nn.Module:
    task.objective.prepare_model(model, task.model_mode)
    return model


def test_one_epoch_fedavg_payload_is_minus_lr_times_the_fedsgd_gradient() -> None:
    model = yolo_v8_n_basicblock().eval()
    obj = yolo.attack.yolo_objective(ComputeLoss(model), IMG_SIZE)
    x, target = torch.rand(1, 3, IMG_SIZE, IMG_SIZE), _toy_target()
    lr = 0.1
    avg = Task(obj, FedAvg(local_epochs=1, lr=lr), model_mode="eval").victim_signal(model, x, target, input_shape=tuple(x.shape))
    sgd = Task(obj, FedSGD(), model_mode="eval").victim_signal(model, x, target, input_shape=tuple(x.shape))
    assert len(avg.gradients) == len(sgd.gradients)
    for d, g in zip(avg.gradients, sgd.gradients):
        assert torch.allclose(d, -lr * g, atol=1e-6, rtol=1e-4)


class TestVictimSignal:
    def test_victim_signal_is_a_real_delta_w_not_all_zero(self):
        """One detached delta per trainable parameter; a frozen one is not part of the payload."""
        model = yolo_v8_n_basicblock().to(DEVICE)
        task = _task(ComputeLoss(model))
        images = torch.rand(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
        obs = task.victim_signal(_prepared(model, task), images, _toy_target())

        assert len(obs.gradients) == sum(p.requires_grad for p in model.parameters())
        assert any(g.abs().sum().item() > 0 for g in obs.gradients)
        assert all(not g.requires_grad for g in obs.gradients)

    def test_multi_epoch_client_takes_more_than_one_local_step(self):
        """local_epochs > 1 must chain several SGD steps into the delta."""
        model = yolo_v8_n_basicblock().to(DEVICE)
        loss_fn = ComputeLoss(model)
        images = torch.rand(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
        target = _toy_target()
        one, three = _task(loss_fn, epochs=1, lr=0.5), _task(loss_fn, epochs=3, lr=0.5)
        obs_1 = one.victim_signal(_prepared(model, one), images, target)
        obs_3 = three.victim_signal(_prepared(model, three), images, target)

        norm_1 = sum(g.norm().item() ** 2 for g in obs_1.gradients) ** 0.5
        norm_3 = sum(g.norm().item() ** 2 for g in obs_3.gradients) ** 0.5
        assert norm_1 != pytest.approx(norm_3)


class TestGradientFlow:
    def test_pixel_gradient_flows_but_box_gradient_does_not(self):
        """ComputeLoss's no_grad assignment blocks box gradients whichever protocol calls it."""
        model = yolo_v8_n_basicblock().to(DEVICE)
        target = _toy_target()
        images = torch.rand(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)

        task = _task(ComputeLoss(model))
        ctx = Context(
            model=model, observation=None, modality=None, task=task,
            input_shape=images.shape, device=DEVICE, axes=AxisManager(),
        )
        ctx.session.begin_stage(ctx)

        rep = PixelRepresentation(0.0, 1.0)
        state = State()
        rep = rep.bind(state, ctx)
        box_target = JointBoxTarget(oracle_labels=target.box_labels, k_max=2)
        box_target = box_target.bind(state, ctx)

        dummy_obs = task.simulate_dummy_signal(rep.decode(), box_target.current(), ctx)
        sum(g.pow(2).sum() for g in dummy_obs.gradients).backward()

        assert rep._param.grad is not None
        assert rep._param.grad.norm().item() > 0
        assert box_target._raw_boxes.grad is None


class TestModelMode:
    """main.py runs model.eval() + model.head.simulate_train_on_eval = True. The preset's task
    reproduces that on the shadow model it deep-copies, whatever mode the caller built the model in."""

    def test_preset_default_model_mode_is_eval(self):
        attack = yolo.attack.yolo_attack(object(), IMG_SIZE, torch.zeros((1, 1), dtype=torch.long), 1)
        assert attack.task.model_mode == "eval"
        assert isinstance(attack.task.protocol, FedAvg)

    def test_preset_recovers_the_image_only(self):
        """ComputeLoss gives the boxes no gradient, so the preset holds them at the target passed to run_attack."""
        attack = yolo.attack.yolo_attack(object(), IMG_SIZE, torch.zeros((1, 1), dtype=torch.long), 1)
        assert isinstance(attack.stages[0].target_source, FixedTarget)

    def test_begin_stage_puts_execution_model_in_eval_with_simulate_train_on_eval(self):
        model = yolo_v8_n_basicblock().to(DEVICE).train()  # deliberately wrong starting mode
        task = _task(ComputeLoss(model))
        ctx = Context(
            model=model, observation=None, modality=None, task=task,
            input_shape=(1, 3, IMG_SIZE, IMG_SIZE), device=DEVICE, axes=AxisManager(),
        )
        ctx.session.begin_stage(ctx)

        shadow = ctx.session.execution_model
        assert not shadow.training
        assert shadow.head.simulate_train_on_eval is True

    def test_prepare_model_puts_a_victim_in_eval_with_simulate_train_on_eval(self):
        model = yolo_v8_n_basicblock().to(DEVICE).train()
        _prepared(model, _task(ComputeLoss(model)))
        assert not model.training
        assert model.head.simulate_train_on_eval is True

    def test_train_mode_is_still_available_on_request(self):
        model = yolo_v8_n_basicblock().to(DEVICE).eval()
        task = _task(ComputeLoss(model), model_mode="train")
        ctx = Context(
            model=model, observation=None, modality=None, task=task,
            input_shape=(1, 3, IMG_SIZE, IMG_SIZE), device=DEVICE, axes=AxisManager(),
        )
        ctx.session.begin_stage(ctx)
        assert ctx.session.execution_model.training
