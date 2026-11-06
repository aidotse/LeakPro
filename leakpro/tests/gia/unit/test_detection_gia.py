#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""End-to-end wiring for the detection GIA preset: JointBoxTarget, BoxEvaluator,
CompositeEvaluator, and detection() run through run_attack().

No existing example covers a full detection attack, so this is the pipeline
smoke test for it — real RetinaNet forward/backward through the actual
run_attack() path, not mocked components.
"""
from __future__ import annotations

import math

import pytest
import torch

torchvision = pytest.importorskip("torchvision")
from torchvision.models.detection import retinanet_resnet50_fpn  # noqa: E402

from leakpro.attacks.gia_attacks.modular.components.evaluators.box import BoxEvaluator  # noqa: E402
from leakpro.attacks.gia_attacks.modular.components.objectives import TorchvisionDetection  # noqa: E402
from leakpro.attacks.gia_attacks.modular.components.targets.box import JointBoxTarget  # noqa: E402
from leakpro.attacks.gia_attacks.modular.components.utils.boxes import pad_boxes  # noqa: E402
from leakpro.attacks.gia_attacks.modular.core.attack import run_attack  # noqa: E402
from leakpro.attacks.gia_attacks.modular.core.axes import AxisManager  # noqa: E402
from leakpro.attacks.gia_attacks.modular.core.protocol import FedSGD  # noqa: E402
from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo  # noqa: E402
from leakpro.attacks.gia_attacks.modular.core.task import Task  # noqa: E402
from leakpro.tests.gia.helpers import LossRecorder  # noqa: E402
from leakpro.attacks.gia_attacks.modular.presets import detection  # noqa: E402

IMG_SIZE = 64


def _tiny_detector() -> torch.nn.Module:
    model = retinanet_resnet50_fpn(weights=None, weights_backbone=None, min_size=IMG_SIZE, max_size=IMG_SIZE)
    model.train()
    return model


def _toy_batch(k_max: int = 2):
    images = torch.rand(2, 3, IMG_SIZE, IMG_SIZE)
    boxes_list = [
        torch.tensor([[10.0, 10.0, 30.0, 30.0]]),
        torch.tensor([[5.0, 5.0, 20.0, 20.0], [30.0, 30.0, 50.0, 50.0]]),
    ]
    labels_list = [torch.tensor([1]), torch.tensor([2, 1])]
    boxes, labels, mask = pad_boxes(boxes_list, labels_list, k_max=k_max)
    return images, boxes, labels, mask


class TestJointBoxTarget:
    def _bind(self, k_max=2, n=2):
        import types
        labels = torch.zeros((n, k_max), dtype=torch.long)
        target = JointBoxTarget(oracle_labels=labels, k_max=k_max)
        ctx = types.SimpleNamespace(
            input_shape=(n, 3, IMG_SIZE, IMG_SIZE),
            axes=AxisManager(),
            device=torch.device("cpu"),
        )
        from leakpro.attacks.gia_attacks.modular.core.state import State
        target = target.bind(State(), ctx)
        return target

    def test_decoded_boxes_are_never_degenerate(self):
        """Sigmoid parameterization must guarantee x2 > x1 and y2 > y1."""
        target = self._bind()
        current = target.current()
        boxes = current.boxes
        assert (boxes[..., 2] > boxes[..., 0]).all()
        assert (boxes[..., 3] > boxes[..., 1]).all()

    def test_decoded_boxes_stay_within_image_bounds(self):
        target = self._bind()
        boxes = target.current().boxes
        assert (boxes >= 0).all()
        assert (boxes[..., 0] <= IMG_SIZE).all() and (boxes[..., 2] <= IMG_SIZE).all()
        assert (boxes[..., 1] <= IMG_SIZE).all() and (boxes[..., 3] <= IMG_SIZE).all()

    def test_box_mask_is_all_true(self):
        """k_max is a fixed capacity assumption — every slot counts as real."""
        target = self._bind(k_max=3)
        assert target.current().box_mask.all()

    def test_labels_pass_through_unchanged_oracle(self):
        labels = torch.tensor([[1, 2], [3, 4]])
        import types
        t = JointBoxTarget(oracle_labels=labels, k_max=2)
        ctx = types.SimpleNamespace(input_shape=(2, 3, IMG_SIZE, IMG_SIZE), axes=AxisManager(), device=torch.device("cpu"))
        from leakpro.attacks.gia_attacks.modular.core.state import State
        t = t.bind(State(), ctx)
        assert torch.equal(t.current().box_labels, labels)

    def test_only_box_coords_are_trainable_parameters(self):
        target = self._bind()
        groups = target.parameters()
        assert len(groups) == 1
        assert groups[0]["params"][0].requires_grad

    def test_resume_from_state_target_reuses_raw_boxes(self):
        target = self._bind()
        first = target.current()

        import types
        from leakpro.attacks.gia_attacks.modular.core.state import State
        state2 = State(target=first.detach().clone())
        ctx2 = types.SimpleNamespace(input_shape=(2, 3, IMG_SIZE, IMG_SIZE), axes=AxisManager(), device=torch.device("cpu"))
        target2 = JointBoxTarget(oracle_labels=target.oracle_labels, k_max=2)
        target2 = target2.bind(state2, ctx2)

        assert torch.allclose(target2.current().boxes, first.boxes)

    def _bind_with(self, axes, n=2, k_max=2):
        import types

        from leakpro.attacks.gia_attacks.modular.core.state import State
        labels = torch.arange(n * k_max).reshape(n, k_max)
        ctx = types.SimpleNamespace(input_shape=(n, 3, IMG_SIZE, IMG_SIZE), axes=axes, device=torch.device("cpu"))
        return JointBoxTarget(oracle_labels=labels, k_max=k_max).bind(State(), ctx)

    def test_seed_axis_prefixes_every_field(self):
        from leakpro.attacks.gia_attacks.modular.core.axes import Axis
        cur = self._bind_with(AxisManager(Axis("seed", 3))).current()
        assert cur.axis_ndim == 1
        assert tuple(cur.boxes.shape) == (3, 2, 2, 4)
        assert tuple(cur.box_labels.shape) == (3, 2, 2) and tuple(cur.box_mask.shape) == (3, 2, 2)
        assert tuple(cur.raw_boxes.shape) == (3, 2, 2, 4)

    def test_each_seed_slice_is_a_valid_single_batch_target(self):
        from leakpro.attacks.gia_attacks.modular.core.axes import Axis
        src = self._bind_with(AxisManager(Axis("epoch", 2)))
        one = src.slice(src.current(), 0, 1)
        assert one.axis_ndim == 0 and tuple(one.boxes.shape) == (2, 2, 4)
        assert torch.equal(one.box_labels, src.oracle_labels)
        cur = src.current()
        for i in range(2):
            sl = src.slice(cur, 0, i)
            assert torch.allclose(sl.boxes, src._decode_boxes(cur.raw_boxes[i]))
        assert not torch.allclose(src.slice(cur, 0, 0).boxes, src.slice(cur, 0, 1).boxes)

    def test_carried_over_raw_boxes_expand_or_raise_for_any_prefix(self):
        import types

        from leakpro.attacks.gia_attacks.modular.core.axes import Axis
        from leakpro.attacks.gia_attacks.modular.core.state import State
        labels = torch.zeros((2, 2), dtype=torch.long)

        def bind(raw, axes):
            ctx = types.SimpleNamespace(input_shape=(2, 3, IMG_SIZE, IMG_SIZE), axes=axes, device=torch.device("cpu"))
            state = State(target=TargetInfo(raw_boxes=raw))
            return JointBoxTarget(oracle_labels=labels, k_max=2).bind(state, ctx)

        raw = torch.randn(2, 2, 4)
        no_prefix = bind(raw, AxisManager())
        assert tuple(no_prefix._raw_boxes.shape) == (2, 2, 4)
        two_axes = bind(raw, AxisManager(Axis("seed", 3), Axis("epoch", 2)))
        assert tuple(two_axes._raw_boxes.shape) == (3, 2, 2, 2, 4)
        assert torch.allclose(two_axes._raw_boxes[1, 0], raw)
        same = bind(torch.randn(3, 2, 2, 4), AxisManager(Axis("seed", 3)))
        assert tuple(same._raw_boxes.shape) == (3, 2, 2, 4)
        with pytest.raises(ValueError, match="axis prefix"):
            bind(torch.randn(3, 2, 2, 4), AxisManager(Axis("seed", 3), Axis("epoch", 2)))
        with pytest.raises(ValueError, match="axis prefix"):
            bind(torch.randn(3, 2, 2, 4), AxisManager())

    def test_reduce_mean_averages_boxes_and_keeps_oracle_labels(self):
        from leakpro.attacks.gia_attacks.modular.core.axes import Axis
        src = self._bind_with(AxisManager(Axis("seed", 3)))
        cur = src.current()
        red = src.reduce(cur, 0, "mean")
        assert red.axis_ndim == 0 and tuple(red.boxes.shape) == (2, 2, 4)
        assert torch.allclose(red.boxes, cur.boxes.detach().mean(0), atol=1e-5)
        assert torch.equal(red.box_labels, src.oracle_labels) and red.box_mask.all()

    def test_reduce_rejects_an_unknown_method(self):
        from leakpro.attacks.gia_attacks.modular.core.axes import Axis
        src = self._bind_with(AxisManager(Axis("seed", 2)))
        with pytest.raises(ValueError, match="method"):
            src.reduce(src.current(), 0, "median")

    def test_labels_of_the_wrong_shape_fail_at_bind(self):
        import types

        from leakpro.attacks.gia_attacks.modular.core.state import State
        ctx = types.SimpleNamespace(input_shape=(2, 3, IMG_SIZE, IMG_SIZE), axes=AxisManager(), device=torch.device("cpu"))
        with pytest.raises(ValueError, match="oracle_labels"):
            JointBoxTarget(oracle_labels=torch.zeros((3, 2), dtype=torch.long), k_max=2).bind(State(), ctx)


class TestBoxEvaluator:
    def test_target_with_an_axis_prefix_raises_asking_to_aggregate(self):
        from leakpro.attacks.gia_attacks.modular.core.state import State

        images, boxes, labels, mask = _toy_batch()
        seeded = boxes.expand(3, *boxes.shape)
        state = State(input=images, target=TargetInfo(boxes=seeded, box_labels=labels.expand(3, *labels.shape),
                                                     box_mask=mask.expand(3, *mask.shape), axis_ndim=1))
        with pytest.raises(ValueError, match="aggregate"):
            BoxEvaluator(ground_truth={"boxes": boxes, "labels": labels, "mask": mask}).evaluate(images, images, state=state)

    def test_perfect_reconstruction_scores_full_marks(self):
        from leakpro.attacks.gia_attacks.modular.core.state import State

        images, boxes, labels, mask = _toy_batch()
        state = State(input=images, target=TargetInfo(boxes=boxes, box_labels=labels, box_mask=mask))
        state.artifacts[BoxEvaluator.GT_KEY] = {"boxes": boxes, "labels": labels, "mask": mask}

        result = BoxEvaluator(iou_threshold=0.5).evaluate(images, images, state=state)

        assert result.metrics["precision_at_iou"] == pytest.approx(1.0)
        assert result.metrics["recall_at_iou"] == pytest.approx(1.0)
        assert result.metrics["mean_matched_iou"] == pytest.approx(1.0)
        assert result.metrics["label_accuracy_on_matches"] == pytest.approx(1.0)

    def test_missing_ground_truth_returns_empty_result(self):
        from leakpro.attacks.gia_attacks.modular.core.state import State
        images, boxes, labels, mask = _toy_batch()
        state = State(input=images, target=TargetInfo(boxes=boxes, box_labels=labels, box_mask=mask))
        result = BoxEvaluator().evaluate(images, images, state=state)
        assert result.metrics == {}

    def test_wrong_labels_on_correctly_placed_boxes_lowers_label_accuracy_only(self):
        images, boxes, labels, mask = _toy_batch()
        wrong_labels = labels.clone()
        wrong_labels[wrong_labels > 0] += 1  # perturb only real (non-padding) labels
        from leakpro.attacks.gia_attacks.modular.core.state import State
        state = State(input=images, target=TargetInfo(boxes=boxes, box_labels=wrong_labels, box_mask=mask))
        state.artifacts[BoxEvaluator.GT_KEY] = {"boxes": boxes, "labels": labels, "mask": mask}

        result = BoxEvaluator(iou_threshold=0.5).evaluate(images, images, state=state)

        assert result.metrics["mean_matched_iou"] == pytest.approx(1.0)
        assert result.metrics["label_accuracy_on_matches"] == pytest.approx(0.0)


class TestDetectionGiaEndToEnd:
    def test_run_attack_produces_scored_reconstruction(self):
        """The full preset through run_attack(): real gradients in, scored recon out."""
        torch.manual_seed(0)
        model = _tiny_detector()
        images, boxes, labels, mask = _toy_batch()
        victim_target = TargetInfo(boxes=boxes, box_labels=labels, box_mask=mask)

        task = Task(TorchvisionDetection())
        obs = task.victim_signal(model, images, victim_target, input_shape=images.shape)

        atk = detection(oracle_labels=labels, k_max=2, max_iter=2, lr=0.05)
        state, ctx = run_attack(atk, model=model, observation=obs, input_shape=images.shape, device=torch.device("cpu"))

        assert state.input.shape == images.shape
        assert state.target.boxes.shape == boxes.shape

        state.artifacts[BoxEvaluator.GT_KEY] = {"boxes": boxes, "labels": labels, "mask": mask}
        result = ctx.evaluator.evaluate(state.input, images, ctx=ctx, state=state)

        for key in ("image.psnr", "image.ssim", "boxes.precision_at_iou", "boxes.mean_matched_iou"):
            assert key in result.metrics
            assert isinstance(result.metrics[key], float)

    def test_two_stage_detector_rejected_at_task_construction_is_not_this_presets_job(self):
        """detection() itself doesn't validate the model architecture — the
        RoIAlign double-backward failure surfaces from simulate_dummy_signal, not
        from preset construction. Documented on TorchvisionDetection; pinned there."""
        atk = detection(oracle_labels=torch.zeros((1, 1), dtype=torch.long), k_max=1)
        assert isinstance(atk.task.protocol, FedSGD)


import copy  # noqa: E402

from leakpro.attacks.gia_attacks.modular.components.objectives import HuggingFaceDetection  # noqa: E402
from leakpro.attacks.gia_attacks.modular.core.protocol import FedAvg  # noqa: E402
from leakpro.tests.gia.contract import registry_detection as reg  # noqa: E402


class TestDetectionPreset:
    def _victim(self, protocol):
        model = copy.deepcopy(reg._detr()).train()
        x, t = reg._det_input(), reg._box_target()
        objective = HuggingFaceDetection(img_size=reg.DET_SIZE)
        obs = Task(objective, protocol).victim_signal(model, x, t, input_shape=tuple(x.shape))
        return model, x, t, objective, obs

    def test_unknown_loss_type_raises(self):
        with pytest.raises(ValueError, match="loss_type"):
            detection(oracle_labels=torch.zeros((1, 1), dtype=torch.long), k_max=1, loss_type="l1")

    def test_defaults_are_torchvision_fedsgd(self):
        atk = detection(oracle_labels=torch.zeros((1, 1), dtype=torch.long), k_max=1)
        assert isinstance(atk.task.objective, TorchvisionDetection) and isinstance(atk.task.protocol, FedSGD)

    def test_detr_attack_loss_decreases(self):
        model, x, t, objective, obs = self._victim(FedSGD())
        torch.manual_seed(0)
        rec = LossRecorder()
        atk = detection(t.box_labels, 1, objective=objective, max_iter=40, lr=0.05)
        run_attack(atk, model=model, observation=obs, input_shape=tuple(x.shape), device=torch.device("cpu"), callbacks=[rec])
        assert len(rec.losses) >= 10
        # DETR's Hungarian matching makes the per-step loss jump between assignments, so the last steps are not
        # reliably below the first; the optimisation works if some iterate is clearly better than the start
        # (observed best/start ratios 0.18-0.50 across torch 2.5 and 2.14 and several victims).
        assert min(rec.losses) < 0.75 * rec.losses[0]

    def test_detr_fedavg_attack_runs_and_scores(self):
        model, x, t, objective, obs = self._victim(FedAvg(lr=0.01))
        atk = detection(t.box_labels, 1, objective=objective, protocol=FedAvg(lr=0.01), max_iter=3)
        state, ctx = run_attack(atk, model=model, observation=obs, input_shape=tuple(x.shape), device=torch.device("cpu"))
        ctx.evaluator.evaluators["boxes"].ground_truth = t
        res = ctx.evaluator.evaluate(state.input, x, ctx=ctx, state=state)
        assert "image.psnr" in res.metrics
        for key in ("boxes.f1_at_iou", "boxes.mean_matched_iou"):
            assert key in res.metrics and math.isfinite(res.metrics[key])

    def test_image_only_recovery_keeps_the_given_boxes(self):
        model, x, t, objective, obs = self._victim(FedSGD())
        atk = detection(t.box_labels, 1, objective=objective, recover="image", max_iter=2)
        state, _ = run_attack(atk, model=model, observation=obs, input_shape=tuple(x.shape),
                              device=torch.device("cpu"), target=t)
        assert torch.equal(state.target.boxes, t.boxes)

    def test_two_seeds_give_an_averaged_single_batch_target(self):
        model, x, t, objective, obs = self._victim(FedSGD())
        atk = detection(t.box_labels, 1, objective=objective, num_seeds=2, max_iter=2)
        state, _ = run_attack(atk, model=model, observation=obs, input_shape=tuple(x.shape), device=torch.device("cpu"))
        assert tuple(state.input.shape) == tuple(x.shape) and tuple(state.target.boxes.shape) == tuple(t.boxes.shape)


def test_detection_threat_model_grants_what_inferred_needs() -> None:
    attack = detection(torch.zeros((1, 1), dtype=torch.long), 1, bn_weight=0.1)
    assert attack.threat_model.has_bn_statistics and attack.threat_model.has_local_hyperparameters
    running = detection(torch.zeros((1, 1), dtype=torch.long), 1, bn_weight=0.1, bn_strategy="running")
    assert running.threat_model.has_bn_statistics and running.threat_model.has_local_hyperparameters is False
