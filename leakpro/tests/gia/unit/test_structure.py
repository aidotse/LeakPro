#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Structural tests for the core/component boundary and rank-based signals.

These pin architectural decisions rather than numerical behaviour: what core is
allowed to know about components, and which signals the framework is allowed to
infer from tensor rank.
"""

import pytest
import torch
import torch.nn as nn


from leakpro.attacks.gia_attacks.modular.components.drivers import GradientDriver
from leakpro.attacks.gia_attacks.modular.core.result import DriverResult
from leakpro.attacks.gia_attacks.modular.components.losses import CosineGradientLoss
from leakpro.attacks.gia_attacks.modular.components.modalities.image import (
    ImageNormalizationModality,
)
from leakpro.attacks.gia_attacks.modular.components.representations.pixel import (
    PixelRepresentation,
)
from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget
from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
from leakpro.attacks.gia_attacks.modular.core.task import Task
from leakpro.attacks.gia_attacks.modular.core import interfaces
from leakpro.attacks.gia_attacks.modular.core.attack import Attack, run_attack
from leakpro.attacks.gia_attacks.modular.core.axes import Axis, AxisManager
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.attacks.gia_attacks.modular.core.state import (
    ClientObservation,
    Context,
    State,
    TargetInfo,
)


# ── 1.4: the parallel axis is configurable ────────────────────────────────────

def _tiny_ctx(axes: AxisManager) -> tuple[Context, nn.Module]:
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 4)).eval()
    task = Task(CrossEntropy())
    imgs = torch.rand(1, 3, 4, 4)
    obs_raw = task.victim_signal(model, imgs, TargetInfo(value=torch.tensor([0])))
    obs = ClientObservation(gradients=[g.detach().clone() for g in obs_raw.gradients])
    ctx = Context(
        model=model,
        observation=obs,
        modality=ImageNormalizationModality([0.5] * 3, [0.5] * 3, torch.device("cpu")),
        task=task,
        input_shape=(1, 3, 4, 4),
        device=torch.device("cpu"),
        axes=axes,
    )
    return ctx, model


class TestGradientDriverParallelAxis:
    def test_defaults_to_seed(self):
        assert GradientDriver().parallel_axis == "seed"

    def test_iterates_a_custom_axis(self):
        """A driver whose parallel axis is 'restart' must produce a
        restart-prefixed output, not collapse to a single pass."""
        axes = AxisManager(Axis("restart", 3))
        ctx, _ = _tiny_ctx(axes)
        stage = OptimStage(
            losses={"grad": (CosineGradientLoss(), 1.0)},
            rep=PixelRepresentation(min_val=0.0, max_val=1.0),
            driver=GradientDriver(lr=1e-2, max_iter=2, return_best=True,
                                  schedule=None, parallel_axis="restart"),
            target_source=FixedTarget(),
        )
        state = State(target=TargetInfo(value=torch.tensor([0])))
        out = stage.run(state, ctx)
        assert out.input.shape == (3, 1, 3, 4, 4)

    def test_missing_axis_degenerates_to_single_pass(self):
        ctx, _ = _tiny_ctx(AxisManager())
        stage = OptimStage(
            losses={"grad": (CosineGradientLoss(), 1.0)},
            rep=PixelRepresentation(min_val=0.0, max_val=1.0),
            driver=GradientDriver(lr=1e-2, max_iter=2, return_best=True,
                                  schedule=None, parallel_axis="restart"),
            target_source=FixedTarget(),
        )
        out = stage.run(State(target=TargetInfo(value=torch.tensor([0]))), ctx)
        assert out.input.shape == (1, 3, 4, 4)

    def test_slice_axis_is_reported_to_losses(self):
        """A loss that cares which axis it is being sliced along can read it
        instead of assuming 'seed'."""
        seen: list[str | None] = []

        class RecordingLoss(CosineGradientLoss):
            def compute(self, inputs):
                seen.append(inputs.slice_axis)
                return super().compute(inputs)

        ctx, _ = _tiny_ctx(AxisManager(Axis("restart", 2)))
        stage = OptimStage(
            losses={"grad": (RecordingLoss(), 1.0)},
            rep=PixelRepresentation(min_val=0.0, max_val=1.0),
            driver=GradientDriver(lr=1e-2, max_iter=1, return_best=False,
                                  schedule=None, parallel_axis="restart"),
            target_source=FixedTarget(),
        )
        stage.run(State(target=TargetInfo(value=torch.tensor([0]))), ctx)
        assert set(seen) == {"restart"}


class TestAggregatorAxis:
    def test_aggregators_accept_a_custom_axis(self):
        from leakpro.attacks.gia_attacks.modular.components.aggregators import (
            BestLossAggregator,
            MeanAggregator,
        )
        assert MeanAggregator().axis == "seed"
        assert BestLossAggregator(axis="restart").axis == "restart"

    def test_mean_aggregator_collapses_the_named_axis(self):
        from leakpro.attacks.gia_attacks.modular.components.aggregators import MeanAggregator

        ctx, _ = _tiny_ctx(AxisManager(Axis("restart", 4)))
        stage = OptimStage(
            losses={},
            rep=PixelRepresentation(min_val=0.0, max_val=1.0),
            driver=GradientDriver(max_iter=0),
            target_source=FixedTarget(),
        )
        recon = torch.rand(4, 1, 3, 4, 4)
        result = DriverResult(input=recon, target=TargetInfo(value=torch.tensor([0])), artifacts={}, loss=0.0,
                              iterations=0, slice_losses=None)
        out = MeanAggregator(axis="restart").aggregate(result, ctx, stage).input
        assert out.shape == (1, 3, 4, 4)
        assert torch.allclose(out, recon.mean(dim=0))


# ── 1.4: DimitrovInference activation types ───────────────────────────────────

class TestDimitrovActivationTypes:
    def test_defaults_cover_common_activation_families(self):
        from leakpro.attacks.gia_attacks.modular.components.target_inference import (
            DimitrovInference,
        )
        assert nn.ReLU in DimitrovInference.DEFAULT_ACTIVATION_TYPES
        assert nn.GELU in DimitrovInference.DEFAULT_ACTIVATION_TYPES

    def test_strict_mode_rejects_a_model_with_no_known_activation(self):
        """Without this, a GELU/SiLU network silently fell back to a different
        quantity (|output|.sum) and produced quietly wrong label counts."""
        from leakpro.attacks.gia_attacks.modular.components.target_inference import (
            DimitrovInference,
        )

        inference = DimitrovInference(
            epochs=1, batch_size=1, activation_types=(nn.ReLU,), strict_activation=True
        )
        model = nn.Sequential(nn.Flatten(), nn.Tanh(), nn.Linear(3 * 4 * 4, 4))
        ctx, _ = _tiny_ctx(AxisManager())
        ctx = Context(
            model=model, observation=ctx.observation, modality=ctx.modality,
            task=ctx.task, input_shape=(1, 3, 4, 4), device=torch.device("cpu"),
        )
        with pytest.raises(ValueError, match="no activation of type"):
            inference.infer(ctx)


# ── Evaluator wiring ──────────────────────────────────────────────────────────

class TestEvaluatorWiring:
    def test_modality_no_longer_scores(self):
        """Transform and scoring vary independently; Modality is only a transform."""
        assert not hasattr(interfaces.Modality, "compute_metrics")

    def test_run_attack_threads_the_evaluator_onto_ctx(self):
        from leakpro.attacks.gia_attacks.modular.components.evaluators import ImageEvaluator

        # 16x16 rather than the 4x4 used elsewhere: SSIM's default gaussian
        # kernel is 11x11 and cannot be applied to a smaller image.
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, 4)).eval()
        task = Task(CrossEntropy())
        imgs = torch.rand(1, 3, 16, 16)
        obs_raw = task.victim_signal(model, imgs, TargetInfo(value=torch.tensor([0])))
        obs = ClientObservation(gradients=[g.detach().clone() for g in obs_raw.gradients])

        atk = Attack(
            modality=ImageNormalizationModality([0.5] * 3, [0.5] * 3, torch.device("cpu")),
            task=task,
            evaluator=ImageEvaluator(),
            stages=[OptimStage(
                losses={"grad": (CosineGradientLoss(), 1.0)},
                rep=PixelRepresentation(min_val=0.0, max_val=1.0),
                driver=GradientDriver(lr=1e-2, max_iter=1, schedule=None),
                target_source=FixedTarget(),
            )],
        )
        state, ctx = run_attack(
            atk, model, obs, (1, 3, 16, 16), torch.device("cpu"),
            target=TargetInfo(value=torch.tensor([0])),
        )
        assert isinstance(ctx.evaluator, ImageEvaluator)

        result = ctx.evaluator.evaluate(state.input, imgs, ctx=ctx, state=state)
        # metrics stay scalar-only; the permutation is an extra, not a fake float.
        assert set(result.metrics) == {"psnr", "ssim"}
        assert all(isinstance(v, float) for v in result.metrics.values())
        assert "permutation" in result.extras
        # Legacy flat-dict access still works for callers that used the old API.
        assert result["psnr"] == result.metrics["psnr"]
        assert result["permutation"] == result.extras["permutation"]
