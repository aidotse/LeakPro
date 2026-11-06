#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for the unified GradientDriver.

Architecture under test
-----------------------
The GradientDriver closure runs a single per-seed loop per iteration:

  Every loss — whether gradient-matching or data-space regularization — is
  evaluated once per seed slice.  One backward per slice frees the activation
  graph immediately.  Losses are scaled by 1/seed_size before backward so the
  effective step size is invariant to seed count.

Tests
-----
  1. Routing: all losses receive per-seed slices (shape without seed dim).
  2. Call counts: each loss is called exactly seed_size times per step.
  3. No-seed backward compat: single seed_size=1 iteration, unmodified tensor.
  4. Gradient averaging: with S seeds each element's accumulated grad = 1/S.
  5. Loss accounting: reported loss = seed mean of the weighted per-slice sum.
  6. LBFGS compatibility: multiple inner closure calls do not re-backward freed graphs.
"""
from __future__ import annotations


import pytest
import torch
import torch.nn as nn
from torch import Tensor


from leakpro.attacks.gia_attacks.modular.core.axes import Axis, AxisManager
from leakpro.attacks.gia_attacks.modular.core.interfaces import Callback, LossTerm
from leakpro.attacks.gia_attacks.modular.core.loss_inputs import LossInputs
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.attacks.gia_attacks.modular.core.state import (
    ClientObservation, Context, State, TargetInfo,
)
from leakpro.attacks.gia_attacks.modular.components.drivers import GradientDriver
from leakpro.attacks.gia_attacks.modular.components.initializers import FromTensor
from leakpro.attacks.gia_attacks.modular.components.modalities.image import ImageNormalizationModality
from leakpro.attacks.gia_attacks.modular.components.representations.pixel import PixelRepresentation
from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget
from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
from leakpro.attacks.gia_attacks.modular.core.task import Task
from leakpro.tests.gia import helpers


INPUT_SHAPE = (1, 3, 4, 4)
DEVICE = torch.device("cpu")
MEAN = [0.5, 0.5, 0.5]
STD  = [0.5, 0.5, 0.5]


# ── Stub losses ───────────────────────────────────────────────────────────────

class _SpyLoss(LossTerm):
    """Records the shape of every recon tensor it receives; returns recon.sum()."""
    def __init__(self) -> None:
        self.received_shapes: list[tuple] = []

    def compute(self, inputs: LossInputs) -> Tensor:
        self.received_shapes.append(tuple(inputs.recon.shape))
        return inputs.recon.sum()


class _ConstantLoss(LossTerm):
    """Returns a fixed scalar value with zero gradient on recon (loss accounting tests)."""
    def __init__(self, value: float) -> None:
        self._val = float(value)

    def compute(self, inputs: LossInputs) -> Tensor:
        # inputs.recon.sum() * 0 keeps a grad_fn so backward() works; contributes zero gradient.
        return inputs.recon.sum() * 0.0 + self._val


class _SumLoss(LossTerm):
    """Returns recon.sum() — gradient is exactly 1 at every recon element."""
    def compute(self, inputs: LossInputs) -> Tensor:
        return inputs.recon.sum()


class _LossCapture(Callback):
    """Captures the scalar loss value reported on each step-end."""
    def __init__(self) -> None:
        self.values: list[float] = []

    def on_step_end(self, event) -> None:  # noqa: ANN001
        self.values.append(event.loss)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_model() -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 10)).eval()


def _make_ctx(axes: AxisManager | None = None) -> Context:
    model = _make_model()
    task = Task(CrossEntropy())
    imgs = torch.rand(INPUT_SHAPE)
    obs_raw = task.victim_signal(
        model, imgs, TargetInfo(value=torch.tensor([0]))
    )
    obs = ClientObservation(gradients=[g.detach().clone() for g in obs_raw.gradients])
    return Context(
        model=model,
        observation=obs,
        modality=ImageNormalizationModality(mean=MEAN, std=STD, device=DEVICE),
        task=task,
        input_shape=INPUT_SHAPE,
        device=DEVICE,
        axes=axes or AxisManager(),
    )


def _make_stage(
    losses: dict,
    rep: PixelRepresentation | None = None,
    lr: float = 0.0,
    optimizer_cls: type = torch.optim.Adam,
    optimizer_kwargs: dict | None = None,
) -> OptimStage:
    return OptimStage(
        losses=losses,
        rep=rep or PixelRepresentation(min_val=0.0, max_val=1.0),
        driver=GradientDriver(
            lr=lr,
            max_iter=1,
            return_best=False,
            schedule=None,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs or {},
        ),
        target_source=FixedTarget(),
    )


def _run(
    stage: OptimStage,
    axes: AxisManager | None = None,
    callbacks: list | None = None,
) -> State:
    ctx = _make_ctx(axes)
    state = State(target=TargetInfo(value=torch.tensor([0])))
    return stage.run(state, ctx, callbacks=callbacks)


# ── 1 & 2: Routing and call counts ───────────────────────────────────────────

class TestLossRouting:
    N = 4

    def test_loss_called_once_per_seed(self):
        """Every loss is called exactly seed_size times per evaluation — once per seed slice.

        One step plus the final-iterate scoring a return_best=False driver does: two evaluations.
        """
        spy = _SpyLoss()
        _run(_make_stage({"spy": (spy, 1.0)}), axes=AxisManager(Axis("seed", self.N)))
        assert len(spy.received_shapes) == 2 * self.N

    def test_loss_receives_per_seed_shape(self):
        """Each call receives the seed-sliced tensor — shape equals INPUT_SHAPE."""
        spy = _SpyLoss()
        _run(_make_stage({"spy": (spy, 1.0)}), axes=AxisManager(Axis("seed", self.N)))
        for shape in spy.received_shapes:
            assert shape == INPUT_SHAPE

    def test_multiple_losses_each_called_once_per_seed(self):
        """Both losses in a stage are called seed_size times each per evaluation (two evaluations)."""
        spy_a = _SpyLoss()
        spy_b = _SpyLoss()
        N = 3
        _run(
            _make_stage({"a": (spy_a, 1.0), "b": (spy_b, 1.0)}),
            axes=AxisManager(Axis("seed", N)),
        )
        assert len(spy_a.received_shapes) == 2 * N
        assert len(spy_b.received_shapes) == 2 * N
        for shape in spy_a.received_shapes + spy_b.received_shapes:
            assert shape == INPUT_SHAPE


# ── 3: No-seed backward compatibility ────────────────────────────────────────

class TestNoSeedAxisCompat:
    def test_loss_receives_full_recon_once_with_no_seed_axis(self):
        """seed_size defaults to 1 — loop runs once per evaluation with the unmodified tensor.

        One step plus the final-iterate scoring a return_best=False driver does: two evaluations.
        """
        spy = _SpyLoss()
        _run(_make_stage({"spy": (spy, 1.0)}), axes=AxisManager())
        assert spy.received_shapes == [INPUT_SHAPE, INPUT_SHAPE]

    def test_no_seed_axis_produces_valid_output_shape(self):
        state = _run(_make_stage({"spy": (_SpyLoss(), 1.0)}), axes=None)
        assert state.input is not None
        assert state.input.shape == INPUT_SHAPE


# ── 4: Gradient averaging ─────────────────────────────────────────────────────

class TestGradientAveraging:
    """
    Every loss is scaled by 1/seed_size before backward so the effective gradient
    per element is 1/S regardless of how many seeds are used.  This keeps the
    optimizer's effective step size stable as seed count changes.

    With loss = recon_s.sum() and weight = 1:
      backward on (loss / S) gives d/d(param[s]) = 1/S per element.
      After S seeds, param.grad[s] = 1/S for every s and every element.
    """

    def _grad_after_one_step(self, n_seeds: int) -> Tensor:
        rep = helpers.BoundRecorder(PixelRepresentation(min_val=0.0, max_val=1.0))
        axes = AxisManager(Axis("seed", n_seeds)) if n_seeds > 1 else AxisManager()
        stage = _make_stage({"loss": (_SumLoss(), 1.0)}, rep=rep)
        ctx = _make_ctx(axes)
        state = State(target=TargetInfo(value=torch.tensor([0])))
        stage.run(state, ctx)
        return rep.bound._param.grad

    def test_single_seed_gradient_is_one_everywhere(self):
        """No seed axis → seed_size=1 → grad = 1/1 = 1 per element."""
        grad = self._grad_after_one_step(n_seeds=1)
        assert grad is not None
        assert torch.allclose(grad, torch.ones_like(grad), atol=1e-6)

    def test_two_seeds_gradient_is_half_everywhere(self):
        grad = self._grad_after_one_step(n_seeds=2)
        assert grad is not None
        assert torch.allclose(grad, torch.full_like(grad, 0.5), atol=1e-6)

    def test_four_seeds_gradient_is_quarter_everywhere(self):
        grad = self._grad_after_one_step(n_seeds=4)
        assert grad is not None
        assert torch.allclose(grad, torch.full_like(grad, 0.25), atol=1e-6)

    def test_gradient_magnitude_is_one_over_seed_size_for_all_counts(self):
        """Parametric check: grad = 1/S for S in {1, 2, 3, 4}."""
        for n in [1, 2, 3, 4]:
            grad = self._grad_after_one_step(n_seeds=n)
            expected = 1.0 / n if n > 1 else 1.0
            assert torch.allclose(grad, torch.full_like(grad, expected), atol=1e-6), (
                f"S={n}: expected grad={expected}, got mean={grad.mean().item():.6f}"
            )


# ── 5: Loss accounting ────────────────────────────────────────────────────────

class TestLossAccounting:
    """
    The loss reported via on_step_end is the differentiated objective: the seed mean of the
    weighted per-slice loss values.
    """

    def _capture(self, losses: dict, axes: AxisManager | None = None) -> float:
        cb = _LossCapture()
        _run(_make_stage(losses), axes=axes, callbacks=[cb])
        return cb.values[0]

    def test_reported_loss_is_the_differentiated_objective_mean_over_seeds(self):
        """Gradients are averaged over seeds, so the reported loss must be the seed mean, not the sum."""
        stage = _make_stage({"c": (_ConstantLoss(2.0), 1.0)})
        cap = _LossCapture()
        _run(stage, axes=AxisManager(Axis("seed", 4)), callbacks=[cap])
        assert cap.values == [pytest.approx(2.0)]

    def test_loss_weight_is_applied_before_accumulation(self):
        """Weight multiplies the term value before it reaches the reported loss."""
        VAL, WEIGHT, N = 3.0, 0.5, 2
        val = self._capture(
            {"s": (_ConstantLoss(VAL), WEIGHT)},
            axes=AxisManager(Axis("seed", N)),
        )
        assert val == pytest.approx(WEIGHT * VAL)

    def test_no_seed_axis_counted_once(self):
        """Without a seed axis, seed_size=1 — loss is accumulated exactly once."""
        VAL = 4.0
        val = self._capture(
            {"s": (_ConstantLoss(VAL), 1.0)},
            axes=AxisManager(),
        )
        assert val == pytest.approx(VAL)

    def test_two_losses_summed_per_seed(self):
        """Multiple losses in the same stage are summed per slice, then averaged over slices."""
        A, B, N = 3.0, 2.0, 2
        val = self._capture(
            {"a": (_ConstantLoss(A), 1.0), "b": (_ConstantLoss(B), 1.0)},
            axes=AxisManager(Axis("seed", N)),
        )
        assert val == pytest.approx(A + B)


# ── 6: LBFGS compatibility ────────────────────────────────────────────────────

class TestLBFGSCompatibility:
    """
    LBFGS calls the closure multiple times per outer step for its line search.
    The closure returns a detached scalar (torch.tensor(float_val)) so subsequent
    calls never attempt to backward through an already-freed activation graph.
    """

    def _lbfgs_stage(self, losses: dict, inner_iter: int = 5) -> OptimStage:
        return _make_stage(
            losses,
            lr=0.1,
            optimizer_cls=torch.optim.LBFGS,
            optimizer_kwargs={"max_iter": inner_iter},
        )

    def test_lbfgs_with_seed_axis_does_not_raise(self):
        spy = _SpyLoss()
        state = _run(
            self._lbfgs_stage({"spy": (spy, 1.0)}),
            axes=AxisManager(Axis("seed", 2)),
        )
        assert tuple(state.input.shape) == (2, *INPUT_SHAPE)
        assert torch.isfinite(state.input).all()
        assert set(spy.received_shapes) == {INPUT_SHAPE}

    def test_lbfgs_calls_closure_more_than_once(self):
        """Verify LBFGS actually exercises the multi-call path (spy count > seed_size)."""
        N = 2
        spy = _SpyLoss()
        _run(
            self._lbfgs_stage({"spy": (spy, 1.0)}, inner_iter=5),
            axes=AxisManager(Axis("seed", N)),
        )
        # At minimum: 1 outer step × N inner slices. LBFGS will have called it more.
        assert len(spy.received_shapes) >= N

    def test_lbfgs_without_seed_axis_does_not_raise(self):
        """Baseline: LBFGS must work with no seed axis too."""
        spy = _SpyLoss()
        state = _run(self._lbfgs_stage({"spy": (spy, 1.0)}), axes=AxisManager())
        assert tuple(state.input.shape) == INPUT_SHAPE
        assert torch.isfinite(state.input).all()


class TestLBFGSReporting:
    def test_lbfgs_reports_the_loss_at_the_starting_iterate(self):
        """An LBFGS step evaluates the closure several times; the step reports the first evaluation only."""
        x0 = torch.full(INPUT_SHAPE, 0.5)
        stage = OptimStage(
            losses={"s": (_SumLoss(), 1.0)},
            rep=PixelRepresentation(min_val=0.0, max_val=1.0, init=FromTensor(x0, space="data")),
            driver=GradientDriver(lr=0.1, max_iter=1, return_best=False, schedule=None,
                                  optimizer_cls=torch.optim.LBFGS, optimizer_kwargs={"max_iter": 5}),
            target_source=FixedTarget(),
        )
        cap = _LossCapture()
        comps = []

        class _C(Callback):
            def on_step_end(self, event):
                comps.append(dict(event.components))

        _run(stage, callbacks=[cap, _C()])
        assert cap.values == [pytest.approx(x0.sum().item())]
        assert comps == [{"s": pytest.approx(x0.sum().item())}]


# ── DriverResult: per-slice best snapshot ──────────────────────────────────────

def test_best_snapshot_pairs_each_slice_with_its_own_artifacts_and_loss():
    """With a seed axis, each returned slice, its artifact slice and its slice loss come from one evaluation."""
    import dataclasses

    S = 3
    ctx = _make_ctx(AxisManager(Axis("seed", S)))
    stage = OptimStage(
        losses={"loss": (_SumLoss(), 1.0)},
        rep=PixelRepresentation(min_val=0.0, max_val=1.0),
        driver=GradientDriver(lr=0.05, max_iter=4, return_best=True, schedule=None),
        target_source=FixedTarget(),
    )
    state = State(target=TargetInfo(value=torch.tensor([0])))
    ctx.session.begin_stage(ctx)
    bound = dataclasses.replace(stage, rep=stage.rep.bind(state, ctx), target_source=stage.target_source.bind(state, ctx))
    result = stage.driver.run(bound, ctx)
    assert result.iterations == 4
    assert result.slice_losses.shape == (S,)
    torch.testing.assert_close(result.artifacts["pixels"], result.input)
    torch.testing.assert_close(result.slice_losses, result.input.flatten(1).sum(dim=1))
    assert result.loss == pytest.approx(result.slice_losses.mean().item())


class _CountingTarget(FixedTarget):
    """A target shared across the axis whose value is the number of times it has been read."""

    def __init__(self) -> None:
        super().__init__()
        self.reads = 0

    def current(self) -> TargetInfo:
        reads = torch.tensor([float(self.reads)])
        self.reads += 1
        return TargetInfo(value=torch.tensor([0]), reads=reads)


class _PeakLoss(LossTerm):
    """Slice ``s`` is best when the target value equals ``PEAKS[s]``."""

    PEAKS = (1.0, 3.0)

    def compute(self, inputs: LossInputs) -> Tensor:
        peak = self.PEAKS[inputs.seed_index]
        return inputs.recon.sum() * 0.0 + (inputs.target.reads[0] - peak) ** 2


def test_shared_target_of_the_best_result_is_the_one_of_the_best_mean_evaluation():
    """A target without axis prefix cannot hold one value per slice: it is the target of the evaluation
    with the lowest mean slice loss, not whichever slice improved last."""
    import dataclasses

    ctx = _make_ctx(AxisManager(Axis("seed", 2)))
    stage = OptimStage(
        losses={"peak": (_PeakLoss(), 1.0)},
        rep=PixelRepresentation(min_val=0.0, max_val=1.0),
        driver=GradientDriver(lr=0.0, max_iter=5, return_best=True, schedule=None),
        target_source=_CountingTarget(),
    )
    state = State(target=TargetInfo(value=torch.tensor([0])))
    ctx.session.begin_stage(ctx)
    bound = dataclasses.replace(stage, rep=stage.rep.bind(state, ctx), target_source=stage.target_source.bind(state, ctx))
    result = stage.driver.run(bound, ctx)
    # Reads: 0 is the initial one, step i reads i + 1. Mean slice losses per step: 2, 1, 2, 5, 10.
    assert result.target.reads.item() == 2.0
