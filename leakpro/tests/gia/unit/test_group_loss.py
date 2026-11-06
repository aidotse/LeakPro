#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for the two-pass GROUP loss mechanism.

These tests verify:
1. GroupConsistencyLoss produces true cross-seed gradients (no detach).
2. The GROUP pass is skipped when seed_size == 1 (no behavioral change for single-seed presets).
3. GROUP losses are skipped when no seed axis is registered.
4. Loss scope partitioning: GroupConsistencyLoss goes to group, others go to per_slice.
5. Parity guard: a PER_SLICE _SumLoss over S seeds still gives grad = 1/S (two-pass doesn't break it).
"""
from __future__ import annotations


import torch
import torch.nn as nn
from torch import Tensor


from leakpro.attacks.gia_attacks.modular.core.axes import Axis, AxisManager
from leakpro.attacks.gia_attacks.modular.core.interfaces import LossTerm
from leakpro.attacks.gia_attacks.modular.core.loss_inputs import LossInputs, LossScope
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.attacks.gia_attacks.modular.core.state import (
    ClientObservation, Context, State, TargetInfo,
)
from leakpro.attacks.gia_attacks.modular.components.drivers import GradientDriver
from leakpro.attacks.gia_attacks.modular.components.losses import GroupConsistencyLoss
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

class _SumLoss(LossTerm):
    """PER_SLICE: returns recon.sum() — gradient = 1 everywhere."""
    def compute(self, inputs: LossInputs) -> Tensor:
        return inputs.recon.sum()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_model() -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 10)).eval()


def _make_ctx(axes: AxisManager) -> Context:
    model = _make_model()
    task = Task(CrossEntropy())
    imgs = torch.rand(INPUT_SHAPE)
    obs_raw = task.victim_signal(model, imgs, TargetInfo(value=torch.tensor([0])))
    obs = ClientObservation(gradients=[g.detach().clone() for g in obs_raw.gradients])
    return Context(
        model=model,
        observation=obs,
        modality=ImageNormalizationModality(mean=MEAN, std=STD, device=DEVICE),
        task=task,
        input_shape=INPUT_SHAPE,
        device=DEVICE,
        axes=axes,
    )


def _make_stage(losses: dict, axes: AxisManager | None = None) -> tuple[OptimStage, Context]:
    rep = PixelRepresentation(min_val=0.0, max_val=1.0)
    stage = OptimStage(
        losses=losses,
        rep=rep,
        driver=GradientDriver(lr=0.0, max_iter=1, return_best=False, schedule=None),
        target_source=FixedTarget(),
    )
    ctx = _make_ctx(axes or AxisManager())
    return stage, ctx


def _run_one_step(losses: dict, axes: AxisManager) -> PixelRepresentation:
    """Run one GradientDriver step; return the rep so we can inspect .grad."""
    rep = helpers.BoundRecorder(PixelRepresentation(min_val=0.0, max_val=1.0))
    stage = OptimStage(
        losses=losses,
        rep=rep,
        driver=GradientDriver(lr=0.0, max_iter=1, return_best=False, schedule=None),
        target_source=FixedTarget(),
    )
    ctx = _make_ctx(axes)
    state = State(target=TargetInfo(value=torch.tensor([0])))
    stage.run(state, ctx)
    return rep.bound


# ── 1. True cross-seed gradient flow ─────────────────────────────────────────

class TestCrossSeedGradientFlow:
    """GroupConsistencyLoss must produce gradients that depend on OTHER seeds' values.

    Under the old detached-consensus design this was impossible: seed-0's gradient
    was computed from `(recon_0 - recon.detach().mean())`, making the gradient of
    seed-0 independent of seed-1's value.  With a live consensus, perturbing seed-1
    changes the consensus and therefore changes seed-0's gradient.
    """

    def _get_grad_seed0(self, seed1_init: float) -> Tensor:
        """Run one step with 2 seeds; seed-1 initialized to seed1_init. Return seed-0 grad."""
        axes = AxisManager(Axis("seed", 2))
        ctx = _make_ctx(axes)

        # Build the initial pixel tensor and inject via state.input so bind() preserves it.
        # Shape: [S=2, N=1, C=3, H=4, W=4]
        init_pixels = torch.rand(2, *INPUT_SHAPE)
        init_pixels[1].fill_(seed1_init)  # seed dim=0, index 1

        rep = helpers.BoundRecorder(PixelRepresentation(min_val=0.0, max_val=1.0))
        stage = OptimStage(
            losses={"group": (GroupConsistencyLoss(), 1.0)},
            rep=rep,
            driver=GradientDriver(lr=0.0, max_iter=1, return_best=False, schedule=None),
            target_source=FixedTarget(),
        )
        # Pass init_pixels as state.input — bind() will clone it with requires_grad=True.
        state = State(input=init_pixels, target=TargetInfo(value=torch.tensor([0])))

        # Run one step (lr=0 so param values don't change, but grad is computed).
        stage.run(state, ctx)
        return rep.bound._param.grad[0].clone()  # gradient for seed 0

    def test_seed0_grad_depends_on_seed1_value(self):
        """Perturbing seed-1 changes the consensus, which changes seed-0's gradient."""
        grad_a = self._get_grad_seed0(seed1_init=0.0)
        grad_b = self._get_grad_seed0(seed1_init=1.0)
        # If the consensus were detached, grad_a == grad_b always.
        assert not torch.allclose(grad_a, grad_b), (
            "seed-0 gradient is identical for different seed-1 values — "
            "the consensus is likely detached (old behavior)"
        )

    def test_seed0_grad_is_nonzero(self):
        """GroupConsistencyLoss must produce a nonzero gradient for non-trivial seeds."""
        # seed-0 ≈ 0.5 (uniform init), seed-1 = 0.0 → consensus ≠ seed-0 → nonzero grad
        grad = self._get_grad_seed0(seed1_init=0.0)
        assert grad.abs().max().item() > 0.0


# ── 2. Single-seed degeneracy ─────────────────────────────────────────────────

class TestSingleSeedDegeneracy:
    """With seed_size == 1, GroupConsistencyLoss must be a no-op (returns 0, no grad)."""

    def test_returns_zero_for_single_seed(self):
        """GROUP loss with only one seed: run completes without error and grad is zero."""
        rep = _run_one_step(
            losses={"group": (GroupConsistencyLoss(), 1.0), "dummy": (_SumLoss(), 0.0)},
            axes=AxisManager(Axis("seed", 1)),
        )
        # _SumLoss weight=0 produces zero grad, so any gradient comes from the group term, which must be zero.
        assert rep._param.grad is not None
        assert rep._param.grad.abs().max().item() == 0.0

    def test_returns_zero_with_no_seed_axis(self):
        """Without a seed axis, GROUP pass is skipped entirely; no error and no gradient from it."""
        rep = _run_one_step(
            losses={"group": (GroupConsistencyLoss(), 1.0), "dummy": (_SumLoss(), 0.0)},
            axes=AxisManager(),
        )
        assert rep._param.grad is not None
        assert rep._param.grad.abs().max().item() == 0.0


# ── 3. Loss scope partitioning ───────────────────────────────────────────────

class TestLossScopePartitioning:
    def test_group_consistency_has_group_scope(self):
        assert GroupConsistencyLoss.scope is LossScope.GROUP

    def test_sum_loss_has_per_slice_scope_by_default(self):
        assert getattr(_SumLoss, "scope", LossScope.PER_SLICE) is LossScope.PER_SLICE

    def test_driver_partitions_correctly(self):
        """The driver partitions stage losses into per_slice and group dicts."""
        # We can verify by checking that _SumLoss (PER_SLICE) still runs seed_size times.
        # Use a spy to count calls.
        class _CountingLoss(LossTerm):
            calls = 0
            def compute(self, inputs: LossInputs) -> Tensor:
                _CountingLoss.calls += 1
                return inputs.recon.sum() * 0

        N = 3
        _CountingLoss.calls = 0
        _run_one_step(
            losses={"count": (_CountingLoss(), 1.0), "group": (GroupConsistencyLoss(), 0.0)},
            axes=AxisManager(Axis("seed", N)),
        )
        # One step plus the final-iterate scoring of a return_best=False driver: two evaluations.
        assert _CountingLoss.calls == 2 * N, (
            f"PER_SLICE loss should be called {2 * N} times, got {_CountingLoss.calls}"
        )


# ── 4. PER_SLICE parity guard (1/S gradient) ─────────────────────────────────

class TestPerSliceParity:
    """The two-pass refactor must not change per-slice gradient magnitude."""

    def test_per_slice_sum_loss_gives_one_over_S_gradient(self):
        """With only PER_SLICE losses and S seeds, grad = 1/S per element."""
        for S in [1, 2, 4]:
            axes = AxisManager(Axis("seed", S)) if S > 1 else AxisManager()
            rep = _run_one_step(losses={"loss": (_SumLoss(), 1.0)}, axes=axes)
            expected = 1.0 / S
            assert torch.allclose(
                rep._param.grad,
                torch.full_like(rep._param.grad, expected),
                atol=1e-6,
            ), f"S={S}: expected grad={expected}, got mean={rep._param.grad.mean().item():.6f}"
