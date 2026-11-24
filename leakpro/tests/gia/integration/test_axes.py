#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Dynamic axis registry tests.

Validates:
  1. AxisManager.shape returns the correct prefix tuple.
  2. AxisManager.get_dim returns correct indices and None for absent axes.
  3. AxisManager.get_size returns correct sizes and 1 for absent axes.
  4. Empty AxisManager yields () prefix — existing single-seed code unaffected.
  5. Context carries the Attack's AxisManager through run_attack.
  6. PixelRepresentation allocates [*axes.shape, N, C, H, W] when axes are set.
  7. PixelRepresentation allocates [N, C, H, W] when AxisManager is empty (backward compat).
  8. PixelRepresentation resume from state.input preserves the prefixed shape.
  9. run_attack with axes=AxisManager(seed=4) produces a (4,1,3,8,8) reconstruction.
 10. Existing split-run parity holds with no axes (empty AxisManager default).
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

# (pytest's rootdir-relative insertion lands on a namespace-package boundary
# here, which otherwise breaks `import leakpro.attacks...`).

from leakpro.attacks.gia_attacks.modular.core.axes import Axis, AxisManager
from leakpro.attacks.gia_attacks.modular.core.attack import Attack, run_attack
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, Context, RunMetadata, TargetInfo
from leakpro.attacks.gia_attacks.modular.components.losses import CosineGradientLoss, TVLoss
from leakpro.attacks.gia_attacks.modular.components.drivers import GradientDriver
from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget
from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
from leakpro.attacks.gia_attacks.modular.core.task import Task
from leakpro.attacks.gia_attacks.modular.components.modalities.image import ImageNormalizationModality
from leakpro.attacks.gia_attacks.modular.components.representations.pixel import PixelRepresentation


# ── Fixtures ──────────────────────────────────────────────────────────────────

INPUT_SHAPE = (1, 3, 8, 8)
DEVICE = torch.device("cpu")
MEAN = [0.5, 0.5, 0.5]
STD  = [0.5, 0.5, 0.5]


def make_model() -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 10))


def make_observation(model: nn.Module, images: torch.Tensor, target: TargetInfo) -> ClientObservation:
    task = Task(CrossEntropy())
    obs = task.victim_signal(model, images, target)
    return ClientObservation(gradients=[g.detach().clone() for g in obs.gradients])


def make_attack(axes: AxisManager | None = None, max_iter: int = 5) -> Attack:
    return Attack(
        modality=ImageNormalizationModality(mean=MEAN, std=STD, device=DEVICE),
        task=Task(CrossEntropy()),
        stages=[
            OptimStage(
                losses={"grad_match": (CosineGradientLoss(), 1.0), "tv": (TVLoss(), 1e-4)},
                rep=PixelRepresentation(min_val=0.0, max_val=1.0),
                driver=GradientDriver(lr=1e-2, max_iter=max_iter, return_best=False),
                target_source=FixedTarget(),
            )
        ],
        axes=axes if axes is not None else AxisManager(),
    )


# ── 1–4: AxisManager unit tests ───────────────────────────────────────────────

def test_axis_manager_shape():
    am = AxisManager(Axis("epoch", 2), Axis("seed", 4))
    assert am.shape == (2, 4), f"Expected (2, 4), got {am.shape}"
    print("[OK] AxisManager.shape")


def test_axis_manager_get_dim():
    am = AxisManager(Axis("epoch", 2), Axis("seed", 4))
    assert am.get_dim("epoch") == 0
    assert am.get_dim("seed") == 1
    assert am.get_dim("missing") is None
    print("[OK] AxisManager.get_dim")


def test_axis_manager_get_size():
    am = AxisManager(Axis("epoch", 3), Axis("seed", 5))
    assert am.get_size("epoch") == 3
    assert am.get_size("seed") == 5
    assert am.get_size("absent") == 1, "absent axis should default to 1"
    print("[OK] AxisManager.get_size")


def test_empty_axis_manager():
    am = AxisManager()
    assert am.shape == (), f"Expected (), got {am.shape}"
    assert am.get_dim("seed") is None
    assert am.get_size("seed") == 1
    print("[OK] empty AxisManager yields () prefix")


# ── 5: Context carries axes through run_attack ────────────────────────────────

def test_context_carries_axes():
    torch.manual_seed(0)
    am = AxisManager(Axis("seed", 3))
    model = make_model().to(DEVICE)
    images = torch.rand(INPUT_SHAPE, device=DEVICE)
    target = TargetInfo(value=torch.tensor([2], device=DEVICE))
    obs = make_observation(model, images, target)

    # max_iter=0: initializes the rep but runs no loss steps; this only verifies the axes flow through to Context.
    atk = make_attack(axes=am, max_iter=0)
    _, ctx = run_attack(atk, model, obs, INPUT_SHAPE, DEVICE, target=target)

    assert ctx.axes == am, f"Context.axes mismatch: {ctx.axes!r} != {am!r}"
    assert ctx.axes.get_dim("seed") == 0
    print("[OK] Context carries AxisManager from Attack")


# ── 6: PixelRepresentation allocates prefixed tensor ─────────────────────────

def test_pixel_rep_prefixed_shape():
    torch.manual_seed(0)
    am = AxisManager(Axis("seed", 4))
    model = make_model().to(DEVICE)
    images = torch.rand(INPUT_SHAPE, device=DEVICE)
    target = TargetInfo(value=torch.tensor([1], device=DEVICE))
    obs = make_observation(model, images, target)

    # max_iter=0: no loss steps; verify the rep allocates (*axes.shape, N, C, H, W).
    atk = make_attack(axes=am, max_iter=0)
    state, _ = run_attack(atk, model, obs, INPUT_SHAPE, DEVICE, target=target)

    expected = (4, *INPUT_SHAPE)
    assert state.input.shape == expected, (
        f"Expected shape {expected}, got {tuple(state.input.shape)}"
    )
    print("[OK] PixelRepresentation allocates prefixed shape (4,1,3,8,8)")


# ── 7: Backward compat — empty axes yields unmodified shape ──────────────────

def test_pixel_rep_no_axes_backward_compat():
    torch.manual_seed(0)
    model = make_model().to(DEVICE)
    images = torch.rand(INPUT_SHAPE, device=DEVICE)
    target = TargetInfo(value=torch.tensor([5], device=DEVICE))
    obs = make_observation(model, images, target)

    atk = make_attack(axes=None, max_iter=1)  # defaults to AxisManager()
    state, _ = run_attack(atk, model, obs, INPUT_SHAPE, DEVICE, target=target)

    assert state.input.shape == INPUT_SHAPE, (
        f"Expected {INPUT_SHAPE}, got {tuple(state.input.shape)}"
    )
    print("[OK] empty AxisManager: shape unchanged (backward compat)")


# ── 8: Resume from state.input preserves prefixed shape ──────────────────────

def test_pixel_rep_resume_preserves_shape():
    """bind() with a prefixed state.input must not reshape or re-init the tensor."""
    torch.manual_seed(0)
    am = AxisManager(Axis("seed", 3))
    prefixed_shape = (3, *INPUT_SHAPE)

    ctx_stub = Context(
        model=make_model().to(DEVICE),
        observation=ClientObservation(gradients=[]),
        modality=ImageNormalizationModality(mean=MEAN, std=STD, device=DEVICE),
        task=Task(CrossEntropy()),
        input_shape=INPUT_SHAPE,
        device=DEVICE,
        axes=am,
    )

    from leakpro.attacks.gia_attacks.modular.core.state import State
    prior_input = torch.rand(prefixed_shape, device=DEVICE)
    state = State(input=prior_input)

    rep = PixelRepresentation()
    rep = rep.bind(state, ctx_stub)

    out = rep.decode()
    assert out.shape == prefixed_shape, (
        f"Expected {prefixed_shape} after resume, got {tuple(out.shape)}"
    )
    print("[OK] PixelRepresentation.bind() preserves prefixed shape on resume")


# ── 9: run_attack with seed axis produces (4,1,3,8,8) reconstruction ─────────

def test_run_attack_seed_axis_shape():
    """run_attack with a seed axis allocates and returns a prefixed reconstruction.

    max_iter=0 is intentional: this validates the allocation and routing only, not per-seed gradient matching.
    """
    torch.manual_seed(42)
    N_SEEDS = 4
    am = AxisManager(Axis("seed", N_SEEDS))
    model = make_model().to(DEVICE)
    images = torch.rand(INPUT_SHAPE, device=DEVICE)
    target = TargetInfo(value=torch.tensor([7], device=DEVICE))
    obs = make_observation(model, images, target)

    atk = make_attack(axes=am, max_iter=0)
    state, ctx = run_attack(atk, model, obs, INPUT_SHAPE, DEVICE, target=target)

    expected = (N_SEEDS, *INPUT_SHAPE)
    assert state.input.shape == expected, (
        f"run_attack with seed axis: expected {expected}, got {tuple(state.input.shape)}"
    )
    assert ctx.axes.get_size("seed") == N_SEEDS
    print(f"[OK] run_attack with axes=seed={N_SEEDS} → reconstruction shape {expected}")


# ── 10: Existing split-run parity with empty axes ────────────────────────────

def test_no_axes_split_run_parity():
    """Confirm existing parity behaviour is unaffected by the axes default."""
    import tempfile

    from leakpro.attacks.gia_attacks.modular.core.attack import Attack
    from leakpro.attacks.gia_attacks.modular.core.io import save_state, load_state

    def make_two_stage():
        losses = {"grad_match": (CosineGradientLoss(), 1.0), "tv": (TVLoss(), 1e-4)}
        def stage():
            return OptimStage(
                losses=losses,
                rep=PixelRepresentation(min_val=0.0, max_val=1.0),
                driver=GradientDriver(lr=1e-2, max_iter=5, return_best=False),
                target_source=FixedTarget(),
            )
        return Attack(
            modality=ImageNormalizationModality(mean=MEAN, std=STD, device=DEVICE),
            task=Task(CrossEntropy()),
            stages=[stage(), stage()],
            # axes omitted → defaults to AxisManager()
        )

    torch.manual_seed(0)
    model = make_model().to(DEVICE)
    images = torch.rand(INPUT_SHAPE, device=DEVICE)
    target = TargetInfo(value=torch.tensor([3], device=DEVICE))
    obs = make_observation(model, images, target)
    meta = RunMetadata(seed=42)

    atk1 = make_two_stage()
    state_cont, _ = run_attack(atk1, model, obs, INPUT_SHAPE, DEVICE, target=target, metadata=meta)

    atk2 = make_two_stage()
    state0, _ = run_attack(atk2, model, obs, INPUT_SHAPE, DEVICE, target=target, metadata=meta, start=0, stop=1)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt = f.name
    try:
        save_state(state0, ckpt)
        loaded = load_state(ckpt)
        state_split, _ = run_attack(
            atk2, model, obs, INPUT_SHAPE, DEVICE,
            target=target, metadata=meta, start=1, stop=2, input_state=loaded,
        )
    finally:
        os.unlink(ckpt)

    assert torch.allclose(state_cont.input, state_split.input, atol=1e-6), (
        f"Split run diverged: max diff "
        f"{(state_cont.input - state_split.input).abs().max().item():.2e}"
    )
    print("[OK] split-run parity unaffected by axes default")


# ── Entry point ───────────────────────────────────────────────────────────────
