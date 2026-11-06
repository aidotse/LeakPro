#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Parity test: split run == continuous run.

A two-stage attack run in one call must produce the same output as running
stage 0, saving state to disk, loading it back, and running stage 1 separately.
Uses a tiny synthetic model and input so no dataset download is required.
"""
import os
import tempfile

import torch
import torch.nn as nn

# (pytest's rootdir-relative insertion lands on a namespace-package boundary
# here, which otherwise breaks `import leakpro.attacks...`).

from leakpro.attacks.gia_attacks.modular.core.attack import Attack, run_attack
from leakpro.attacks.gia_attacks.modular.core.io import save_state, load_state
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, TargetInfo, RunMetadata
from leakpro.attacks.gia_attacks.modular.components.losses import CosineGradientLoss, TVLoss
from leakpro.attacks.gia_attacks.modular.components.drivers import GradientDriver
from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
from leakpro.attacks.gia_attacks.modular.core.task import Task
from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget
from leakpro.attacks.gia_attacks.modular.components.modalities.image import ImageNormalizationModality
from leakpro.attacks.gia_attacks.modular.components.representations.pixel import PixelRepresentation


# ── Fixtures ─────────────────────────────────────────────────────────────────

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


def make_two_stage_attack(n_iters_per_stage: int = 10) -> Attack:
    """Two identical stages so we can test state hand-off between them."""
    modality = ImageNormalizationModality(mean=MEAN, std=STD, device=DEVICE)
    task = Task(CrossEntropy())
    losses = {
        "grad_match": (CosineGradientLoss(), 1.0),
        "tv": (TVLoss(), 1e-4),
    }
    stage = OptimStage(
        losses=losses,
        rep=PixelRepresentation(min_val=0.0, max_val=1.0),
        driver=GradientDriver(lr=1e-2, max_iter=n_iters_per_stage, return_best=False),
        target_source=FixedTarget(),
    )
    # Two independent stages — same config, each gets a fresh rep instance
    stage2 = OptimStage(
        losses=losses,
        rep=PixelRepresentation(min_val=0.0, max_val=1.0),
        driver=GradientDriver(lr=1e-2, max_iter=n_iters_per_stage, return_best=False),
        target_source=FixedTarget(),
    )
    return Attack(modality=modality, task=task, stages=[stage, stage2])


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_split_run_equals_continuous():
    torch.manual_seed(0)
    model = make_model().to(DEVICE)
    images = torch.rand(INPUT_SHAPE, device=DEVICE)
    target = TargetInfo(value=torch.tensor([3], device=DEVICE))
    observation = make_observation(model, images, target)
    metadata = RunMetadata(seed=42)

    atk = make_two_stage_attack(n_iters_per_stage=10)

    # Continuous: run both stages in one call
    state_continuous, _ = run_attack(
        atk, model, observation, INPUT_SHAPE, DEVICE,
        target=target, metadata=metadata,
    )

    # Split: stage 0, checkpoint, stage 1
    atk2 = make_two_stage_attack(n_iters_per_stage=10)
    state_0, _ = run_attack(
        atk2, model, observation, INPUT_SHAPE, DEVICE,
        target=target, metadata=metadata,
        start=0, stop=1,
    )

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name

    try:
        save_state(state_0, ckpt_path)
        loaded = load_state(ckpt_path)

        state_split, _ = run_attack(
            atk2, model, observation, INPUT_SHAPE, DEVICE,
            target=target, metadata=metadata,
            start=1, stop=2,
            input_state=loaded,
        )
    finally:
        os.unlink(ckpt_path)

    assert torch.allclose(state_continuous.input, state_split.input, atol=1e-6), (
        f"Split run diverged from continuous run.\n"
        f"  max diff: {(state_continuous.input - state_split.input).abs().max().item():.2e}"
    )
    print("PASS: split run == continuous run")


def test_tunables_and_with_overrides():
    """tunables() surfaces the correct keys; with_overrides() injects them correctly."""
    atk = make_two_stage_attack()

    tunables = atk.tunables()
    assert "0.driver.lr" in tunables
    assert "0.loss.tv.weight" in tunables
    assert "1.driver.lr" in tunables

    overrides = {"0.driver.lr": 0.05, "0.loss.tv.weight": 2e-4, "1.driver.lr": 0.02}
    atk2 = atk.with_overrides(overrides)

    assert atk2.stages[0].driver.lr == 0.05
    assert atk2.stages[0].losses["tv"][1] == 2e-4
    assert atk2.stages[1].driver.lr == 0.02
    # Original untouched
    assert atk.stages[0].driver.lr != 0.05
    print("PASS: tunables() and with_overrides() correct")


def test_resumed_run_leaves_the_callers_input_state_unchanged():
    torch.manual_seed(0)
    model = make_model().to(DEVICE)
    images = torch.rand(INPUT_SHAPE, device=DEVICE)
    target = TargetInfo(value=torch.tensor([3], device=DEVICE))
    observation = make_observation(model, images, target)
    atk = make_two_stage_attack(n_iters_per_stage=3)
    state_0, _ = run_attack(
        atk, model, observation, INPUT_SHAPE, DEVICE, target=target, metadata=RunMetadata(seed=42), start=0, stop=1,
    )
    state_0.artifacts["extra"] = torch.ones(2)
    held = (state_0.input, state_0.target, state_0.artifacts, state_0.metadata, state_0.loss, state_0.iteration)
    before = (state_0.input.clone(), state_0.target.value.clone(), state_0.artifacts["extra"].clone())

    state_1, _ = run_attack(
        atk, model, observation, INPUT_SHAPE, DEVICE, start=1, input_state=state_0, metadata=RunMetadata(seed=7),
    )

    assert state_1 is not state_0
    assert (state_0.input, state_0.target, state_0.artifacts, state_0.metadata, state_0.loss, state_0.iteration) == held
    assert torch.equal(state_0.input, before[0])
    assert torch.equal(state_0.target.value, before[1])
    assert torch.equal(state_0.artifacts["extra"], before[2])
    assert state_0.metadata.seed == 42
    assert not torch.equal(state_1.input, before[0])
