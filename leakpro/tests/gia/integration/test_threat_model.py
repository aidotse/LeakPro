#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for ThreatModel (data-firewall) + ClientObservation.

Acceptance criterion: ThreatModel(has_bn_statistics=False) strips bn_stats before
the attack runs, so BNStatsLoss raises ValueError at execution time.
ThreatModel(has_bn_statistics=True) preserves it and BNStatsLoss computes
the feature-statistics regularization via forward hooks.
"""

import math

import pytest
import torch
import torch.nn as nn

# correctly and prevent double-loading (which would create two distinct exception
# classes and break pytest.raises identity checks).

from leakpro.attacks.gia_attacks.modular.core.attack import Attack, run_attack
from leakpro.attacks.gia_attacks.modular.core.events import StepEvent
from leakpro.attacks.gia_attacks.modular.core.interfaces import Callback
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, TargetInfo
from leakpro.attacks.gia_attacks.modular.core.threat_model import ThreatModel
from leakpro.attacks.gia_attacks.modular.components.losses import BNStatsLoss, CosineGradientLoss
from leakpro.attacks.gia_attacks.modular.components.drivers import GradientDriver
from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget
from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
from leakpro.attacks.gia_attacks.modular.core.task import Task
from leakpro.attacks.gia_attacks.modular.components.modalities.image import ImageNormalizationModality
from leakpro.attacks.gia_attacks.modular.components.representations.pixel import PixelRepresentation


# ── Fixtures ─────────────────────────────────────────────────────────────────

INPUT_SHAPE = (1, 3, 8, 8)
DEVICE = torch.device("cpu")
MEAN = [0.5, 0.5, 0.5]
STD  = [0.5, 0.5, 0.5]
BN_STATS = [(torch.zeros(3), torch.ones(3))]  # one BN layer with 3 channels


def make_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 3, 3, padding=1),
        nn.BatchNorm2d(3),
        nn.Flatten(),
        nn.Linear(3 * 8 * 8, 10),
    )


def make_observation(with_bn_stats: bool = True) -> ClientObservation:
    model = make_model()
    task = Task(CrossEntropy())
    images = torch.rand(INPUT_SHAPE)
    target = TargetInfo(value=torch.tensor([3]))
    obs = task.victim_signal(model, images, target)
    return ClientObservation(
        gradients=[g.detach().clone() for g in obs.gradients],
        bn_stats=list(BN_STATS) if with_bn_stats else None,
        batch_size=4,
    )


class _Components(Callback):
    """Records the weighted per-term loss values of every step."""

    def __init__(self) -> None:
        self.steps: list[dict[str, float]] = []

    def on_step_end(self, event: StepEvent) -> None:
        self.steps.append(dict(event.components))


def make_attack_with_bn_loss(threat_model=None) -> Attack:
    modality = ImageNormalizationModality(mean=MEAN, std=STD, device=DEVICE)
    task = Task(CrossEntropy())
    stage = OptimStage(
        losses={
            "grad_match": (CosineGradientLoss(), 1.0),
            "bn_stats": (BNStatsLoss(strategy="running"), 0.1),
        },
        rep=PixelRepresentation(min_val=0.0, max_val=1.0),
        driver=GradientDriver(lr=1e-2, max_iter=5, return_best=False),
        target_source=FixedTarget(),
    )
    return Attack(modality=modality, task=task, stages=[stage], threat_model=threat_model)


def make_attack_grad_only(threat_model=None) -> Attack:
    modality = ImageNormalizationModality(mean=MEAN, std=STD, device=DEVICE)
    task = Task(CrossEntropy())
    stage = OptimStage(
        losses={"grad_match": (CosineGradientLoss(), 1.0)},
        rep=PixelRepresentation(min_val=0.0, max_val=1.0),
        driver=GradientDriver(lr=1e-2, max_iter=5, return_best=False),
        target_source=FixedTarget(),
    )
    return Attack(modality=modality, task=task, stages=[stage], threat_model=threat_model)


# ── sanitize() unit tests ─────────────────────────────────────────────────────

def test_sanitize_strips_bn_stats_when_false():
    """has_bn_statistics=False — sanitize must set bn_stats to None."""
    tm = ThreatModel(has_bn_statistics=False)
    obs = ClientObservation(gradients=[torch.randn(10)], bn_stats=list(BN_STATS))
    result = tm.sanitize(obs)
    assert result is not obs, "sanitize returns a copy"
    assert result.bn_stats is None
    assert obs.bn_stats is not None


def test_sanitize_preserves_bn_stats_when_true():
    """has_bn_statistics=True — sanitize must not touch bn_stats."""
    tm = ThreatModel(has_bn_statistics=True)
    obs = ClientObservation(gradients=[torch.randn(10)], bn_stats=list(BN_STATS))
    result = tm.sanitize(obs)
    assert result.bn_stats is not None


def test_sanitize_strips_hyperparams_when_false():
    """has_local_hyperparameters=False — batch_size and loss_value must be stripped."""
    tm = ThreatModel(has_local_hyperparameters=False)
    obs = ClientObservation(gradients=[torch.randn(10)], batch_size=8, loss_value=0.5)
    obs = tm.sanitize(obs)
    assert obs.batch_size is None
    assert obs.loss_value is None


def test_sanitize_preserves_hyperparams_when_true():
    """has_local_hyperparameters=True — batch_size and loss_value are kept."""
    tm = ThreatModel(has_local_hyperparameters=True)
    obs = ClientObservation(gradients=[torch.randn(10)], batch_size=8, loss_value=0.5)
    obs = tm.sanitize(obs)
    assert obs.batch_size == 8
    assert obs.loss_value == 0.5


# ── End-to-end firewall tests ─────────────────────────────────────────────────

def test_no_bn_statistics_causes_bn_stats_loss_to_raise():
    """Core acceptance criterion.

    ThreatModel(has_bn_statistics=False) sanitizes the observation (strips bn_stats).
    run_attack's requirement check then finds bn_stats=None and raises ValueError.
    """
    obs = make_observation(with_bn_stats=True)
    atk = make_attack_with_bn_loss(threat_model=ThreatModel(has_bn_statistics=False))
    target = TargetInfo(value=torch.tensor([3]))

    with pytest.raises(ValueError, match="bn_stats"):
        run_attack(atk, make_model(), obs, INPUT_SHAPE, DEVICE, target=target)


def test_with_bn_statistics_computes_loss():
    """ThreatModel(has_bn_statistics=True) preserves bn_stats: the BN term contributes a finite, non-zero loss."""
    obs = make_observation(with_bn_stats=True)
    atk = make_attack_with_bn_loss(threat_model=ThreatModel(has_bn_statistics=True))
    target = TargetInfo(value=torch.tensor([3]))
    rec = _Components()

    state, _ = run_attack(atk, make_model(), obs, INPUT_SHAPE, DEVICE, target=target, callbacks=[rec])
    assert state.input is not None
    bn_terms = [step["bn_stats"] for step in rec.steps]
    assert bn_terms and all(math.isfinite(v) for v in bn_terms)
    assert any(v != 0.0 for v in bn_terms)


def test_no_threat_model_does_not_sanitize():
    """threat_model=None: the observation passes through unsanitized and BNStatsLoss uses its statistics."""
    obs = make_observation(with_bn_stats=True)
    atk = make_attack_with_bn_loss(threat_model=None)
    target = TargetInfo(value=torch.tensor([3]))
    rec = _Components()

    run_attack(atk, make_model(), obs, INPUT_SHAPE, DEVICE, target=target, callbacks=[rec])
    assert obs.bn_stats is not None and obs.batch_size == 4
    assert all("bn_stats" in step and math.isfinite(step["bn_stats"]) for step in rec.steps)


def test_grad_only_attack_unaffected_by_threat_model():
    """A gradient-matching-only attack works regardless of has_bn_statistics and has no BN term."""
    obs = make_observation(with_bn_stats=False)
    atk = make_attack_grad_only(threat_model=ThreatModel(has_bn_statistics=False))
    target = TargetInfo(value=torch.tensor([3]))
    rec = _Components()

    state, _ = run_attack(atk, make_model(), obs, INPUT_SHAPE, DEVICE, target=target, callbacks=[rec])
    assert state.input is not None
    assert rec.steps and all(set(step) == {"grad_match"} and math.isfinite(step["grad_match"]) for step in rec.steps)


# ── ClientObservation field tests ─────────────────────────────────────────────

def test_client_observation_fields():
    """ClientObservation has the expected fields; removed fields are gone."""
    obs = ClientObservation(
        gradients=[torch.randn(10)],
        bn_stats=list(BN_STATS),
        batch_size=4,
        loss_value=0.42,
        extra={"custom": "value"},
    )
    assert obs.bn_stats is not None
    assert obs.batch_size == 4
    assert obs.loss_value == 0.42
    assert obs.extra["custom"] == "value"
    assert not hasattr(obs, "model_weights")
    assert not hasattr(obs, "data_mean")
    assert not hasattr(obs, "data_std")
    assert not hasattr(obs, "num_images")


def test_client_observation_defaults():
    """ClientObservation with only gradients — all optional fields at defaults."""
    obs = ClientObservation(gradients=[torch.randn(10)])
    assert obs.bn_stats is None
    assert obs.batch_size is None
    assert obs.loss_value is None
    assert obs.extra == {}
