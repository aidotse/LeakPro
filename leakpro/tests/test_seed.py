#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Test seeding: seed_everything behavior and audit/attack seeding in the scheduler."""

import random

import numpy as np
import torch
from dotmap import DotMap
from pytest import MonkeyPatch

from leakpro.attacks.attack_scheduler import AttackScheduler
from leakpro.tests.input_handler.image_input_handler import ImageInputHandler
from leakpro.utils.seed import seed_everything


def test_seed_everything_is_reproducible() -> None:
    """The same seed must reproduce the same numpy, torch and random draws."""
    seed_everything(1234)
    first = (np.random.rand(3).tolist(), torch.rand(3).tolist(), random.random())
    seed_everything(1234)
    second = (np.random.rand(3).tolist(), torch.rand(3).tolist(), random.random())
    assert first == second


def test_seed_everything_leaves_cudnn_flags_alone() -> None:
    """Seeding must not force deterministic cuDNN or clear benchmark mode (issue #325)."""
    prev_deterministic = torch.backends.cudnn.deterministic
    prev_benchmark = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        seed_everything(1234)
        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cudnn.benchmark is True
    finally:
        torch.backends.cudnn.deterministic = prev_deterministic
        torch.backends.cudnn.benchmark = prev_benchmark


class _DummyResult:
    def save(self, **kwargs) -> None:  # noqa: ANN003
        pass


class _DummyAttack:
    optuna_params = 0
    bayesian_optimization = False

    def prepare_attack(self) -> None:
        pass

    def run_attack(self) -> _DummyResult:
        return _DummyResult()


def test_attack_scheduler_seeds_audit_and_each_attack(
    image_handler: ImageInputHandler,
    monkeypatch: MonkeyPatch,
    tmp_path,  # noqa: ANN001
) -> None:
    """The audit random_seed must be applied once at audit start and again before every attack."""
    image_handler.configs.audit.random_seed = 777

    seeds_applied = []
    monkeypatch.setattr("leakpro.attacks.attack_scheduler.seed_everything", seeds_applied.append)

    scheduler = AttackScheduler(image_handler, output_dir=str(tmp_path))
    assert seeds_applied == [777], "audit start must seed with audit.random_seed"

    scheduler.attacks = [_DummyAttack(), _DummyAttack()]
    scheduler.attack_names = ["dummy_a", "dummy_b"]
    scheduler.run_attacks()
    assert seeds_applied == [777, 777, 777], "every attack must re-seed with audit.random_seed"


def test_attack_scheduler_defaults_seed_when_config_lacks_one(
    image_handler: ImageInputHandler,
    monkeypatch: MonkeyPatch,
    tmp_path,  # noqa: ANN001
) -> None:
    """Without random_seed in the config (e.g. a raw DotMap), the scheduler falls back to the AuditConfig default."""
    audit_dict = image_handler.configs.audit.model_dump()
    audit_dict.pop("random_seed", None)
    image_handler.configs.audit = DotMap(audit_dict)

    seeds_applied = []
    monkeypatch.setattr("leakpro.attacks.attack_scheduler.seed_everything", seeds_applied.append)

    scheduler = AttackScheduler(image_handler, output_dir=str(tmp_path))
    assert scheduler.random_seed == 42
    assert seeds_applied == [42]
