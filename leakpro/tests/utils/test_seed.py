#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for leakpro.utils.seed.

Covers:
- seed_everything() does not depend on get_device()/HPU acquisition succeeding
- HPU RNG seeding is attempted only when habana_frameworks is installed
- a failure inside HPU seeding is caught and logged, not raised
- seeding is reproducible and leaves the cuDNN flags alone (issue #325)
- AttackScheduler applies audit.random_seed at audit start and before every attack
"""
import random
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from dotmap import DotMap
from pytest import LogCaptureFixture, MonkeyPatch

import leakpro.utils.seed as seed_module
from leakpro.attacks.attack_scheduler import AttackScheduler
from leakpro.tests.input_handler.image_input_handler import ImageInputHandler
from leakpro.utils.seed import seed_everything


def _mock_habana_hpu_module() -> MagicMock:
    """Build a fake habana_frameworks.torch.hpu module importable via sys.modules.

    ``import a.b.c as x`` binds ``x`` via attribute access on the top-level module
    object (equivalent to ``import a.b.c; x = a.b.c``), not just the sys.modules
    cache — so the parent mocks must hold the child as a real attribute too.
    """
    mock_hthpu = MagicMock()
    mock_torch = MagicMock(hpu=mock_hthpu)
    mock_top = MagicMock(torch=mock_torch)
    modules = {
        "habana_frameworks": mock_top,
        "habana_frameworks.torch": mock_torch,
        "habana_frameworks.torch.hpu": mock_hthpu,
    }
    return modules, mock_hthpu


class TestSeedEverything:
    def test_does_not_call_get_device(self):
        """Seeding must not go through get_device(), which can raise HPUAcquisitionError
        on a host where habana_frameworks is installed but no card can be acquired.

        Patches leakpro.utils.device.get_device to raise -- not just checking that
        hpu_is_installed() was called, which only shows seeding takes that path, not
        that it avoids get_device() entirely. If seed_everything() ever went through
        get_device() as well, this would surface as seed_everything() propagating the
        RuntimeError below instead of returning normally.
        """
        with patch.object(seed_module, "hpu_is_installed", return_value=False) as mock_installed, \
             patch("leakpro.utils.device.get_device", side_effect=RuntimeError("must not be called")):
            seed_everything(0)  # must not raise
        mock_installed.assert_called_once()

    def test_skips_hpu_seeding_when_not_installed(self):
        """habana_frameworks must never even be imported when hpu_is_installed() is False.

        Uses the same sys.modules-injection helper as the tests below (rather than
        ``patch("habana_frameworks.torch.hpu", create=True)``) because a string-target
        patch still has to import the parent ``habana_frameworks.torch`` module to
        resolve where to patch -- ``create=True`` only permits creating the final
        attribute, not skipping that import. That's harmless on a host with the real
        package installed, but raises ModuleNotFoundError on CI runners that
        (correctly) don't have it, which is exactly the case this test means to cover.
        """
        modules, mock_hthpu = _mock_habana_hpu_module()
        with patch.object(seed_module, "hpu_is_installed", return_value=False), \
             patch.dict("sys.modules", modules):
            seed_everything(0)
        mock_hthpu.manual_seed_all.assert_not_called()

    def test_seeds_hpu_when_installed(self):
        modules, mock_hthpu = _mock_habana_hpu_module()
        with patch.object(seed_module, "hpu_is_installed", return_value=True), \
             patch.dict("sys.modules", modules):
            seed_everything(0)
        mock_hthpu.manual_seed_all.assert_called_once_with(0)

    def test_hpu_seed_failure_is_caught_not_raised(self):
        modules, mock_hthpu = _mock_habana_hpu_module()
        mock_hthpu.manual_seed_all.side_effect = RuntimeError("synStatus=8")
        with patch.object(seed_module, "hpu_is_installed", return_value=True), \
             patch.dict("sys.modules", modules):
            seed_everything(0)  # must not raise


    def test_is_reproducible(self) -> None:
        """The same seed must reproduce the same numpy, torch and random draws."""
        seed_everything(1234)
        first = (np.random.rand(3).tolist(), torch.rand(3).tolist(), random.random())
        seed_everything(1234)
        second = (np.random.rand(3).tolist(), torch.rand(3).tolist(), random.random())
        assert first == second

    def test_leaves_cudnn_flags_alone(self) -> None:
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


class TestAuditSeeding:
    """The audit random_seed must reach the scheduler, not just sit in the config."""

    def test_seeds_audit_and_each_attack(
        self,
        image_handler: ImageInputHandler,
        monkeypatch: MonkeyPatch,
        tmp_path,  # noqa: ANN001
    ) -> None:
        """audit.random_seed is applied once at audit start and again before every attack."""
        image_handler.configs.audit.random_seed = 777

        seeds_applied = []
        monkeypatch.setattr("leakpro.attacks.attack_scheduler.seed_everything", seeds_applied.append)

        scheduler = AttackScheduler(image_handler, output_dir=str(tmp_path))
        assert seeds_applied == [777], "audit start must seed with audit.random_seed"

        scheduler.attacks = [_DummyAttack(), _DummyAttack()]
        scheduler.attack_names = ["dummy_a", "dummy_b"]
        scheduler.run_attacks()
        assert seeds_applied == [777, 777, 777], "every attack must re-seed with audit.random_seed"

    def test_defaults_seed_when_config_lacks_one(
        self,
        image_handler: ImageInputHandler,
        monkeypatch: MonkeyPatch,
        tmp_path,  # noqa: ANN001
    ) -> None:
        """Without random_seed in the config (e.g. a raw DotMap), fall back to the AuditConfig default."""
        audit_dict = image_handler.configs.audit.model_dump()
        audit_dict.pop("random_seed", None)
        image_handler.configs.audit = DotMap(audit_dict)

        seeds_applied = []
        monkeypatch.setattr("leakpro.attacks.attack_scheduler.seed_everything", seeds_applied.append)

        scheduler = AttackScheduler(image_handler, output_dir=str(tmp_path))
        assert scheduler.random_seed == 42
        assert seeds_applied == [42]

    def test_warns_and_defaults_on_non_int_seed(
        self,
        image_handler: ImageInputHandler,
        monkeypatch: MonkeyPatch,
        caplog: LogCaptureFixture,
        tmp_path,  # noqa: ANN001
    ) -> None:
        """A non-int random_seed (config typo) must warn rather than silently default."""
        audit_dict = image_handler.configs.audit.model_dump()
        audit_dict["random_seed"] = "42"
        image_handler.configs.audit = DotMap(audit_dict)

        seeds_applied = []
        monkeypatch.setattr("leakpro.attacks.attack_scheduler.seed_everything", seeds_applied.append)

        with caplog.at_level("WARNING"):
            scheduler = AttackScheduler(image_handler, output_dir=str(tmp_path))

        assert scheduler.random_seed == 42
        assert seeds_applied == [42]
        assert any("random_seed" in record.message and "42" in record.message for record in caplog.records)
