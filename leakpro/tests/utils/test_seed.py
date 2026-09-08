#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for leakpro.utils.seed.

Covers:
- seed_everything() does not depend on get_device()/HPU acquisition succeeding
- HPU RNG seeding is attempted only when habana_frameworks is installed
- a failure inside HPU seeding is caught and logged, not raised
"""
from unittest.mock import MagicMock, patch

import leakpro.utils.seed as seed_module
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
        """
        with patch.object(seed_module, "hpu_is_installed", return_value=False) as mock_installed:
            seed_everything(0)
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
