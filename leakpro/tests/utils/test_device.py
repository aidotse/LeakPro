#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for leakpro.utils.device.

Tests cover:
- require_hpu(): False only when _hthpu is None; raises HPUAcquisitionError
  for any installed-but-unusable case (is_available() False/raises, probe fails)
- get_device(): HPU path, CUDA path, CPU path, HPU-beats-CUDA priority, lru_cache,
  HPU acquisition failure raises instead of silently falling back
- mark_step(): no-op conditions (None htcore, eager mode, non-HPU device),
               actual call on HPU in lazy mode, default-arg fallback to get_device()
- hpu_import_error(): None and string cases
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

import leakpro.utils.device as device_module
from leakpro.utils.device import (
    HPUAcquisitionError,
    _detect_hpu_lazy_mode,
    get_device,
    hpu_import_error,
    mark_step,
    require_hpu,
)


@pytest.fixture(autouse=True)
def clear_device_cache(monkeypatch):
    """Clear the lru_cache on get_device before and after every test.

    Also clears LEAKPRO_DEVICE so a value exported in the shell (or left over from
    an override test) can't leak into unrelated tests that assert real detection.
    """
    monkeypatch.delenv("LEAKPRO_DEVICE", raising=False)
    get_device.cache_clear()
    yield
    get_device.cache_clear()


# ---------------------------------------------------------------------------
# require_hpu
# ---------------------------------------------------------------------------

class TestRequireHpu:
    def test_returns_false_when_hthpu_is_none(self):
        with patch.object(device_module, "_hthpu", None):
            assert require_hpu() is False

    def test_returns_true_when_is_available_true(self):
        mock_hthpu = MagicMock()
        mock_hthpu.is_available.return_value = True
        with patch.object(device_module, "_hthpu", mock_hthpu), \
             patch.object(device_module, "_probe_hpu_acquisition"):
            assert require_hpu() is True

    def test_raises_when_is_available_false(self):
        """habana_frameworks is installed but reports no device: that's an error, not CPU fallback."""
        mock_hthpu = MagicMock()
        mock_hthpu.is_available.return_value = False
        with patch.object(device_module, "_hthpu", mock_hthpu), \
             pytest.raises(HPUAcquisitionError):
            require_hpu()

    def test_raises_when_is_available_raises(self):
        mock_hthpu = MagicMock()
        mock_hthpu.is_available.side_effect = RuntimeError("HPU init failed")
        with patch.object(device_module, "_hthpu", mock_hthpu), \
             pytest.raises(HPUAcquisitionError):
            require_hpu()

    def test_raises_when_acquisition_fails_after_is_available_true(self):
        """is_available() can report True while the device still can't be acquired."""
        mock_hthpu = MagicMock()
        mock_hthpu.is_available.return_value = True
        with patch.object(device_module, "_hthpu", mock_hthpu), \
             patch.object(
                 device_module, "_probe_hpu_acquisition",
                 side_effect=RuntimeError("synStatus=8 [Device not found]"),
             ), \
             pytest.raises(HPUAcquisitionError):
            require_hpu()


# ---------------------------------------------------------------------------
# get_device  (lru_cache cleared by autouse fixture)
# ---------------------------------------------------------------------------

class TestGetDevice:
    def test_hpu_path(self):
        mock_hthpu = MagicMock()
        mock_hthpu.is_available.return_value = True
        with patch.object(device_module, "_hthpu", mock_hthpu), \
             patch.object(device_module, "_probe_hpu_acquisition"):
            device = get_device()
        assert device == torch.device("hpu")

    def test_cuda_path(self):
        with patch.object(device_module, "_hthpu", None), \
             patch("torch.cuda.is_available", return_value=True):
            device = get_device()
        assert device == torch.device("cuda")

    def test_cpu_path(self):
        with patch.object(device_module, "_hthpu", None), \
             patch("torch.cuda.is_available", return_value=False):
            device = get_device()
        assert device == torch.device("cpu")

    def test_hpu_takes_priority_over_cuda(self):
        mock_hthpu = MagicMock()
        mock_hthpu.is_available.return_value = True
        with patch.object(device_module, "_hthpu", mock_hthpu), \
             patch.object(device_module, "_probe_hpu_acquisition"), \
             patch("torch.cuda.is_available", return_value=True):
            device = get_device()
        assert device == torch.device("hpu")

    def test_hpu_acquisition_failure_raises_instead_of_falling_back(self):
        """habana_frameworks installed + unusable card must raise, never silently pick CUDA/CPU."""
        mock_hthpu = MagicMock()
        mock_hthpu.is_available.return_value = True
        with patch.object(device_module, "_hthpu", mock_hthpu), \
             patch.object(
                 device_module, "_probe_hpu_acquisition",
                 side_effect=RuntimeError("synStatus=8 [Device not found]"),
             ), \
             patch("torch.cuda.is_available", return_value=True), \
             pytest.raises(HPUAcquisitionError):
            get_device()

    def test_result_is_cached(self):
        with patch.object(device_module, "_hthpu", None), \
             patch("torch.cuda.is_available", return_value=False) as mock_cuda:
            get_device()
            get_device()
        # cuda.is_available is inside get_device's body; with the cache the
        # function body runs exactly once regardless of call count.
        assert mock_cuda.call_count == 1

    def test_override_set_after_first_call_is_not_ignored(self, monkeypatch):
        """Regression test: LEAKPRO_DEVICE set *after* the first get_device() call.

        This mirrors ``import leakpro`` calling get_device() once via its startup
        banner before user code has a chance to set LEAKPRO_DEVICE. A plain
        parameter-less ``@lru_cache`` on get_device() would return the stale
        pre-override result here instead of picking up the override.
        """
        with patch.object(device_module, "_hthpu", None), \
             patch("torch.cuda.is_available", return_value=False):
            first = get_device()  # no override yet -> normal detection -> cpu
        assert first == torch.device("cpu")

        monkeypatch.setenv("LEAKPRO_DEVICE", "cuda")
        with patch.object(device_module, "_hthpu", None), \
             patch("torch.cuda.is_available", return_value=True):
            second = get_device()
        assert second == torch.device("cuda")

    def test_override_cpu(self, monkeypatch):
        monkeypatch.setenv("LEAKPRO_DEVICE", "cpu")
        # Even with HPU and CUDA available, the override wins.
        mock_hthpu = MagicMock()
        mock_hthpu.is_available.return_value = True
        with patch.object(device_module, "_hthpu", mock_hthpu), \
             patch("torch.cuda.is_available", return_value=True):
            device = get_device()
        assert device == torch.device("cpu")

    def test_override_cuda(self, monkeypatch):
        monkeypatch.setenv("LEAKPRO_DEVICE", "cuda")
        with patch.object(device_module, "_hthpu", None), \
             patch("torch.cuda.is_available", return_value=True):
            device = get_device()
        assert device == torch.device("cuda")

    def test_override_cuda_raises_when_unavailable(self, monkeypatch):
        """An explicit request must never be silently downgraded."""
        monkeypatch.setenv("LEAKPRO_DEVICE", "cuda")
        with patch.object(device_module, "_hthpu", None), \
             patch("torch.cuda.is_available", return_value=False), \
             pytest.raises(RuntimeError):
            get_device()

    def test_override_hpu_raises_when_not_installed(self, monkeypatch):
        """habana_frameworks missing entirely must raise, not silently fall through."""
        monkeypatch.setenv("LEAKPRO_DEVICE", "hpu")
        with patch.object(device_module, "_hthpu", None), \
             patch("torch.cuda.is_available", return_value=False), \
             pytest.raises(HPUAcquisitionError):
            get_device()

    def test_override_hpu_raises_when_unusable(self, monkeypatch):
        """habana_frameworks installed but the card can't be acquired must raise."""
        monkeypatch.setenv("LEAKPRO_DEVICE", "hpu")
        mock_hthpu = MagicMock()
        mock_hthpu.is_available.return_value = False
        with patch.object(device_module, "_hthpu", mock_hthpu), \
             pytest.raises(HPUAcquisitionError):
            get_device()

    def test_override_hpu_succeeds_when_available(self, monkeypatch):
        monkeypatch.setenv("LEAKPRO_DEVICE", "hpu")
        mock_hthpu = MagicMock()
        mock_hthpu.is_available.return_value = True
        with patch.object(device_module, "_hthpu", mock_hthpu), \
             patch.object(device_module, "_probe_hpu_acquisition"):
            device = get_device()
        assert device == torch.device("hpu")

    def test_override_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("LEAKPRO_DEVICE", "CPU")
        with patch.object(device_module, "_hthpu", None), \
             patch("torch.cuda.is_available", return_value=True):
            device = get_device()
        assert device == torch.device("cpu")

    def test_invalid_override_falls_through_to_detection(self, monkeypatch):
        monkeypatch.setenv("LEAKPRO_DEVICE", "tpu")
        with patch.object(device_module, "_hthpu", None), \
             patch("torch.cuda.is_available", return_value=False):
            device = get_device()
        assert device == torch.device("cpu")


# ---------------------------------------------------------------------------
# _detect_hpu_lazy_mode
# ---------------------------------------------------------------------------

class TestDetectHpuLazyMode:
    def test_prefers_habana_is_lazy_true(self, monkeypatch):
        monkeypatch.delenv("PT_HPU_LAZY_MODE", raising=False)
        with patch.object(device_module, "_habana_is_lazy", lambda: True):
            assert _detect_hpu_lazy_mode() is True

    def test_prefers_habana_is_lazy_false(self, monkeypatch):
        """Habana's own helper wins even if the env var would suggest otherwise."""
        monkeypatch.setenv("PT_HPU_LAZY_MODE", "1")
        with patch.object(device_module, "_habana_is_lazy", lambda: False):
            assert _detect_hpu_lazy_mode() is False

    def test_falls_back_to_env_var_when_helper_unavailable(self, monkeypatch):
        monkeypatch.setenv("PT_HPU_LAZY_MODE", "1")
        with patch.object(device_module, "_habana_is_lazy", None):
            assert _detect_hpu_lazy_mode() is True

    def test_unset_env_var_defaults_to_eager_matching_habana(self, monkeypatch):
        """Regression test: this used to default to lazy=True, the opposite of
        Habana's own default, causing mark_step() to fire needlessly whenever
        PT_HPU_LAZY_MODE was left unset (the common case)."""
        monkeypatch.delenv("PT_HPU_LAZY_MODE", raising=False)
        with patch.object(device_module, "_habana_is_lazy", None):
            assert _detect_hpu_lazy_mode() is False

    def test_falls_back_to_env_var_when_helper_raises(self, monkeypatch):
        monkeypatch.setenv("PT_HPU_LAZY_MODE", "1")

        def _raise():
            raise RuntimeError("boom")

        with patch.object(device_module, "_habana_is_lazy", _raise):
            assert _detect_hpu_lazy_mode() is True


# ---------------------------------------------------------------------------
# mark_step
# ---------------------------------------------------------------------------

class TestMarkStep:
    def test_noop_when_htcore_is_none(self):
        with patch.object(device_module, "_htcore", None):
            mark_step()  # must not raise

    def test_noop_when_not_in_lazy_mode(self):
        mock_htcore = MagicMock()
        with patch.object(device_module, "_htcore", mock_htcore), \
             patch.object(device_module, "_HPU_LAZY_MODE", False):
            mark_step(torch.device("hpu"))
        mock_htcore.mark_step.assert_not_called()

    def test_noop_on_cpu_device(self):
        mock_htcore = MagicMock()
        with patch.object(device_module, "_htcore", mock_htcore), \
             patch.object(device_module, "_HPU_LAZY_MODE", True):
            mark_step(torch.device("cpu"))
        mock_htcore.mark_step.assert_not_called()

    def test_noop_on_cuda_device(self):
        mock_htcore = MagicMock()
        with patch.object(device_module, "_htcore", mock_htcore), \
             patch.object(device_module, "_HPU_LAZY_MODE", True):
            mark_step(torch.device("cuda"))
        mock_htcore.mark_step.assert_not_called()

    def test_calls_mark_step_on_hpu_in_lazy_mode(self):
        mock_htcore = MagicMock()
        with patch.object(device_module, "_htcore", mock_htcore), \
             patch.object(device_module, "_HPU_LAZY_MODE", True):
            mark_step(torch.device("hpu"))
        mock_htcore.mark_step.assert_called_once()

    def test_uses_get_device_when_no_argument_given(self):
        """mark_step() with no arg falls back to get_device(); should fire on HPU."""
        mock_htcore = MagicMock()
        mock_hthpu = MagicMock()
        mock_hthpu.is_available.return_value = True
        with patch.object(device_module, "_htcore", mock_htcore), \
             patch.object(device_module, "_hthpu", mock_hthpu), \
             patch.object(device_module, "_probe_hpu_acquisition"), \
             patch.object(device_module, "_HPU_LAZY_MODE", True):
            mark_step()
        mock_htcore.mark_step.assert_called_once()

    def test_noop_via_get_device_on_cpu(self):
        """mark_step() with no arg and CPU active must remain a no-op."""
        mock_htcore = MagicMock()
        with patch.object(device_module, "_htcore", mock_htcore), \
             patch.object(device_module, "_hthpu", None), \
             patch("torch.cuda.is_available", return_value=False), \
             patch.object(device_module, "_HPU_LAZY_MODE", True):
            mark_step()
        mock_htcore.mark_step.assert_not_called()


# ---------------------------------------------------------------------------
# hpu_import_error
# ---------------------------------------------------------------------------

class TestHpuImportError:
    def test_returns_none_when_no_error(self):
        with patch.object(device_module, "_HPU_IMPORT_ERROR", None):
            assert hpu_import_error() is None

    def test_returns_error_string(self):
        msg = "No module named 'habana_frameworks'"
        with patch.object(device_module, "_HPU_IMPORT_ERROR", msg):
            assert hpu_import_error() == msg
