#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for leakpro.utils.device.

Tests cover:
- is_hpu_available(): None _hthpu, runtime exception, normal True/False
- get_device(): HPU path, CUDA path, CPU path, HPU-beats-CUDA priority, lru_cache
- mark_step(): no-op conditions (None htcore, eager mode, non-HPU device),
               actual call on HPU in lazy mode, default-arg fallback to get_device()
- hpu_import_error(): None and string cases
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

import leakpro.utils.device as device_module
from leakpro.utils.device import get_device, hpu_import_error, is_hpu_available, mark_step


@pytest.fixture(autouse=True)
def clear_device_cache():
    """Clear the lru_cache on get_device before and after every test."""
    get_device.cache_clear()
    yield
    get_device.cache_clear()


# ---------------------------------------------------------------------------
# is_hpu_available
# ---------------------------------------------------------------------------

class TestIsHpuAvailable:
    def test_returns_false_when_hthpu_is_none(self):
        with patch.object(device_module, "_hthpu", None):
            assert is_hpu_available() is False

    def test_returns_true_when_is_available_true(self):
        mock_hthpu = MagicMock()
        mock_hthpu.is_available.return_value = True
        with patch.object(device_module, "_hthpu", mock_hthpu):
            assert is_hpu_available() is True

    def test_returns_false_when_is_available_false(self):
        mock_hthpu = MagicMock()
        mock_hthpu.is_available.return_value = False
        with patch.object(device_module, "_hthpu", mock_hthpu):
            assert is_hpu_available() is False

    def test_returns_false_when_is_available_raises(self):
        mock_hthpu = MagicMock()
        mock_hthpu.is_available.side_effect = RuntimeError("HPU init failed")
        with patch.object(device_module, "_hthpu", mock_hthpu):
            assert is_hpu_available() is False


# ---------------------------------------------------------------------------
# get_device  (lru_cache cleared by autouse fixture)
# ---------------------------------------------------------------------------

class TestGetDevice:
    def test_hpu_path(self):
        mock_hthpu = MagicMock()
        mock_hthpu.is_available.return_value = True
        with patch.object(device_module, "_hthpu", mock_hthpu):
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
             patch("torch.cuda.is_available", return_value=True):
            device = get_device()
        assert device == torch.device("hpu")

    def test_result_is_cached(self):
        with patch.object(device_module, "_hthpu", None), \
             patch("torch.cuda.is_available", return_value=False) as mock_cuda:
            get_device()
            get_device()
        # cuda.is_available is inside get_device's body; with the cache the
        # function body runs exactly once regardless of call count.
        assert mock_cuda.call_count == 1

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
        with patch.object(device_module, "_hthpu", None):
            device = get_device()
        assert device == torch.device("cuda")

    def test_override_hpu(self, monkeypatch):
        monkeypatch.setenv("LEAKPRO_DEVICE", "hpu")
        with patch.object(device_module, "_hthpu", None), \
             patch("torch.cuda.is_available", return_value=False):
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
