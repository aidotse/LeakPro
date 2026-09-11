#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for the WBC reduction (leakpro.attacks.mia_attacks.llm.wbc).

Tests cover:
- geometric_windows reproduces the paper's grid (eq. 12) and keeps duplicates
- window_stat: hand-worked sign / mean / median / min on a short sequence; windows never cross
  into padding; a window larger than the sequence yields nan
- wbc_scores: mean over fitting windows only; a row too short for every window falls back to one
  whole-sequence window; scores lie in [0, 1] for sign
- the WBC delta equals EZ-MIA's delta (paper's l^R - l^T over losses == lp^T - lp^R over log-probs)
- config validation (w_max >= w_min, n_windows >= 2)
- end-to-end against the fake handler with variable-length rows
"""

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from leakpro.attacks.mia_attacks.llm.wbc import AttackWBC, WBCConfig, geometric_windows, wbc_scores, window_stat
from leakpro.reporting.mia_result import MIAResult
from leakpro.tests.mia_attacks.attacks.test_llm_base import _fake_handler


def test_geometric_windows_matches_paper_defaults_and_keeps_duplicates() -> None:
    """w_min=2, w_max=40, |W|=10 → the paper's grid; a degenerate range yields repeated sizes."""
    ws = geometric_windows(2, 40, 10)
    assert len(ws) == 10
    assert ws[0] == 2
    assert ws[-1] == 40
    assert ws == sorted(ws)
    assert geometric_windows(3, 3, 4) == [3, 3, 3, 3]
    assert geometric_windows(1, 2, 5) == [1, 1, 1, 2, 2]   # rounding produces duplicates; they are kept


def test_window_stat_hand_worked() -> None:
    """One row, length 4 (+1 padding), w=2: windows [0,1],[1,2],[2,3] with sums 1, -1, 2."""
    delta = np.array([[2.0, -1.0, 0.0, 2.0, 99.0]])   # last position is padding and must not be used
    lengths = np.array([4])
    np.testing.assert_allclose(window_stat(delta, lengths, 2, "sign"), [2 / 3])
    np.testing.assert_allclose(window_stat(delta, lengths, 2, "mean"), [(1 - 1 + 2) / 3])
    np.testing.assert_allclose(window_stat(delta, lengths, 2, "median"), [1.0])
    np.testing.assert_allclose(window_stat(delta, lengths, 2, "min"), [-1.0])
    # w == length: exactly one window
    np.testing.assert_allclose(window_stat(delta, lengths, 4, "mean"), [3.0])
    # w > length: no window fits
    assert np.isnan(window_stat(delta, lengths, 5, "sign"))[0]
    # whole-sequence fallback
    np.testing.assert_allclose(window_stat(delta, lengths, None, "sign"), [1.0])
    np.testing.assert_allclose(window_stat(delta, lengths, None, "mean"), [3.0])


def test_wbc_scores_skips_unfitting_windows_and_falls_back_for_short_rows() -> None:
    """Row 0 fits w=2 and w=3; row 1 (length 2) fits only w=2; row 2 (length 1) fits nothing."""
    delta = np.array([[1.0, 1.0, -5.0, 0.0],
                      [-1.0, 2.0, 0.0, 0.0],
                      [0.7, 0.0, 0.0, 0.0]])
    lengths = np.array([3, 2, 1])
    scores = wbc_scores(delta, lengths, [2, 3], "sign")
    # row 0: w=2 → sums [2, -4] → 1/2 ; w=3 → sum [-3] → 0 ; mean = 0.25
    # row 1: w=2 → sum [1] → 1 ; w=3 doesn't fit → skipped ; mean = 1
    # row 2: nothing fits → single whole window: 0.7 > 0 → 1
    np.testing.assert_allclose(scores, [0.25, 1.0, 1.0])
    assert np.all((scores >= 0) & (scores <= 1))


def test_delta_convention_matches_ez_mia() -> None:
    """Paper WBC: Delta = l^R - l^T with l = -log p. That is lp^T - lp^R, EZ-MIA's delta."""
    rng = np.random.default_rng(1)
    lp_t, lp_r = rng.normal(size=(4, 6)), rng.normal(size=(4, 6))
    loss_t, loss_r = -lp_t, -lp_r
    np.testing.assert_allclose(loss_r - loss_t, lp_t - lp_r)


def test_config_validation() -> None:
    """Grid constraints are enforced before any forward pass."""
    assert WBCConfig().w_min == 2
    with pytest.raises(ValidationError, match="w_max"):
        WBCConfig(w_min=10, w_max=5)
    with pytest.raises(ValidationError):
        WBCConfig(n_windows=1)
    with pytest.raises(ValidationError):
        WBCConfig(aggregation="max")


def test_attack_end_to_end_variable_length(monkeypatch: pytest.MonkeyPatch) -> None:
    """Variable-length rows (3..8 tokens) with the default w_max=40 exercise the fallback path."""
    monkeypatch.setattr("leakpro.signals.token_evidence.get_device", lambda: torch.device("cpu"))
    handler = _fake_handler(n_train=4, n_test=3)
    attack = AttackWBC(handler, {"references": [{"source": "random_init"}], "batch_size": 3})
    assert len(attack.description()) == 4
    attack.prepare_attack()
    result = attack.run_attack()
    assert isinstance(result, MIAResult)
    assert result.signal_values.shape == (7,)
    assert np.all(np.isfinite(result.signal_values))
    assert np.all((result.signal_values >= 0) & (result.signal_values <= 1))
    assert result.metadata["w_min"] == 2
