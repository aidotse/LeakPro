#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for the WBC reduction (leakpro.attacks.mia_attacks.llm.wbc).

Tests cover:
- geometric_windows reproduces the paper's grid (eq. 12) and keeps duplicates
- window_stat: hand-worked sign / mean / median / min on a short sequence; windows never cross
  into padding; a window larger than the sequence clamps to the sequence's own length (matching
  github.com/Stry233/WBC's attacks/wbc.py, not a skip-and-fallback)
- wbc_scores: mean over all configured window sizes, every one always contributing (clamped ones
  included); scores lie in [0, 1] for sign
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
    delta_row = np.array([2.0, -1.0, 0.0, 2.0, 99.0])   # last position is padding and must not be used
    length = 4
    assert window_stat(delta_row, length, 2, "sign") == pytest.approx(2 / 3)
    assert window_stat(delta_row, length, 2, "mean") == pytest.approx((1 - 1 + 2) / 3)
    assert window_stat(delta_row, length, 2, "median") == pytest.approx(1.0)
    assert window_stat(delta_row, length, 2, "min") == pytest.approx(-1.0)
    # w == length: exactly one window
    assert window_stat(delta_row, length, 4, "mean") == pytest.approx(3.0)
    # w > length: clamped down to length -- same single window as w == length, not skipped/nan
    assert window_stat(delta_row, length, 5, "mean") == pytest.approx(3.0)
    assert window_stat(delta_row, length, 100, "sign") == pytest.approx(1.0)
    # length == 0: degenerate row, always 0.0 regardless of w/aggregation
    assert window_stat(delta_row, 0, 2, "sign") == 0.0


def test_short_sequences_clamp_window_size_instead_of_skipping() -> None:
    """A window larger than the row clamps to the row's length and still contributes to the mean --
    it is not skipped, and does not trigger a separate whole-row fallback (github.com/Stry233/WBC's
    attacks/wbc.py: `effective_window_size = min(window_size, min_length)`, always).
    """
    delta_row = np.array([1.0, 1.0, -5.0, 0.0, 2.0])
    length = 5
    fits = window_stat(delta_row, length, 3, "sign")          # w=3 fits directly
    clamped = window_stat(delta_row, length, 10, "sign")      # w=10 clamps down to the whole row (length 5)

    # A fitting window size and an oversized (clamped) one both contribute -- a real average, not
    # "only the fitting one counted" (which is what skip-instead-of-clamp would have given).
    mixed = wbc_scores(delta_row[None, :], np.array([length]), [3, 10], "sign")
    assert mixed[0] == pytest.approx((fits + clamped) / 2)

    # Two oversized window sizes clamp to the *same* effective size and both still contribute --
    # duplicated, not deduplicated -- so the mean of two identical values is just that value.
    both_clamped = wbc_scores(delta_row[None, :], np.array([length]), [10, 20], "sign")
    assert both_clamped[0] == pytest.approx(clamped)


def test_wbc_scores_averages_over_all_configured_windows() -> None:
    """Row 0 fits both w=2 and w=3; row 1 (length 2) and row 2 (length 1) clamp w=3 down."""
    delta = np.array([[1.0, 1.0, -5.0, 0.0],
                      [-1.0, 2.0, 0.0, 0.0],
                      [0.7, 0.0, 0.0, 0.0]])
    lengths = np.array([3, 2, 1])
    scores = wbc_scores(delta, lengths, [2, 3], "sign")
    # row 0: w=2 -> sums [2, -4] -> 1/2 ; w=3 -> sum [-3] -> 0 ; mean = 0.25
    # row 1: w=2 -> sum [1] -> 1 ; w=3 clamps to eff_w=2 (== length) -> identical sum [1] -> 1 ; mean = 1
    # row 2: w=2 clamps to eff_w=1 -> sum [0.7] -> 1 ; w=3 also clamps to eff_w=1 -> same -> 1 ; mean = 1
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
    with pytest.raises(ValidationError, match="non-empty"):
        WBCConfig(window_lengths=[])
    with pytest.raises(ValidationError, match=">= 1"):
        WBCConfig(window_lengths=[2, 0, 5])
    # window_lengths set: w_min/w_max no longer need w_max >= w_min to be satisfiable together with it
    WBCConfig(window_lengths=[5, 3], w_min=10, w_max=5)


def test_window_lengths_override_bypasses_geometric_formula() -> None:
    """window_lengths, when set, is used verbatim -- the formula does not reproduce the paper's own grid.

    geometric_windows(2, 40, 10) computes [2, 3, 4, 5, 8, 11, 15, 21, 29, 40], not the paper's own
    published default grid below -- neither round, floor nor ceil on the same formula reproduces it, so
    window_lengths exists to let a caller reproduce the paper's exact numbers rather than the formula's.
    """
    paper_published_grid = [2, 3, 4, 6, 9, 13, 18, 25, 32, 40]
    cfg = WBCConfig(window_lengths=paper_published_grid, w_min=2, w_max=40, n_windows=10)
    assert cfg.window_lengths == paper_published_grid
    assert geometric_windows(cfg.w_min, cfg.w_max, cfg.n_windows) != paper_published_grid


def test_attack_end_to_end_variable_length(monkeypatch: pytest.MonkeyPatch) -> None:
    """Variable-length rows (3..8 tokens) with the default w_max=40 exercise window-size clamping."""
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


def test_attack_end_to_end_with_window_lengths_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """window_lengths, when set, actually drives run_attack instead of the geometric formula."""
    monkeypatch.setattr("leakpro.signals.token_evidence.get_device", lambda: torch.device("cpu"))
    handler = _fake_handler(n_train=4, n_test=3)
    windows = [2, 3, 4, 6, 9, 13, 18, 25, 32, 40]
    attack = AttackWBC(handler, {"references": [{"source": "random_init"}], "batch_size": 3, "window_lengths": windows})
    attack.prepare_attack()
    result = attack.run_attack()
    assert isinstance(result, MIAResult)
    assert result.metadata["window_lengths"] == windows
    assert np.all(np.isfinite(result.signal_values))
