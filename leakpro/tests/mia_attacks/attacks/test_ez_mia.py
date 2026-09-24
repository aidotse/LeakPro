#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for the EZ-MIA reduction (leakpro.attacks.mia_attacks.llm.ez_mia).

Tests cover:
- hand-worked P / N / EZ on a 2-row example under every aggregation
- position 0 of every row is always excluded (ignore_bos), even when it looks like error signal
- N == 0 and fewer-than-2-error-position rows score plain 0.0 under every aggregation, matching the
  paper's own released reference code (not forced above ordinary rows, contra the paper's appendix)
- log_ratio and ratio produce identical rankings on ordinary rows (monotone transform)
- padding never contributes to P or N
- end-to-end against the fake handler: a self-reference gives delta == 0 everywhere, so every row
  is degenerate (N == 0) and the result is still finite and well-formed
"""

import numpy as np
import pytest
import torch

from leakpro.attacks.mia_attacks.llm.ez_mia import AttackEZMIA, EZMIAConfig, ez_scores
from leakpro.reporting.mia_result import MIAResult
from leakpro.tests.mia_attacks.attacks.test_llm_base import _fake_handler

AGGREGATIONS = ("ratio", "log_ratio", "positive_fraction", "difference", "mean_delta", "median_delta")


def _example() -> tuple:
    """Two rows. Row 0: P=3, N=1. Row 1: P=0.5, N=2. Column 0 is a dummy, always non-error, so
    ignore_bos never changes these numbers; column 3 of row 0 is a non-error and must be ignored."""
    delta = np.array([[0.0, 2.0, -1.0, 9.0, 1.0],
                      [0.0, 0.5, -1.5, -0.5, 0.0]])
    error = np.array([[False, True, True, False, True],
                      [False, True, True, True, True]])
    return delta, error


def test_hand_worked_values_per_aggregation() -> None:
    """Each aggregation matches its definition on the worked example (no forced rows)."""
    delta, error = _example()
    expected = {
        "ratio": [3.0 / 1.0, 0.5 / 2.0],
        "log_ratio": [np.log(3.0), np.log(0.25)],
        "positive_fraction": [3.0 / 4.0, 0.5 / 2.5],
        "difference": [2.0, -1.5],
        "mean_delta": [(2.0 - 1.0 + 1.0) / 3, (0.5 - 1.5 - 0.5 + 0.0) / 4],
        "median_delta": [1.0, -0.25],
    }
    for agg, exp in expected.items():
        np.testing.assert_allclose(ez_scores(delta, error, agg), exp, err_msg=agg)


@pytest.mark.parametrize("aggregation", AGGREGATIONS)
def test_degenerate_rows_score_zero_under_every_aggregation(aggregation: str) -> None:
    """N == 0 (all upward) and fewer-than-2-error-position rows score plain 0.0 -- not forced above
    ordinary rows. This matches the paper's own released reference code, not its appendix prose."""
    delta = np.array([[0.0, 2.0, -1.0, 1.0],    # ordinary: P=3, N=1
                      [0.0, 1.0, 2.0, 0.5],     # N == 0 (all upward, 3 error positions) → degenerate
                      [0.0, 5.0, -5.0, 3.0],    # only 1 real error position (col 0 is ignore_bos) → degenerate
                      [0.0, 0.0, 0.0, 0.0],     # zero error positions → degenerate
                      [0.0, -1.0, -2.0, 0.5]])  # ordinary: P=0.5, N=3
    error = np.array([[False, True, True, True],
                      [False, True, True, True],
                      [False, True, False, False],
                      [False, False, False, False],
                      [False, True, True, True]])
    scores = ez_scores(delta, error, aggregation)
    assert np.all(np.isfinite(scores)), aggregation
    np.testing.assert_array_equal(scores[[1, 2, 3]], [0.0, 0.0, 0.0])
    assert scores[0] != 0.0, aggregation
    assert scores[4] != 0.0, aggregation


def test_log_ratio_and_ratio_rank_identically() -> None:
    """Log is monotone, so the two aggregations give the same ROC."""
    rng = np.random.default_rng(0)
    delta = rng.normal(size=(50, 30))
    error = rng.random((50, 30)) < 0.6
    delta[:5] = -np.abs(delta[:5])          # P == 0, N > 0: ratio 0, log_ratio -inf — must rank lowest in both
    a, b = ez_scores(delta, error, "ratio"), ez_scores(delta, error, "log_ratio")
    np.testing.assert_array_equal(np.argsort(a, kind="stable"), np.argsort(b, kind="stable"))
    assert np.all(a[:5] <= a[5:].min())
    assert np.all(b[:5] <= b[5:].min())


def test_padding_positions_do_not_contribute() -> None:
    """Values at masked-out positions (error == False) are ignored even when huge."""
    delta = np.array([[999.0, 1.0, -0.5, 1e6, -1e6]])
    error = np.array([[True, True, True, False, False]])
    np.testing.assert_allclose(ez_scores(delta, error, "ratio"), [2.0])


def test_ignore_bos_excludes_first_position() -> None:
    """Position 0 is always excluded from E, even when it is a large error-position delta."""
    delta = np.array([[100.0, 2.0, -1.0]])   # if column 0 counted, P would be huge instead of 2
    error = np.array([[True, True, True]])
    np.testing.assert_allclose(ez_scores(delta, error, "ratio"), [2.0])


def test_attack_end_to_end_with_self_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Self-reference makes delta == 0 → every row N == 0 → all forced; result must still be well-formed."""
    monkeypatch.setattr("leakpro.signals.token_evidence.get_device", lambda: torch.device("cpu"))
    handler = _fake_handler(n_train=4, n_test=3)
    attack = AttackEZMIA(handler, {"references": [{"source": "self"}], "batch_size": 4})
    assert len(attack.description()) == 4
    attack.prepare_attack()
    result = attack.run_attack()
    assert isinstance(result, MIAResult)
    assert result.signal_values.shape == (7,)
    assert np.all(np.isfinite(result.signal_values))
    assert result.metadata["aggregation"] == "ratio"
    assert result.metadata["references"][0]["source"] == "self"


def test_attack_requires_a_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a reference model prepare_attack fails with an actionable message."""
    monkeypatch.setattr("leakpro.signals.token_evidence.get_device", lambda: torch.device("cpu"))
    attack = AttackEZMIA(_fake_handler(), {})
    with pytest.raises(ValueError, match="references"):
        attack.prepare_attack()


def test_config_rejects_unknown_aggregation() -> None:
    """Config validation catches typos before any forward pass."""
    with pytest.raises(ValueError, match="aggregation"):
        EZMIAConfig(aggregation="p_over_n")
