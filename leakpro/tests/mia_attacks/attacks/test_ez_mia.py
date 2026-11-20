#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for the EZ-MIA reduction (leakpro.attacks.mia_attacks.llm.ez_mia).

Tests cover:
- hand-worked P / N / EZ on a 2-row example under every aggregation
- position 0 of every row is excluded by default (ignore_first_position), even when it looks like
  error signal, and is configurable
- zero error positions (E.5) rank as members under every aggregation; N == 0 with P > 0 (E.6) ranks
  as a member only for aggregations where that is genuinely undefined/infinite (ratio, log_ratio) --
  not for positive_fraction/difference/mean_delta/median_delta, which have ordinary finite values there
- rows with 1 <= n_err < min_error_positions that are not otherwise forced score the bottom of the
  aggregation's range under every aggregation -- in particular a lone *downward* error (the attack's
  own non-member signal) must not outrank a genuine member, nor even a P < N row
- error positions with no movement at all (P == N == 0) are not forced (no E.6 evidence) and, under
  the ratio family where 0 / 0 is undefined, score the bottom of the range instead of raising
- among forced rows, zero-error-position rows (E.5) always rank above N==0 rows (E.6), tiebroken by P
  within each bucket -- not by row position, so a batch of genuinely tied rows (e.g. a self-reference
  audit) gets equal scores regardless of input order
- ratio, log_ratio and positive_fraction (monotone in P / N) produce identical rankings, ties included,
  with floored rows anywhere in the input
- padding never contributes to P or N
- end-to-end against the fake handler: a self-reference gives delta == 0 everywhere, so every row with
  an error position ties and the result is still finite and well-formed
"""

import numpy as np
import pytest
import torch

from leakpro.attacks.mia_attacks.llm.ez_mia import AttackEZMIA, EZMIAConfig, ez_scores
from leakpro.reporting.mia_result import MIAResult
from leakpro.tests.mia_attacks.attacks.test_llm_base import _fake_handler

AGGREGATIONS = ("ratio", "log_ratio", "positive_fraction", "difference", "mean_delta", "median_delta")
UNDEFINED_AT_N_ZERO = ("ratio", "log_ratio")


def _example() -> tuple:
    """Two rows. Row 0: P=3, N=1. Row 1: P=0.5, N=2. Column 0 is a dummy, always non-error, so
    ignore_first_position never changes these numbers; column 3 of row 0 is a non-error and must be ignored."""
    delta = np.array([[0.0, 2.0, -1.0, 9.0, 1.0],
                      [0.0, 0.5, -1.5, -0.5, 0.0]])
    error = np.array([[False, True, True, False, True],
                      [False, True, True, True, True]])
    return delta, error


def test_hand_worked_values_per_aggregation() -> None:
    """Each aggregation matches its definition on the worked example (no forced/insufficient rows)."""
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
        result = ez_scores(delta, error, agg)
        np.testing.assert_allclose(result.scores, exp, err_msg=agg)
        assert not result.forced.any(), agg
        assert not result.insufficient.any(), agg


@pytest.mark.parametrize("aggregation", AGGREGATIONS)
def test_zero_error_positions_ranks_as_member_under_every_aggregation(aggregation: str) -> None:
    """n_err == 0 (paper E.5) is forced above all ordinary rows regardless of aggregation."""
    delta = np.array([[0.0, 2.0, -1.0, 1.0],    # ordinary: P=3, N=1
                      [0.0, 0.0, 0.0, 0.0],     # zero error positions -> forced
                      [0.0, -1.0, -2.0, 0.5]])  # ordinary: P=0.5, N=3
    error = np.array([[False, True, True, True],
                      [False, False, False, False],
                      [False, True, True, True]])
    result = ez_scores(delta, error, aggregation)
    assert np.all(np.isfinite(result.scores)), aggregation
    np.testing.assert_array_equal(result.forced, [False, True, False], err_msg=aggregation)
    assert not result.insufficient.any(), aggregation
    assert result.scores[1] > max(result.scores[0], result.scores[2]), aggregation


@pytest.mark.parametrize("aggregation", UNDEFINED_AT_N_ZERO)
def test_n_zero_ranks_as_member_only_where_undefined(aggregation: str) -> None:
    """N == 0 with enough error positions (E.6) is forced only for ratio/log_ratio, where P/N is
    genuinely undefined/infinite -- not a general 'too little evidence' rule."""
    delta = np.array([[0.0, 2.0, -1.0, 1.0],   # ordinary: P=3, N=1
                      [0.0, 1.0, 2.0, 0.5]])   # N == 0, 3 error positions -> forced here
    error = np.array([[False, True, True, True],
                      [False, True, True, True]])
    result = ez_scores(delta, error, aggregation)
    np.testing.assert_array_equal(result.forced, [False, True], err_msg=aggregation)
    assert not result.insufficient.any(), aggregation
    assert result.scores[1] > result.scores[0], aggregation


@pytest.mark.parametrize("aggregation", ("positive_fraction", "difference", "mean_delta", "median_delta"))
def test_n_zero_is_not_forced_where_it_has_an_ordinary_finite_value(aggregation: str) -> None:
    """For aggregations where N == 0 is not degenerate, it must not be forced to the top."""
    delta = np.array([[0.0, 1.0, 2.0, 0.5]])  # N == 0, 3 error positions
    error = np.array([[False, True, True, True]])
    result = ez_scores(delta, error, aggregation)
    assert not result.forced[0], aggregation
    assert not result.insufficient[0], aggregation
    assert np.isfinite(result.scores[0]), aggregation


@pytest.mark.parametrize("aggregation", AGGREGATIONS)
def test_no_movement_at_error_positions_is_not_forced(aggregation: str) -> None:
    """P == N == 0 with error positions is 0 / 0 -- no evidence, not E.6 (which needs P > 0). It must not
    be forced above a real member; under the ratio family it scores the bottom instead of raising on nan."""
    delta = np.zeros((3, 8))
    error = np.zeros((3, 8), bool)
    error[0, 2:5] = True                       # P = N = 0
    error[1, 2:5] = True
    delta[1, 2:5] = [5.0, 5.0, -0.01]          # P / N = 1000
    error[2, 2:5] = True
    delta[2, 2:5] = [0.1, -1.0, -1.0]          # P < N
    result = ez_scores(delta, error, aggregation)
    assert not result.forced.any(), aggregation
    assert np.all(np.isfinite(result.scores)), aggregation
    assert result.scores[1] > result.scores[0], aggregation
    if aggregation in ("ratio", "log_ratio", "positive_fraction"):
        assert result.insufficient[0], aggregation
        assert result.scores[0] < result.scores[2], aggregation


@pytest.mark.parametrize("aggregation", AGGREGATIONS)
def test_lone_downward_error_does_not_outrank_a_real_member(aggregation: str) -> None:
    """A single error whose mass moved entirely downward (P == 0, the attack's own non-member
    signal) must rank below every ordinary row with P > 0 -- including P < N ones -- under every
    aggregation, not at 0.0, which for log_ratio/difference/mean_delta/median_delta means P == N."""
    delta = np.zeros((4, 8))
    error = np.zeros((4, 8), bool)
    error[0, 2] = True
    delta[0, 2] = -5.0                       # 1 error position, P = 0, N = 5 -- insufficient, not forced
    error[1, 2:5] = True
    delta[1, 2:5] = [1.0, 1.0, -0.1]         # ordinary member: P / N = 20
    error[2, 2:5] = True
    delta[2, 2:5] = [0.1, -1.0, -1.0]        # ordinary, P < N
    error[3, 2:5] = True
    delta[3, 2:5] = [1.0, -1.0, -0.5]        # ordinary, P / N = 0.67
    result = ez_scores(delta, error, aggregation)
    np.testing.assert_array_equal(result.forced, [False] * 4, err_msg=aggregation)
    np.testing.assert_array_equal(result.insufficient, [True, False, False, False], err_msg=aggregation)
    assert result.scores[0] < result.scores[1:].min(), aggregation


def test_min_error_positions_is_configurable() -> None:
    """Raising min_error_positions marks insufficient a row that would otherwise be ordinary
    (N != 0, 2 errors) -- it is scored the bottom (0.0 for ratio), not forced to the top."""
    delta = np.array([[0.0, 5.0, -3.0, 1.0]])  # 2 real error positions (col 0 excluded): P=5, N=3
    error = np.array([[False, True, True, False]])
    low = ez_scores(delta, error, "ratio", min_error_positions=2)
    assert not low.forced[0] and not low.insufficient[0]
    high = ez_scores(delta, error, "ratio", min_error_positions=3)
    assert not high.forced[0] and high.insufficient[0]
    assert high.scores[0] == 0.0


def test_forced_tiebreak_is_lexicographic_zero_errors_above_n_zero() -> None:
    """Among forced rows, zero-error-position rows (E.5, the paper's strongest case) always rank
    above N==0 rows (E.6), regardless of P -- a tiebreak on P alone can't express "no errors at all"."""
    delta = np.array([[0.0, 0.0, 0.0, 0.0],     # zero error positions, P = 0 -- E.5
                      [0.0, 50.0, 10.0, 5.0]])  # N == 0, P = 65 -- E.6, much larger P than row 0
    error = np.array([[False, False, False, False],
                      [False, True, True, True]])
    result = ez_scores(delta, error, "ratio")
    np.testing.assert_array_equal(result.forced, [True, True])
    assert result.scores[0] > result.scores[1]  # E.5 outranks E.6 despite P = 0 < 65


def test_tied_forced_rows_get_equal_scores_regardless_of_order() -> None:
    """Genuinely tied forced rows (equal P, equal E.5/E.6 bucket) must get equal scores -- not an
    ordering that depends on row position (which in AbstractLLMMIA's output correlates with label)."""
    delta = np.zeros((8, 6))
    error = np.zeros((8, 6), bool)  # every row: zero error positions -> all E.5, all P = 0
    result = ez_scores(delta, error, "ratio")
    assert result.forced.all()
    np.testing.assert_allclose(result.scores, result.scores[0])


def _dense_ranks(scores: np.ndarray) -> np.ndarray:
    """Rank by distinct value, so tied scores share a rank -- argsort would hide a tie vs strict order."""
    return np.unique(scores, return_inverse=True)[1]


def test_ratio_family_ranks_identically() -> None:
    """ratio, log_ratio and positive_fraction are monotone in P / N, so they must give the same ranking,
    ties included. Floored rows (lone downward error, no movement) sit mid-array, not at index 0, so a
    stable argsort can't paper over a tie-vs-strict difference with ordinary P == 0 rows."""
    rng = np.random.default_rng(0)
    delta = rng.normal(size=(50, 30))
    error = rng.random((50, 30)) < 0.6
    delta[:5] = -np.abs(delta[:5])          # P == 0, N > 0: ratio 0, log_ratio -inf -- the bottom in all three
    error[20] = False                       # lone downward error -- insufficient
    error[20, 5] = True
    delta[20, 5] = -3.0
    error[30] = False                       # lone upward error -- N == 0, P > 0: E.6 under ratio/log_ratio only
    error[30, 5] = True
    delta[30, 5] = 2.0
    delta[40] = 0.0                         # error positions but no movement: P == N == 0 -- insufficient
    scores = {a: ez_scores(delta, error, a).scores for a in ("ratio", "log_ratio")}
    np.testing.assert_array_equal(_dense_ranks(scores["ratio"]), _dense_ranks(scores["log_ratio"]))
    for s in scores.values():
        floor = s[[0, 1, 2, 3, 4, 20, 40]]
        np.testing.assert_allclose(floor, floor[0])                  # floored rows tie with P == 0 rows
        assert floor[0] < np.delete(s, [0, 1, 2, 3, 4, 20, 40]).min()

    # positive_fraction doesn't force E.6 (it peaks at 1.0 there), so compare it with row 30 removed.
    keep = np.arange(50) != 30
    pf = ez_scores(delta[keep], error[keep], "positive_fraction").scores
    np.testing.assert_array_equal(_dense_ranks(pf), _dense_ranks(scores["ratio"][keep]))

def test_padding_positions_do_not_contribute() -> None:
    """Values at masked-out positions (error == False) are ignored even when huge."""
    delta = np.array([[999.0, 1.0, -0.5, 1e6, -1e6]])
    error = np.array([[True, True, True, False, False]])
    np.testing.assert_allclose(ez_scores(delta, error, "ratio").scores, [2.0])


def test_ignore_first_position_excludes_first_position() -> None:
    """Position 0 is excluded from E by default, even when it is a large error-position delta."""
    delta = np.array([[100.0, 2.0, -1.0]])   # if column 0 counted, P would be huge instead of 2
    error = np.array([[True, True, True]])
    np.testing.assert_allclose(ez_scores(delta, error, "ratio").scores, [2.0])


def test_ignore_first_position_is_configurable() -> None:
    """With ignore_first_position=False, column 0 counts like any other error position."""
    delta = np.array([[100.0, 2.0, -1.0]])
    error = np.array([[True, True, True]])
    np.testing.assert_allclose(ez_scores(delta, error, "ratio", ignore_first_position=False).scores, [102.0])


def test_attack_end_to_end_with_self_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Self-reference makes delta == 0 → P == N == 0 on every row; result must still be well-formed."""
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

    # delta == 0 removes all reference signal: every row with an error position is tied (P == N == 0).
    # Only zero-error rows (E.5) break the tie -- that comes from the target's own predictions, not the
    # reference -- so the AUC is the pairwise definition over these scores, ties counted as 1/2.
    target = attack.evidence_set.target
    error = (target.argmax != target.token_ids) & target.mask
    error[:, 0] = False
    has_errors = error.sum(axis=1) > 0
    assert has_errors.any() and (~has_errors).any()  # the fake data exercises both kinds of row
    tied = result.signal_values[has_errors]
    np.testing.assert_allclose(tied, tied[0])
    assert np.all(result.signal_values[~has_errors] > tied[0])
    labels = attack._audit_labels
    pos, neg = result.signal_values[labels == 1], result.signal_values[labels == 0]
    expected_auc = ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (pos.size * neg.size)
    assert np.isclose(result.roc_auc, expected_auc)


def test_self_reference_without_error_free_rows_gives_auc_one_half() -> None:
    """Self-reference (delta == 0) where every row has an error position: all rows tie, so AUC is 0.5."""
    labels = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    delta = np.zeros((8, 5))
    error = np.zeros((8, 5), bool)
    error[:, 2] = True
    scores = ez_scores(delta, error, "ratio").scores
    # A second, distinct score level so MIAResult builds a ROC (one level alone has no curve); it holds
    # one member and one non-member, so it adds no signal either.
    labels = np.r_[labels, 1, 0]
    scores = np.r_[scores, scores[0] - 1.0, scores[0] - 1.0]
    result = MIAResult.from_full_scores(true_membership=labels, signal_values=scores, result_name="self")
    assert np.isclose(result.roc_auc, 0.5)


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
