#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for how the webapp worker serialises attack results for the UI."""

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="webapp extras (fastapi) not installed")

from leakpro.utils.import_helper import Self


class _Result:
    """Minimal stand-in for MIAResult with the fields the worker reads."""

    def __init__(self: Self, n: int) -> None:
        self.result_name = "RMIA"
        self.roc_auc = 0.5
        self.fixed_fpr_table = {"TPR@1%FPR": 0.01}
        self.fpr = np.linspace(0, 1, n)
        self.tpr = np.linspace(0, 1, n)
        self.signal_values = np.arange(n, dtype=float)
        self.true = np.array([1] * (n // 2) + [0] * (n - n // 2))


class _Handler:
    def __init__(self: Self, train: list, test: list) -> None:
        self.train_indices = np.array(train)
        self.test_indices = np.array(test)


class TestAuditIndices:
    """The Records tab needs population indices, not audit-set positions."""

    def test_indices_follow_audit_set_order(self: Self) -> None:
        """train_indices first, then test_indices — the order every MIA attack scores in."""
        from leakpro.webapp.backend.worker import _audit_population_indices

        assert _audit_population_indices(_Handler([7, 3], [11, 5, 9])) == [7, 3, 11, 5, 9]

    def test_missing_handler_indices_gives_none(self: Self) -> None:
        """A handler without index attributes yields no mapping rather than an error."""
        from leakpro.webapp.backend.worker import _audit_population_indices

        assert _audit_population_indices(object()) is None

    def test_serialised_result_carries_matching_indices(self: Self) -> None:
        """A one-to-one mapping is passed through next to the signals."""
        from leakpro.webapp.backend.worker import _serialise_result

        out = _serialise_result(_Result(5), [7, 3, 11, 5, 9])
        assert out["audit_indices"] == [7, 3, 11, 5, 9]
        assert out["signal_values"] == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_length_mismatch_drops_indices(self: Self) -> None:
        """An attack that subsamples the audit set must not get a wrong mapping."""
        from leakpro.webapp.backend.worker import _serialise_result

        assert _serialise_result(_Result(5), [7, 3, 11])["audit_indices"] is None
        assert _serialise_result(_Result(5), None)["audit_indices"] is None
