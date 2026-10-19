#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for how the webapp worker serialises attack results for the UI."""

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="webapp extras (fastapi) not installed")

from leakpro.reporting.mia_result import MIAResult  # noqa: E402
from leakpro.utils.import_helper import Self  # noqa: E402


def _result(n: int, audit_indices: list | None) -> MIAResult:
    true = [1] * (n // 2) + [0] * (n - n // 2)
    return MIAResult.from_full_scores(true_membership=true, signal_values=np.arange(n, dtype=float),
                                      result_name="RMIA", audit_indices=audit_indices)


class TestSerialiseResult:
    """The Records tab gets population indices only when the attack reported them."""

    def test_attack_reported_indices_pass_through(self: Self) -> None:
        """audit_indices from MIAResult reach the UI unchanged, aligned with the signals."""
        from leakpro.webapp.backend.worker import _serialise_result

        out = _serialise_result(_result(5, [7, 3, 11, 5, 9]))
        assert out["audit_indices"] == [7, 3, 11, 5, 9]
        assert out["signal_values"] == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert out["true_labels"] == [1, 1, 0, 0, 0]

    def test_no_indices_means_none_not_positions(self: Self) -> None:
        """An attack that cannot say what it scored yields None; the worker never guesses."""
        from leakpro.webapp.backend.worker import _serialise_result

        assert _serialise_result(_result(5, None))["audit_indices"] is None

    def test_misaligned_indices_are_rejected_at_the_source(self: Self) -> None:
        """MIAResult refuses a mapping that does not line up with the signals."""
        with pytest.raises(ValueError, match="line up one-to-one"):
            _result(5, [7, 3, 11])
