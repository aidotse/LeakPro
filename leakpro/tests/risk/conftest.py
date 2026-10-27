"""Fixtures for the risk layer tests.

Real ``MIAResult`` objects are used wherever possible so the risk layer cannot drift from the class it
consumes. Duck-typed stubs are only used for cases a real result cannot express, such as a
percentage-valued fixed-FPR table.
"""

import numpy as np
import pytest

from leakpro.reporting.mia_result import MIAResult


def make_result(n_members: int = 400,
                n_non_members: int = 400,
                separation: float = 1.0,
                name: str = "lira",
                seed: int = 0,
                inverted: bool = False) -> MIAResult:
    """Build a real MIAResult with a known member/non-member signal separation.

    Args:
    ----
        n_members: Number of members in the audit set.
        n_non_members: Number of non-members in the audit set.
        separation: Mean shift of the member signal distribution.
        name: Result name.
        seed: RNG seed.
        inverted: When True, members get the *lower* signal, mimicking an attack that passes a raw
            loss to a "higher is member" comparison.

    Returns:
    -------
        A MIAResult built from full scores.

    """
    rng = np.random.default_rng(seed)
    shift = -separation if inverted else separation
    labels = np.array([1] * n_members + [0] * n_non_members)
    signals = np.concatenate([rng.normal(shift, 1.0, n_members), rng.normal(0.0, 1.0, n_non_members)])
    return MIAResult.from_full_scores(true_membership=labels, signal_values=signals, result_name=name)


class StubResult:
    """Minimal duck-typed stand-in for cases a real MIAResult cannot express."""

    def __init__(self, result_name: str, roc_auc: float = 0.8, fixed_fpr_table: dict = None,
                 fpr: list = None, tpr: list = None, true: list = None, signal_values: list = None) -> None:
        self.result_name = result_name
        self.roc_auc = roc_auc
        self.fixed_fpr_table = fixed_fpr_table
        self.fpr = np.array(fpr) if fpr is not None else None
        self.tpr = np.array(tpr) if tpr is not None else None
        self.true = np.array(true) if true is not None else np.array([1] * 400 + [0] * 400)
        self.signal_values = np.array(signal_values) if signal_values is not None else None
        self.id = result_name
        self.metadata = {}


@pytest.fixture
def result() -> MIAResult:
    """Return a well-behaved real result: 400 members, 400 non-members, clear separation."""
    return make_result()
