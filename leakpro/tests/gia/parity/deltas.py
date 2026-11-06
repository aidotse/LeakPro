#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Every known difference between the framework and the v1 reference, with its classification.

BUG: the framework is wrong and will be fixed; the fixing commit deletes the entry.
INTENTIONAL: the framework differs on purpose; the reason says why it is right.
Entries apply as strict xfails that only accept assertion failures (a crash is not a known delta), so a delta that silently disappears fails until its entry is removed.
"""
import pytest

_V1_UPDATE_LABELS = ("INTENTIONAL: v1 applies the client lr a second time to an update payload that already carries "
    "-lr*sum(grad), so its Geng/Dimitrov inference on updates recovers wrong labels (the recorded v1 result differs "
    "from the true labels). The framework divides the payload by the client lr once. "
    "TestDimitrovLabelRecovery shows recovery at client lr 0.01 and 0.1, and TestGengLabelRecovery shows Geng "
    "recovering the labels at its fixed lr of 0.1.")

_V1_INFERRED_BN_BESSEL = ("INTENTIONAL: v1's inferred BN strategy undoes the Bessel correction with the batch size N; "
    "PyTorch's running_var update uses the per-channel element count (N*H*W for BatchNorm2d), so v1's variance "
    "target is too small. The framework uses the element count; test_bn_inferred_target.py shows the term is zero "
    "at the client's batch.")

KNOWN_DELTAS: dict[str, str] = {
    "loss-gia_running:BNStatisticsRegularization(inferred_bn_statistics)": _V1_INFERRED_BN_BESSEL,
    "loss-gia_running": _V1_INFERRED_BN_BESSEL,
    "traj-gia_running": _V1_INFERRED_BN_BESSEL,
    "labels-dimitrov-updates": _V1_UPDATE_LABELS,
    "labels-geng-updates": _V1_UPDATE_LABELS,
    "traj-see_through_gradients": ("INTENTIONAL: v1 defects not reproduced. (a) With several seeds v1's BN prior "
        "keeps only the last seed's hook value (hooks overwrite r_feature on every forward; "
        "bn_statistics_strategies.py compute_regularization). (b) v1's GroupConsistencyRegularization compares a "
        "[E,N,C,H,W] seed slice with a [E,N,1,C,H,W] consensus, which broadcasts across images "
        "(loss_components.py GroupConsistencyRegularization.compute). "
        "test_stg_matches_v1_with_v1_defects_replicated shows every other part matches."),
}


def delta(test_id: str) -> tuple[pytest.MarkDecorator, ...]:
    """The marks for one parity case: a strict xfail if it is a known delta, otherwise none."""
    reason = KNOWN_DELTAS.get(test_id)
    if reason is None:
        return ()
    if not reason.startswith(("BUG:", "INTENTIONAL:")):
        raise ValueError(f"delta {test_id!r} must start with 'BUG:' or 'INTENTIONAL:'")
    return (pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason),)
