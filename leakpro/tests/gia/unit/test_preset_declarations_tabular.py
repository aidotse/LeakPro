#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Every tabular preset states what the attacker knows."""
from __future__ import annotations

import pytest

from leakpro.attacks.gia_attacks.modular import presets

TABULAR = {
    "continuous_cols": [0, 1], "categorical_col_groups": [(2, 3)], "col_means": {0: 0.0, 1: 0.0},
    "col_stds": {0: 1.0, 1: 1.0}, "num_features": 5,
}

BUILDERS = {
    "tabular": lambda: presets.tabular(**TABULAR, num_classes=2),
    "tabular_raw_onehot": lambda: presets.tabular_raw_onehot(**TABULAR),
}


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_preset_declares_a_threat_model(name: str) -> None:
    assert BUILDERS[name]().threat_model is not None, f"{name} leaves its threat model undeclared"
