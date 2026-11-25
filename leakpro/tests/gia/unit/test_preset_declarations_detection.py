#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Every object-detection preset states what the attacker knows."""
from __future__ import annotations

import pytest
import torch

from leakpro.attacks.gia_attacks.modular import presets

_LABELS = torch.zeros((1, 1), dtype=torch.long)

BUILDERS = {
    "detection": lambda: presets.detection(oracle_labels=_LABELS, k_max=1),
}


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_preset_declares_a_threat_model(name: str) -> None:
    assert BUILDERS[name]().threat_model is not None, f"{name} leaves its threat model undeclared"
