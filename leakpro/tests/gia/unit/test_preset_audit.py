#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""The preset audit reads real values off every framework preset."""
from __future__ import annotations

from leakpro.tests.gia.parity import preset_audit


def test_audit_has_a_framework_column_for_every_reference_preset() -> None:
    assert all(factory is not None for factory in preset_audit.FRAMEWORK_PRESET.values())


def test_audit_reports_real_noise_patience_and_threat_model() -> None:
    text = preset_audit.render()
    assert "not available" not in text
    assert "undeclared" not in text
    assert "| gradient noise std |" in text and "| early-stop patience |" in text
