#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Every preset documents what the attacker knows, what it cites and where it deviates."""
from __future__ import annotations

import inspect
import re

import pytest

from leakpro.attacks.gia_attacks.modular import presets

HEDGES = re.compile(r"\(verify\)|unverified|not verified|not yet identified", re.IGNORECASE)


@pytest.mark.parametrize("name", presets.__all__)
def test_preset_docstring_cites_and_lists_deviations(name: str) -> None:
    doc = inspect.getdoc(getattr(presets, name)) or ""
    for section in ("Threat model:", "References:", "Deviations:"):
        assert section in doc, f"{name}: docstring lacks '{section}'"
    after_references = doc.split("References:", 1)[1].split("Deviations:", 1)[0]
    assert after_references.strip(), f"{name}: empty References"
    assert doc.split("Deviations:", 1)[1].strip(), f"{name}: empty Deviations"
    assert not HEDGES.search(doc), f"{name}: docstring hedges a citation"
