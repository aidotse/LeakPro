#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Everything a user implements or needs to compose a custom attack is importable from ``modular.core``.

The package README's "Extending" table names the interfaces; this checks they and the supporting types
(``LossInputs``, the axes, state checkpointing) are exported, and that every interface in
``core/interfaces.py`` is.
"""
from __future__ import annotations

import abc
import inspect
import re
from pathlib import Path

import leakpro.attacks.gia_attacks.modular.core as core
from leakpro.attacks.gia_attacks.modular.core import interfaces
from leakpro.attacks.gia_attacks.modular.components import representations

README = Path(core.__file__).resolve().parents[1] / "README.md"
SUPPORTING = {"LossInputs", "LossScope", "EvalResult", "Axis", "AxisManager", "save_state", "load_state"}


def test_every_interface_is_exported() -> None:
    abstract = {name for name, obj in vars(interfaces).items()
                if inspect.isclass(obj) and obj.__module__ == interfaces.__name__ and abc.ABC in obj.__mro__}
    assert abstract
    assert abstract <= set(core.__all__)


def test_readme_extending_table_names_resolve_from_core() -> None:
    section = README.read_text().split("## Extending", 1)[1].split("\n## ", 1)[0]
    names = set(re.findall(r"^\| `(\w+)` \|", section, re.M))
    assert {"LossTerm", "Driver", "Representation", "Modality", "Evaluator"} <= names
    assert names <= set(core.__all__)
    assert "modular.core import" in section


def test_supporting_types_are_exported() -> None:
    assert set(core.__all__) >= SUPPORTING
    for name in core.__all__:
        assert getattr(core, name) is not None


def test_representations_export_only_representations() -> None:
    assert "to_unit_range" not in representations.__all__
