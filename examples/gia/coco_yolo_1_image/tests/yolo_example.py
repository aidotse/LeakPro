#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: AGPL-3.0-only
#
"""Loads this example's modules by file path (the folder is not an importable package)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

EXAMPLE_DIR = Path(__file__).resolve().parents[1]


def _load(name: str) -> ModuleType:
    key = f"yolo_example_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, EXAMPLE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def load_yolo_example() -> SimpleNamespace:
    """Import the example's ``model.py`` and ``yolo_attack.py`` by file path, without touching ``sys.path`` or the cwd."""
    return SimpleNamespace(model=_load("model"), attack=_load("yolo_attack"))
