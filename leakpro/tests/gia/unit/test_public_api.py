#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""The public surface of the modular GIA package: what is exported and that every export resolves."""
from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

import leakpro.attacks.gia_attacks.modular as modular
import leakpro.attacks.gia_attacks.modular.components as components
import leakpro.attacks.gia_attacks.modular.presets as presets

TOP_LEVEL = {
    "Attack", "run_attack", "simulate_client", "Task", "FedSGD", "FedAvg", "OptimStage", "ThreatModel",
    "ClientObservation", "TargetInfo", "State", "Context", "RunMetadata",
}
SUBPACKAGES = [
    m.name for m in pkgutil.iter_modules(components.__path__, components.__name__ + ".")
    if m.ispkg and not m.name.endswith(".utils")
]


def test_top_level_exports_are_exactly_the_documented_set():
    assert set(modular.__all__) == TOP_LEVEL
    for name in modular.__all__:
        assert getattr(modular, name) is not None


def test_every_preset_factory_is_exported_and_callable():
    modules = {m.name for m in pkgutil.iter_modules(presets.__path__) if not m.name.startswith("_")}
    assert modules, "no preset modules found"
    missing = []
    for module_name in sorted(modules):
        mod = importlib.import_module(f"{presets.__name__}.{module_name}")
        for name, obj in vars(mod).items():
            if inspect.isfunction(obj) and obj.__module__ == mod.__name__ and not name.startswith("_") and name not in presets.__all__:
                missing.append(f"{module_name}.{name}")
    assert not missing, f"preset factories not exported by presets: {missing}"
    for name in presets.__all__:
        assert callable(getattr(presets, name))


@pytest.mark.parametrize("package", SUBPACKAGES)
def test_every_public_class_of_a_component_subpackage_is_exported(package):
    pkg = importlib.import_module(package)
    assert hasattr(pkg, "__all__"), f"{package} has no __all__"
    missing = []
    for info in pkgutil.iter_modules(pkg.__path__, package + "."):
        mod = importlib.import_module(info.name)
        for name, obj in vars(mod).items():
            if inspect.isclass(obj) and obj.__module__ == mod.__name__ and not name.startswith("_") and name not in pkg.__all__:
                missing.append(f"{info.name}.{name}")
    assert not missing, f"not exported by {package}: {missing}"
    for name in pkg.__all__:
        assert hasattr(pkg, name), f"{package}.__all__ lists {name}, which does not resolve"


def test_every_public_class_of_a_flat_components_module_is_exported():
    missing = []
    for info in pkgutil.iter_modules(components.__path__, components.__name__ + "."):
        if info.ispkg:
            continue
        mod = importlib.import_module(info.name)
        for name, obj in vars(mod).items():
            if inspect.isclass(obj) and obj.__module__ == mod.__name__ and not name.startswith("_") and name not in components.__all__:
                missing.append(f"{info.name}.{name}")
    assert not missing, f"not exported by components: {missing}"
    for name in components.__all__:
        assert hasattr(components, name), f"components.__all__ lists {name}, which does not resolve"
