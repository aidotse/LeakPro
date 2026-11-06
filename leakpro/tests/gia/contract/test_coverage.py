#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Every concrete implementation of a framework interface is in the contract-test registry."""
import importlib
import inspect
import pkgutil
import sys

import pytest

import leakpro.attacks.gia_attacks.modular as framework
from leakpro.tests.gia.contract.registry import IMPLEMENTATIONS


def _all_subclasses(cls: type) -> set[type]:
    out: set[type] = set()
    for sub in cls.__subclasses__():
        out |= {sub} | _all_subclasses(sub)
    return out


def _import_framework() -> None:
    for module in pkgutil.walk_packages(framework.__path__, framework.__name__ + "."):
        importlib.import_module(module.name)


@pytest.mark.parametrize("interface", list(IMPLEMENTATIONS), ids=lambda c: c.__name__)
def test_every_implementation_is_contract_tested(interface: type) -> None:
    _import_framework()
    shipped = {c for c in _all_subclasses(interface)
               if not inspect.isabstract(c) and not c.__name__.startswith("_") and c.__module__.startswith(framework.__name__)}
    missing = shipped - set(IMPLEMENTATIONS[interface])
    assert not missing, f"add to contract/registry.py or registry_<area>.py: {sorted(c.__qualname__ for c in missing)}"


# Interfaces defined in framework/core that have no registry entry, each with the reason it needs none.
EXCLUDED_INTERFACES: dict[str, str] = {
    "Callback": "an optional hook base with empty methods and no behaviour of its own; implementations "
                "(e.g. loggers) are exercised through the driver tests that pass callbacks.",
}


def _core_interfaces() -> set[type]:
    """Every abstract class (an ``abc.ABC`` subclass) defined in ``framework.core``, plus ``Callback``."""
    _import_framework()
    core = framework.__name__ + ".core"
    found: set[type] = set()
    for name, module in list(sys.modules.items()):
        if name == core or name.startswith(core + "."):
            found |= {c for c in vars(module).values()
                      if inspect.isclass(c) and c.__module__ == name and inspect.isabstract(c)}
    return found


def test_every_core_interface_is_registered_or_explicitly_excluded() -> None:
    interfaces = _core_interfaces()
    unregistered = {c.__name__ for c in interfaces if c not in IMPLEMENTATIONS}
    unregistered -= set(EXCLUDED_INTERFACES)
    assert not unregistered, f"add an IMPLEMENTATIONS entry or a commented exclusion: {sorted(unregistered)}"


def test_exclusions_name_existing_unregistered_interfaces() -> None:
    from leakpro.attacks.gia_attacks.modular.core import interfaces  # noqa: PLC0415

    registered = {c.__name__ for c in IMPLEMENTATIONS}
    for name in EXCLUDED_INTERFACES:
        assert hasattr(interfaces, name), f"stale exclusion {name}"
        assert name not in registered, f"{name} is registered; drop its exclusion"
