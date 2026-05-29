#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Component registry — maps string keys to factory callables.

Usage (registering a component)::

    from leakpro.attacks.gia_attacks.modular.config.registry import register

    @register("constraint.clip")
    class ClipConstraint(ConstraintStrategy):
        ...

Usage (building a component)::

    from leakpro.attacks.gia_attacks.modular.config.registry import build_component
    from leakpro.attacks.gia_attacks.modular.config.spec import ComponentSpec

    constraint = build_component(ComponentSpec(type="constraint.clip"))

The registry is populated as a side-effect of importing component modules.
``config/__init__.py`` imports ``_eager_imports`` to ensure all built-in
components are registered before any ``build_component`` call.
"""

from __future__ import annotations

from typing import Any, Callable

_REGISTRY: dict[str, Callable[..., Any]] = {}


def register(name: str) -> Callable:
    """Decorator that registers a class or factory function under *name*.

    Raises ``ValueError`` on duplicate registration to catch copy-paste errors.
    """
    def deco(fn: Callable) -> Callable:
        if name in _REGISTRY:
            raise ValueError(
                f"Duplicate registry entry: '{name}' is already registered by "
                f"{_REGISTRY[name]!r}. Each component must have a unique key."
            )
        _REGISTRY[name] = fn
        return fn
    return deco


def build_component(
    spec: "ComponentSpec",  # noqa: F821 — forward ref resolved at call time
    *,
    live_overrides: dict[str, Any] | None = None,
) -> Any:  # noqa: ANN401 — registry returns arbitrary component types
    """Instantiate a component from a :class:`~leakpro.…config.spec.ComponentSpec`.

    If *live_overrides* is provided and ``spec.id`` matches a key, the
    pre-built object is returned directly (skipping the registry).  This is
    the notebook / large-model escape hatch.

    Raises:
        KeyError: If ``spec.type`` is not registered.  The message includes
            all registered keys in the same family for typo-hunting.

    """
    from leakpro.attacks.gia_attacks.modular.config.spec import ComponentSpec  # noqa: PLC0415 — avoids cycle

    if not isinstance(spec, ComponentSpec):
        raise TypeError(f"Expected ComponentSpec, got {type(spec).__name__}")

    if live_overrides and spec.id and spec.id in live_overrides:
        return live_overrides[spec.id]

    if spec.type not in _REGISTRY:
        family = spec.type.split(".")[0] if "." in spec.type else None
        suggestions = sorted(k for k in _REGISTRY if family and k.startswith(family + "."))
        hint = f" Did you mean one of: {suggestions}?" if suggestions else f" All registered keys: {sorted(_REGISTRY)}"
        raise KeyError(f"Unknown component type '{spec.type}'.{hint}")

    return _REGISTRY[spec.type](**spec.params)


def registered_keys() -> list[str]:
    """Return a sorted list of all registered component keys."""
    return sorted(_REGISTRY)


__all__ = ["register", "build_component", "registered_keys"]
