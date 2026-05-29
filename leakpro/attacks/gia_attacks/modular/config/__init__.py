#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Config package for the modular GIA framework.

Importing this package populates the component registry as a side effect.
``AttackBuilder`` and ``resolve_attack_config`` are lazily imported from
``config.builder`` to avoid circular imports — component modules import
``config.registry`` at load time, which would otherwise trigger the full
builder import chain before component modules finish initialising.
"""

from leakpro.attacks.gia_attacks.modular.config import _eager_imports as _  # noqa: F401
from leakpro.attacks.gia_attacks.modular.config.registry import build_component, register, registered_keys
from leakpro.attacks.gia_attacks.modular.config.schema import (
    AttackConfig,
    DiffusionStageConfig,
    FedAvgConfig,
    OptimizerStageConfig,
    StageConfig,
    TrainingSimulatorConfig,
)
from leakpro.attacks.gia_attacks.modular.config.spec import ComponentSpec


def __getattr__(name: str) -> object:
    """Lazily import builder symbols to avoid circular imports."""
    if name in ("AttackBuilder", "resolve_attack_config"):
        from leakpro.attacks.gia_attacks.modular.config.builder import (  # noqa: PLC0415
            AttackBuilder,
            resolve_attack_config,
        )
        globals()["AttackBuilder"] = AttackBuilder
        globals()["resolve_attack_config"] = resolve_attack_config
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # registry
    "register",
    "build_component",
    "registered_keys",
    # schema
    "AttackConfig",
    "OptimizerStageConfig",
    "DiffusionStageConfig",
    "TrainingSimulatorConfig",
    "FedAvgConfig",
    "StageConfig",
    # spec
    "ComponentSpec",
    # builder (lazy)
    "AttackBuilder",
    "resolve_attack_config",
]
