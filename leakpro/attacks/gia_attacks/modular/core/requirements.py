#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""What components need from the observation, checked once before an attack runs.

Each loss term and target inference declares the observation fields (and run inputs) it reads.
``run_attack`` checks them after the threat model has sanitized the observation, so a missing
field fails before any optimization with a message naming the component and the fix, instead of
a term silently contributing zero or failing mid-run.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, get_args

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.attack import Attack
    from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation

Requirement = Literal["gradients", "bn_stats", "pre_bn_stats", "batch_size", "loss_value", "proxy_dataloader"]
REQUIREMENTS: tuple[Requirement, ...] = get_args(Requirement)

# The threat-model flag that grants each observation field it may strip.
GRANTED_BY: dict[Requirement, str] = {"bn_stats": "has_bn_statistics", "pre_bn_stats": "has_bn_statistics",
                                      "batch_size": "has_local_hyperparameters", "loss_value": "has_local_hyperparameters"}


def _hint(requirement: Requirement) -> str:
    if requirement == "proxy_dataloader":
        return "pass run_attack(..., proxy_dataloader=...)"
    if requirement in GRANTED_BY:
        return f"grant ThreatModel({GRANTED_BY[requirement]}=True) and supply observation.{requirement}"
    return f"supply observation.{requirement}"


def _describe(component: object) -> str:
    """The component's own repr when it defines one (dataclasses do), else its class name."""
    return repr(component) if type(component).__repr__ is not object.__repr__ else type(component).__name__


def _available(requirement: Requirement, observation: ClientObservation, proxy_dataloader: Any | None) -> bool:  # noqa: ANN401 -- proxy dataloader type is caller-defined
    if requirement == "proxy_dataloader":
        return proxy_dataloader is not None
    return getattr(observation, requirement) is not None


def validate_requirements(
    attack: Attack,
    observation: ClientObservation,
    *,
    proxy_dataloader: Any | None,  # noqa: ANN401 -- any iterable of batches
    start: int = 0,
    stop: int | None = None,
) -> None:
    """Check that every requirement of the stages ``[start:stop]`` and the target inference is met.

    ``observation`` is the sanitized observation the components will see.

    Raises:
        ValueError: Listing every unmet requirement, one line per component and field.

    """
    components: list[tuple[str, Any]] = []
    if attack.target_inference is not None:
        components.append(("target inference", attack.target_inference))
    for idx, stage in enumerate(attack.stages[start:stop], start=start):
        components.extend((f"stage {idx} term {name!r}", term) for name, (term, _) in stage.losses.items())

    missing = []
    for label, component in components:
        for requirement in sorted(component.requirements()):
            if requirement not in REQUIREMENTS:
                raise ValueError(f"{label} ({_describe(component)}) declares an unknown requirement {requirement!r}.")
            if not _available(requirement, observation, proxy_dataloader):
                missing.append(f"{label} ({_describe(component)}) requires {requirement}: {_hint(requirement)}")
    if missing:
        raise ValueError("The attack cannot run on this observation:\n" + "\n".join(missing))
