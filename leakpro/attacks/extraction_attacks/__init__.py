#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Diffusion training-data extraction attacks."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from leakpro.attacks.extraction_attacks.carlini import AttackCarliniExtraction
    from leakpro.attacks.extraction_attacks.side import AttackSIDEExtraction

__all__ = ["AttackCarliniExtraction", "AttackSIDEExtraction"]


def __getattr__(name: str) -> object:
    """Load an attack only when requested."""
    if name == "AttackCarliniExtraction":
        from leakpro.attacks.extraction_attacks.carlini import AttackCarliniExtraction  # noqa: PLC0415

        return AttackCarliniExtraction
    if name == "AttackSIDEExtraction":
        from leakpro.attacks.extraction_attacks.side import AttackSIDEExtraction  # noqa: PLC0415

        return AttackSIDEExtraction
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
