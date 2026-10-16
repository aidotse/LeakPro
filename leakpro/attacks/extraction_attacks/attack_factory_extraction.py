#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Factory for diffusion training-data extraction attacks."""

from __future__ import annotations

from leakpro.attacks.extraction_attacks.abstract_extraction import AbstractExtraction
from leakpro.attacks.extraction_attacks.carlini import AttackCarliniExtraction
from leakpro.attacks.extraction_attacks.side import AttackSIDEExtraction
from leakpro.input_handler.extraction_handler import ExtractionHandler


class AttackFactoryExtraction:
    """Create a configured extraction attack by its YAML name."""

    attack_classes = {
        "carlini_diffusion": AttackCarliniExtraction,
        "side": AttackSIDEExtraction,
    }

    @classmethod
    def create_attack(
        cls,
        name: str,
        attack_config: dict,
        handler: ExtractionHandler,
    ) -> AbstractExtraction:
        """Instantiate a registered attack using its handler-specific constructor."""
        if name in cls.attack_classes:
            return cls.attack_classes[name].from_handler(handler, attack_config)
        raise ValueError(f"Unknown extraction attack type: {name}")
