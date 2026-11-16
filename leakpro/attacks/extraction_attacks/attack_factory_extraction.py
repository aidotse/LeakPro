#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Factory for training-data extraction attacks."""

from leakpro.attacks.extraction_attacks.abstract_extraction import AbstractExtractionAttack
from leakpro.attacks.extraction_attacks.memorization_extraction import AttackMemorizationExtraction
from leakpro.input_handler.extraction_handler import ExtractionHandler


class AttackFactoryExtraction:
    """Create extraction attack objects by name."""

    attack_classes = {
        "memorization_extraction": AttackMemorizationExtraction,
    }

    @classmethod
    def create_attack(cls, name: str, attack_config: dict, handler: ExtractionHandler) -> AbstractExtractionAttack:  # noqa: ANN102
        """Create the attack ``name`` from its config entry.

        Args:
        ----
            name: Attack name from ``audit.attack_list``.
            attack_config: The entry's settings (without the ``attack`` key).
            handler: The extraction handler.

        Returns:
        -------
            The attack object.

        Raises:
        ------
            ValueError: If ``name`` is not a known extraction attack.

        """
        attack_cls = cls.attack_classes.get(name)
        if attack_cls is None:
            raise ValueError(f"Unknown extraction attack: {name}")
        return attack_cls(handler, attack_config)
