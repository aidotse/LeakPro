#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Factory for diffusion training-data extraction attacks."""

from __future__ import annotations

from leakpro.attacks.extraction_attacks.abstract_extraction import AbstractExtraction
from leakpro.attacks.extraction_attacks.carlini import AttackCarliniExtraction
from leakpro.attacks.extraction_attacks.configs import CarliniConfig, SIDEConfig
from leakpro.attacks.extraction_attacks.side import AttackSIDEExtraction
from leakpro.attacks.extraction_attacks.utils import (
    extraction_audit_fingerprint,
    normalize_conditions,
    require_authorized,
    resolve_device,
)
from leakpro.input_handler.extraction_handler import ExtractionHandler


class AttackFactoryExtraction:
    """Create a configured extraction attack by its YAML name."""

    attack_classes = {
        "carlini_diffusion": AttackCarliniExtraction,
        "side": AttackSIDEExtraction,
    }

    @classmethod
    def validate_config(cls, name: str, attack_config: dict) -> CarliniConfig | SIDEConfig:
        """Validate a named attack and its authorization before provider access."""
        if name == "carlini_diffusion":
            config: CarliniConfig | SIDEConfig = CarliniConfig(**attack_config)
        elif name == "side":
            config = SIDEConfig(**attack_config)
        else:
            raise ValueError(f"Unknown extraction attack type: {name}")
        require_authorized(config.authorized_audit)
        resolve_device(config.distance_device)
        if isinstance(config, SIDEConfig):
            resolve_device(config.compute_device)
        return config

    @classmethod
    def create_attack(
        cls,
        name: str,
        attack_config: dict,
        handler: ExtractionHandler,
    ) -> AbstractExtraction:
        """Instantiate a supported extraction attack."""
        config = cls.validate_config(name, attack_config)
        if name == "carlini_diffusion":
            if not isinstance(config, CarliniConfig):
                raise RuntimeError("Carlini configuration dispatch failed.")
            conditions = normalize_conditions(handler.get_extraction_conditions())
            reference_images = handler.get_extraction_reference_images()
            audit_fingerprint = extraction_audit_fingerprint(
                handler.configs.target.fingerprint,
                conditions=conditions,
                reference_images=reference_images,
            )
            return AttackCarliniExtraction(
                handler.get_diffusion_adapter(),
                config,
                audit_fingerprint=audit_fingerprint,
                conditions=conditions,
                reference_images=reference_images,
            )
        if name == "side":
            if not isinstance(config, SIDEConfig):
                raise RuntimeError("SIDE configuration dispatch failed.")
            reference_images = handler.get_extraction_reference_images()
            audit_fingerprint = extraction_audit_fingerprint(
                handler.configs.target.fingerprint,
                conditions=None,
                reference_images=reference_images,
            )
            feature_extractor = handler.get_side_feature_extractor()
            if feature_extractor is None:
                raise ValueError("SIDE requires get_side_feature_extractor() to return a torch module.")
            return AttackSIDEExtraction(
                handler.get_diffusion_adapter(),
                feature_extractor,
                config,
                audit_fingerprint=audit_fingerprint,
                reference_images=reference_images,
                feature_transform=handler.get_side_feature_transform(),
                classifier_factory=handler.get_side_classifier_factory(),
                reference_score_fn=handler.get_extraction_reference_score(),
            )
        raise RuntimeError("unreachable extraction attack branch")
