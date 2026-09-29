#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Factory for diffusion training-data extraction attacks."""

from __future__ import annotations

from leakpro.attacks.extraction_attacks.abstract_extraction import AbstractExtraction
from leakpro.attacks.extraction_attacks.carlini import AttackCarliniExtraction
from leakpro.attacks.extraction_attacks.side import AttackSIDEExtraction
from leakpro.attacks.extraction_attacks.utils_generative import (
    extraction_audit_hash,
    normalize_conditions,
    require_authorized,
)
from leakpro.input_handler.extraction_handler import ExtractionHandler
from leakpro.utils.device import get_device


class AttackFactoryExtraction:
    """Create a configured extraction attack by its YAML name."""

    attack_classes = {
        "carlini_diffusion": AttackCarliniExtraction,
        "side": AttackSIDEExtraction,
    }

    @classmethod
    def validate_config(
        cls, name: str, attack_config: dict,
    ) -> AttackCarliniExtraction.AttackConfig | AttackSIDEExtraction.AttackConfig:
        """Validate a named attack and its authorization before provider access."""
        if name == "carlini_diffusion":
            config = AttackCarliniExtraction.AttackConfig(**attack_config)
        elif name == "side":
            config = AttackSIDEExtraction.AttackConfig(**attack_config)
        else:
            raise ValueError(f"Unknown extraction attack type: {name}")
        require_authorized(config.authorized_audit)
        requested_devices = [config.distance_device]
        if isinstance(config, AttackSIDEExtraction.AttackConfig):
            requested_devices.append(config.compute_device)
        if "auto" in requested_devices:
            device = get_device()
            if device.type not in {"cpu", "cuda"}:
                raise ValueError(f"Unsupported extraction device type: {device.type!r}.")
        return config

    @classmethod
    def create_attack(
        cls,
        name: str,
        attack_config: dict,
        handler: ExtractionHandler,
    ) -> AbstractExtraction:
        """Instantiate a supported extraction attack."""
        config = cls.validate_config(
            name,
            {"random_seed": handler.configs.audit.random_seed, **attack_config},
        )
        reference_images = handler.get_extraction_reference_images()
        target_hash = handler.get_extraction_target_hash()
        if target_hash is None:
            target_hash = handler.configs.target.hash
        if target_hash is None:
            raise ValueError("Extraction requires a target hash from the input handler or audit.yaml.")
        audit_hash = extraction_audit_hash(target_hash, reference_images=reference_images)
        if name == "carlini_diffusion":
            if not isinstance(config, AttackCarliniExtraction.AttackConfig):
                raise RuntimeError("Carlini configuration dispatch failed.")
            conditions = normalize_conditions(handler.get_extraction_conditions())
            return AttackCarliniExtraction(
                handler.get_diffusion_adapter(),
                config,
                audit_hash=audit_hash,
                conditions=conditions,
                reference_images=reference_images,
            )
        if name == "side":
            if not isinstance(config, AttackSIDEExtraction.AttackConfig):
                raise RuntimeError("SIDE configuration dispatch failed.")
            feature_extractor = handler.get_side_feature_extractor()
            if feature_extractor is None:
                raise ValueError("SIDE requires get_side_feature_extractor() to return a torch module.")
            return AttackSIDEExtraction(
                handler.get_diffusion_adapter(),
                feature_extractor,
                config,
                audit_hash=audit_hash,
                reference_images=reference_images,
                feature_transform=handler.get_side_feature_transform(),
                classifier_factory=handler.get_side_classifier_factory(),
                reference_score_fn=handler.get_extraction_reference_score(),
            )
        raise RuntimeError("unreachable extraction attack branch")
