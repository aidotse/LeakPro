#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Modality transforms between data space and model input space for the modular gradient-inversion framework."""
from leakpro.attacks.gia_attacks.modular.components.modalities.detection import DetectionImageModality
from leakpro.attacks.gia_attacks.modular.components.modalities.image import ImageNormalizationModality

__all__ = [
    "DetectionImageModality",
    "ImageNormalizationModality",
]
