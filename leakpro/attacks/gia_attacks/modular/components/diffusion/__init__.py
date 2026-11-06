#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Diffusion-sampling strategies used by DiffusionSamplingDriver.

Mean adjustment, noise injection, and timestep-dependent blending schedules.
"""
from leakpro.attacks.gia_attacks.modular.components.diffusion.mean_strategies import (
    AdaptiveMeanOptimization,
    MeanAdjustmentStrategy,
    SimilarityGuidance,
)
from leakpro.attacks.gia_attacks.modular.components.diffusion.noise_strategies import (
    GradientAlignedNoiseInjection,
    NoiseInjectionStrategy,
    StandardNoiseInjection,
)
from leakpro.attacks.gia_attacks.modular.components.diffusion.schedules import (
    BlendingSchedule,
    ConstantSchedule,
    CosineDecaySchedule,
    LinearDecaySchedule,
)

__all__ = [
    "AdaptiveMeanOptimization",
    "BlendingSchedule",
    "ConstantSchedule",
    "CosineDecaySchedule",
    "GradientAlignedNoiseInjection",
    "LinearDecaySchedule",
    "MeanAdjustmentStrategy",
    "NoiseInjectionStrategy",
    "SimilarityGuidance",
    "StandardNoiseInjection",
]
