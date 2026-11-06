#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Contract-test cases of the generative priors: GAN and diffusion representations, their drivers and strategies."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

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
from leakpro.attacks.gia_attacks.modular.components.drivers.cma_es import CMAESDriver
from leakpro.attacks.gia_attacks.modular.components.drivers.diffusion import DiffusionSamplingDriver
from leakpro.attacks.gia_attacks.modular.components.losses import L2GradientLoss
from leakpro.attacks.gia_attacks.modular.components.representations.diffusion import DiffusionRepresentation
from leakpro.attacks.gia_attacks.modular.components.representations.gan import (
    GANRepresentation,
    UnfrozenGANRepresentation,
)
from leakpro.attacks.gia_attacks.modular.components.representations.gifd import GIFDIntermediateRepresentation
from leakpro.attacks.gia_attacks.modular.core.state import State
from leakpro.tests.gia import helpers, stand_ins
from leakpro.tests.gia.contract.registry import (
    DRIVERS,
    IMPLEMENTATIONS,
    MARKS,
    REPRESENTATIONS,
    DriverCase,
    RepresentationCase,
    SimpleCase,
    _needs,
)

LATENT = 8


def _latent_state(ctx: Any) -> State:  # noqa: ANN401
    return State(artifacts={"latent": torch.randn(*ctx.axes.shape, ctx.input_shape[0], LATENT)})


REPRESENTATIONS.update({
    GANRepresentation: lambda: RepresentationCase(
        GANRepresentation(stand_ins.TinyGenerator(LATENT, helpers.IMAGE_SHAPE).train(), LATENT)),
    UnfrozenGANRepresentation: lambda: RepresentationCase(
        UnfrozenGANRepresentation(stand_ins.TinyGenerator(LATENT, helpers.IMAGE_SHAPE).train(), LATENT),
        state=_latent_state),
    GIFDIntermediateRepresentation: lambda: RepresentationCase(
        GIFDIntermediateRepresentation(stand_ins.tiny_layered_generator(latent_dim=LATENT, channels=3),
                                       start_layer=1, l1_radius=10.0),
        state=_latent_state),
    # x_t is not projected (the sampler owns it); only the observable is clamped to the box.
    DiffusionRepresentation: lambda: RepresentationCase(DiffusionRepresentation(), bounded=False),
})


def _diffusion_driver() -> DriverCase:
    driver = DiffusionSamplingDriver(unet=stand_ins.ZeroUNet(), scheduler=stand_ins.tiny_ddpm_scheduler(3),
                                     mean_strategy=SimilarityGuidance(), noise_strategy=StandardNoiseInjection())
    return DriverCase(driver, DiffusionRepresentation(), {"grad": (L2GradientLoss(), 1.0)}, budget=3, nan_loss=True)


DRIVERS.update({
    CMAESDriver: lambda: DriverCase(
        CMAESDriver(population_size=4, sigma0=0.5, max_generations=2),
        GANRepresentation(stand_ins.TinyGenerator(LATENT, helpers.IMAGE_SHAPE), LATENT),
        {"grad": (L2GradientLoss(), 1.0)}, budget=2),
    DiffusionSamplingDriver: _diffusion_driver,
})

MEAN_STRATEGIES: dict[type, Callable[[], SimpleCase]] = {
    SimilarityGuidance: lambda: SimpleCase(SimilarityGuidance(gamma=1.0)),
    AdaptiveMeanOptimization: lambda: SimpleCase(AdaptiveMeanOptimization(LinearDecaySchedule(), inner_steps=2)),
}

NOISE_STRATEGIES: dict[type, Callable[[], SimpleCase]] = {
    StandardNoiseInjection: lambda: SimpleCase(StandardNoiseInjection()),
    GradientAlignedNoiseInjection: lambda: SimpleCase(GradientAlignedNoiseInjection()),
}

BLENDING_SCHEDULES: dict[type, Callable[[], SimpleCase]] = {
    ConstantSchedule: lambda: SimpleCase(ConstantSchedule()),
    LinearDecaySchedule: lambda: SimpleCase(LinearDecaySchedule()),
    CosineDecaySchedule: lambda: SimpleCase(CosineDecaySchedule()),
}

IMPLEMENTATIONS.update({
    MeanAdjustmentStrategy: MEAN_STRATEGIES,
    NoiseInjectionStrategy: NOISE_STRATEGIES,
    BlendingSchedule: BLENDING_SCHEDULES,
})

MARKS.update({
    CMAESDriver: (_needs("cma"),),
    DiffusionSamplingDriver: (_needs("diffusers"),),
})
