#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""GGDM: Gradient-Guided Diffusion Model (Gu et al., WWW 2024)."""

from leakpro.attacks.gia_attacks.modular.config.schema import (
    AttackConfig,
    DiffusionStageConfig,
)
from leakpro.attacks.gia_attacks.modular.config.spec import ComponentSpec


def ggdm_attack(
    diffusion_model_id: str = "google/ddpm-cifar10-32",
    gamma: float = 100.0,
    grad_clip: float = 1.0,
    log_interval: int = 100,
) -> AttackConfig:
    """GGDM — Gradient-Guided Diffusion Model, Gu et al., WWW 2024.

    Guides DDPM reverse sampling via similarity-gradient adjustments at each
    timestep.  No separate optimisation loop — the reconstruction emerges from
    a single guided reverse diffusion pass.

    Threat Model: Model B (Informed Eavesdropper).

    Reference:
        Gu et al., "Federated Learning Vulnerabilities: Privacy Attacks with
        Denoising Diffusion Probabilistic Models", WWW 2024.
    """
    return AttackConfig(
        threat_model_type="model_b",
        label_inference=ComponentSpec(type="label_inference.oracle"),
        stages=[
            DiffusionStageConfig(
                diffusion_model_uri=diffusion_model_id,
                mean_strategy=ComponentSpec(
                    type="mean.similarity_guidance",
                    params={"gamma": gamma, "grad_clip": grad_clip},
                ),
                noise_strategy=ComponentSpec(type="noise.standard"),
                log_interval=log_interval,
                losses=[
                    ComponentSpec(type="loss.gradient_matching", params={"loss_type": "cosine", "weight": 1.0}),
                ],
            ),
        ],
    )
