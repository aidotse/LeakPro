#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""GradInvDiff: Gradient Inversion via Diffusion (Wang et al., 2024)."""

from leakpro.attacks.gia_attacks.modular.config.schema import (
    AttackConfig,
    DiffusionStageConfig,
)
from leakpro.attacks.gia_attacks.modular.config.spec import ComponentSpec


def gradinvdiff_attack(
    diffusion_model_id: str = "google/ddpm-cifar10-32",
    inner_steps: int = 5,
    inner_lr: float = 0.01,
    schedule: str = "linear",
    noise_strategy: str = "gani",
    log_interval: int = 100,
) -> AttackConfig:
    """GradInvDiff — Gradient Inversion via Diffusion, Wang et al. 2024.

    Combines Adaptive Mean Optimisation (AMO) with Gradient-Aligned Noise
    Injection (GANI) for high-quality diffusion-based gradient inversion.

    Threat Model: Model B (Informed Eavesdropper).

    Reference:
        Wang et al., "GradInvDiff: Stealing Medical Privacy in Federated
        Learning via Diffusion-Based Gradient Inversion", 2024.
    """
    return AttackConfig(
        threat_model_type="model_b",
        label_inference=ComponentSpec(type="label_inference.oracle"),
        stages=[
            DiffusionStageConfig(
                diffusion_model_uri=diffusion_model_id,
                mean_strategy=ComponentSpec(
                    type="mean.adaptive_optimization",
                    params={"inner_steps": inner_steps, "inner_lr": inner_lr, "schedule": schedule},
                ),
                noise_strategy=ComponentSpec(type=f"noise.{noise_strategy}"),
                log_interval=log_interval,
                losses=[
                    ComponentSpec(type="loss.gradient_matching", params={"loss_type": "cosine", "weight": 1.0}),
                ],
            ),
        ],
    )
