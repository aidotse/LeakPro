#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""See Through Gradients (Yin et al., CVPR 2021)."""

from leakpro.attacks.gia_attacks.modular.config.schema import (
    AttackConfig,
    OptimizerStageConfig,
    TrainingSimulatorConfig,
)
from leakpro.attacks.gia_attacks.modular.config.spec import ComponentSpec


def see_through_gradients_attack(
    learning_rate: float = 0.1,
    max_iterations: int = 8000,
    tv_weight: float = 1e-4,
    l2_weight: float = 1e-6,
    bn_weight: float = 1e-2,
    group_weight: float = 1e-2,
    num_seeds: int = 4,
    gradient_noise_std: float = 0.2,
    scheduler_type: str = "cosine",
) -> AttackConfig:
    """See Through Gradients — Yin et al., CVPR 2021.

    Multi-seed batch gradient inversion with group consistency regularisation.
    Each image is reconstructed from G parallel random initialisations that are
    jointly optimised and then averaged.

    Threat Model: Model E (Statistical-Informed Eavesdropper) — gradients +
    BN running statistics + auxiliary knowledge.

    Memory note: requires G× memory versus single-seed attacks.

    Reference:
        Yin, H., Molchanov, P., Alvarez, J. M., et al. (2021). See through
        Gradients: Image Batch Recovery via GradInversion. CVPR 2021.
    """
    losses = [
        ComponentSpec(type="loss.gradient_matching", params={"loss_type": "l2", "weight": 1.0}),
    ]
    if bn_weight > 0:
        losses.append(ComponentSpec(type="loss.bn_stats", params={"strategy": "running", "weight": bn_weight}))
    if tv_weight > 0:
        losses.append(ComponentSpec(type="loss.tv", params={"weight": tv_weight}))
    if l2_weight > 0:
        losses.append(ComponentSpec(type="loss.l2", params={"weight": l2_weight}))
    if group_weight > 0 and num_seeds > 1:
        losses.append(ComponentSpec(type="loss.group_consistency", params={"weight": group_weight}))

    return AttackConfig(
        threat_model_type="model_e",
        label_inference=ComponentSpec(type="label_inference.oracle"),
        num_seeds_per_image=num_seeds,
        seed_aggregation=ComponentSpec(type="aggregation.mean"),
        stages=[
            OptimizerStageConfig(
                learning_rate=learning_rate,
                max_iterations=max_iterations,
                optimizer_type="adam",
                scheduler_type=scheduler_type,
                gradient_noise_std=gradient_noise_std,
                constraint=ComponentSpec(type="constraint.clip"),
                losses=losses,
            ),
        ],
        training_simulator=TrainingSimulatorConfig(
            epochs=1,
            model_mode="train",
            compute_mode="updates",
        ),
    )
