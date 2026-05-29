#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Inverting Gradients (Geiping et al., NeurIPS 2020)."""

from leakpro.attacks.gia_attacks.modular.config.schema import (
    AttackConfig,
    OptimizerStageConfig,
    TrainingSimulatorConfig,
)
from leakpro.attacks.gia_attacks.modular.config.spec import ComponentSpec


def inverting_gradients_attack(
    learning_rate: float = 0.1,
    max_iterations: int = 4000,
    optimizer_type: str = "adam",
    tv_weight: float = 1e-3,
    l2_weight: float = 0.0,
    scheduler_type: str = "step",
    use_gradient_sign: bool = True,
    return_best: bool = True,
) -> AttackConfig:
    """Inverting Gradients — Geiping et al. 2020.

    Cosine similarity gradient matching with total variation regularisation
    and gradient-sign step updates.

    Threat Model: Model B (Informed Eavesdropper) — gradients + auxiliary knowledge.

    Reference:
        Geiping, J., Bauermeister, H., Dröge, H., & Moeller, M. (2020).
        Inverting gradients — how easy is it to break privacy in federated learning?
        NeurIPS 2020.
    """
    losses = [
        ComponentSpec(type="loss.gradient_matching", params={"loss_type": "cosine", "weight": 1.0}),
    ]
    if tv_weight > 0:
        losses.append(ComponentSpec(type="loss.tv", params={"weight": tv_weight}))
    if l2_weight > 0:
        losses.append(ComponentSpec(type="loss.l2", params={"weight": l2_weight}))

    return AttackConfig(
        threat_model_type="model_b",
        label_inference=ComponentSpec(type="label_inference.oracle"),
        stages=[
            OptimizerStageConfig(
                learning_rate=learning_rate,
                max_iterations=max_iterations,
                optimizer_type=optimizer_type,
                scheduler_type=scheduler_type,
                use_gradient_sign=use_gradient_sign,
                return_best=return_best,
                constraint=ComponentSpec(type="constraint.clip"),
                losses=losses,
            ),
        ],
        training_simulator=TrainingSimulatorConfig(
            epochs=1,
            model_mode="eval",
            compute_mode="updates",
        ),
    )
