#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Huang et al. gradient inversion with BN statistics (NeurIPS 2021)."""

from leakpro.attacks.gia_attacks.modular.config.schema import (
    AttackConfig,
    OptimizerStageConfig,
    TrainingSimulatorConfig,
)
from leakpro.attacks.gia_attacks.modular.config.spec import ComponentSpec


def huang_attack(
    learning_rate: float = 0.1,
    max_iterations: int = 10000,
    tv_weight: float = 0.052,
    bn_weight: float = 0.00016,
    bn_strategy: str = "running",
    scheduler_type: str = "step",
) -> AttackConfig:
    """Huang et al. 2021 — gradient inversion with running BN statistics.

    Uses the model's running BN statistics (running_mean / running_var) as
    regularisation targets alongside cosine gradient matching.

    Threat Model: Model E (Statistical-Informed Eavesdropper) — gradients +
    BN statistics + full auxiliary knowledge.

    Reference:
        Huang, Y., et al. "Evaluating Gradient Inversion Attacks and Defenses
        in Federated Learning." NeurIPS 2021.
    """
    return AttackConfig(
        threat_model_type="model_e",
        label_inference=ComponentSpec(type="label_inference.oracle"),
        stages=[
            OptimizerStageConfig(
                learning_rate=learning_rate,
                max_iterations=max_iterations,
                optimizer_type="adam",
                scheduler_type=scheduler_type,
                constraint=ComponentSpec(type="constraint.clip"),
                losses=[
                    ComponentSpec(type="loss.gradient_matching", params={"loss_type": "cosine", "weight": 1.0}),
                    ComponentSpec(type="loss.tv", params={"weight": tv_weight}),
                    ComponentSpec(type="loss.bn_stats", params={"strategy": bn_strategy, "weight": bn_weight}),
                ],
            ),
        ],
        training_simulator=TrainingSimulatorConfig(
            epochs=1,
            model_mode="train",
            compute_mode="updates",
        ),
    )
