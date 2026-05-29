#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""GIA Running — inferred BN statistics from momentum updates."""

from leakpro.attacks.gia_attacks.modular.config.schema import (
    AttackConfig,
    OptimizerStageConfig,
    TrainingSimulatorConfig,
)
from leakpro.attacks.gia_attacks.modular.config.spec import ComponentSpec


def gia_running_attack(
    learning_rate: float = 0.1,
    max_iterations: int = 3000,
    tv_weight: float = 0.052,
    bn_weight: float = 0.00016,
    bn_momentum: float = 0.1,
    scheduler_type: str = "step",
) -> AttackConfig:
    """GIA Running — inferred BN statistics.

    Infers the client's batch statistics from how the model's running statistics
    changed during training, using the EMA momentum parameter.  Requires access
    to pre- and post-training running BN statistics.

    Threat Model: Model E (Statistical-Informed Eavesdropper).

    Note: Requires passing client observations with pre/post BN stats to run_attack().
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
                    ComponentSpec(
                        type="loss.bn_stats",
                        params={"strategy": "inferred", "weight": bn_weight, "momentum": bn_momentum},
                    ),
                ],
            ),
        ],
        training_simulator=TrainingSimulatorConfig(
            epochs=1,
            model_mode="train",
            compute_mode="updates",
        ),
    )
