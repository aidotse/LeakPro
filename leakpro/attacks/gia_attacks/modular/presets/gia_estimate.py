#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""GIA Estimate — proxy-data BN statistics."""

from leakpro.attacks.gia_attacks.modular.config.schema import (
    AttackConfig,
    OptimizerStageConfig,
    TrainingSimulatorConfig,
)
from leakpro.attacks.gia_attacks.modular.config.spec import ComponentSpec


def gia_estimate_attack(
    learning_rate: float = 0.1,
    max_iterations: int = 10000,
    tv_weight: float = 0.052,
    bn_weight: float = 0.00016,
    scheduler_type: str = "step",
) -> AttackConfig:
    """GIA Estimate — proxy/surrogate dataset BN statistics.

    Uses a proxy dataset from the same or similar domain to estimate batch
    statistics for BN regularisation.  More realistic than having exact
    running statistics.

    Threat Model: Model D (Data-Enhanced Eavesdropper) — gradients +
    surrogate dataset + auxiliary knowledge.

    Note: Requires passing proxy_dataloader to run_attack().
    """
    return AttackConfig(
        threat_model_type="model_d",
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
                    ComponentSpec(type="loss.bn_stats", params={"strategy": "proxy", "weight": bn_weight}),
                ],
            ),
        ],
        training_simulator=TrainingSimulatorConfig(
            epochs=1,
            model_mode="train",
            compute_mode="updates",
        ),
    )
