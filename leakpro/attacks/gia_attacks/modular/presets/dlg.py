#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""DLG: Deep Leakage from Gradients (Zhu et al., NeurIPS 2019)."""

from leakpro.attacks.gia_attacks.modular.config.schema import (
    AttackConfig,
    OptimizerStageConfig,
    TrainingSimulatorConfig,
)
from leakpro.attacks.gia_attacks.modular.config.spec import ComponentSpec


def dlg_attack(
    learning_rate: float = 1.0,
    max_iterations: int = 300,
    optimizer_type: str = "lbfgs",
    patience: int = 50,
) -> AttackConfig:
    """Deep Leakage from Gradients — Zhu et al. 2019.

    Optimises both data and labels jointly using L-BFGS with L2 gradient matching.

    Threat Model: Model A (Eavesdropper) — gradients only.

    Reference:
        Zhu, L., Liu, Z., & Han, S. (2019). Deep leakage from gradients. NeurIPS 2019.
    """
    return AttackConfig(
        threat_model_type="model_a",
        label_inference=ComponentSpec(type="label_inference.joint"),
        stages=[
            OptimizerStageConfig(
                learning_rate=learning_rate,
                max_iterations=max_iterations,
                optimizer_type=optimizer_type,
                patience=patience,
                use_gradient_sign=False,
                return_best=True,
                losses=[
                    ComponentSpec(type="loss.gradient_matching", params={"loss_type": "l2", "weight": 1.0}),
                ],
            ),
        ],
        training_simulator=TrainingSimulatorConfig(
            epochs=1,
            model_mode="eval",
            compute_mode="gradients",
        ),
    )
