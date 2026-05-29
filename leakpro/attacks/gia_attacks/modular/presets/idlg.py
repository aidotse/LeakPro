#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""iDLG: Improved Deep Leakage from Gradients (Zhao et al., arXiv 2020)."""

from leakpro.attacks.gia_attacks.modular.config.schema import (
    AttackConfig,
    OptimizerStageConfig,
    TrainingSimulatorConfig,
)
from leakpro.attacks.gia_attacks.modular.config.spec import ComponentSpec


def idlg_attack(
    learning_rate: float = 1.0,
    max_iterations: int = 300,
    optimizer_type: str = "lbfgs",
) -> AttackConfig:
    """Improved Deep Leakage from Gradients — Zhao et al. 2020.

    Analytically infers labels from gradient signs, then optimises only the
    data reconstruction using L-BFGS with L2 gradient matching.

    Threat Model: Model A (Eavesdropper) — gradients only.

    Reference:
        Zhao, B., Mopuri, K. R., & Bilen, H. (2020). iDLG: Improved deep
        leakage from gradients. arXiv:2001.02610.
    """
    return AttackConfig(
        threat_model_type="model_a",
        label_inference=ComponentSpec(type="label_inference.idlg"),
        stages=[
            OptimizerStageConfig(
                learning_rate=learning_rate,
                max_iterations=max_iterations,
                optimizer_type=optimizer_type,
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
