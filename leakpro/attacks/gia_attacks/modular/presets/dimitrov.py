#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Dimitrov et al. FedAvg multi-epoch attack (TMLR 2022)."""

from leakpro.attacks.gia_attacks.modular.config.schema import (
    AttackConfig,
    FedAvgConfig,
    OptimizerStageConfig,
    TrainingSimulatorConfig,
)
from leakpro.attacks.gia_attacks.modular.config.spec import ComponentSpec


def dimitrov_fedavg_attack(
    learning_rate: float = 1.0,
    max_iterations: int = 8000,
    tv_weight: float = 1e-4,
    lambda_inv: float = 1e-6,
    epochs: int = 3,
    batch_size: int = 1,
    scheduler_type: str = "cosine",
) -> AttackConfig:
    """Dimitrov et al. FedAvg multi-epoch attack — TMLR 2022.

    Reconstructs client data from FedAvg parameter updates where the client
    performs multiple local epochs.  Uses an epoch-order-invariant prior
    (``EpochOrderInvariantPrior``) added automatically by the builder when
    ``fedavg.lambda_inv > 0``.

    Call :func:`~leakpro.attacks.gia_attacks.modular.config.builder.resolve_attack_config`
    with actual client observations to override ``training_simulator.epochs``
    and ``batch_size`` before building.

    Threat Model: Model C (Parameter-Aware Eavesdropper) — local hyperparameters known.

    Reference:
        Dimitrov, D. I., Balunovic, M., Konstantinov, N., & Vechev, M. (2022).
        Data Leakage in Federated Averaging. TMLR.
    """
    losses = [
        ComponentSpec(type="loss.gradient_matching", params={"loss_type": "cosine", "weight": 1.0}),
    ]
    if tv_weight > 0:
        losses.append(ComponentSpec(type="loss.tv", params={"weight": tv_weight}))

    return AttackConfig(
        threat_model_type="model_c",
        label_inference=ComponentSpec(type="label_inference.oracle"),
        epoch_handling="multi_epoch_separate",
        fedavg=FedAvgConfig(
            lambda_inv=lambda_inv,
            order_invariant_fn="mean",
            matching_metric="l2",
        ),
        stages=[
            OptimizerStageConfig(
                learning_rate=learning_rate,
                max_iterations=max_iterations,
                optimizer_type="adam",
                scheduler_type=scheduler_type,
                constraint=ComponentSpec(type="constraint.clip"),
                losses=losses,
            ),
        ],
        training_simulator=TrainingSimulatorConfig(
            epochs=epochs,
            batch_size=batch_size,
            model_mode="eval",
            compute_mode="updates",
        ),
    )
