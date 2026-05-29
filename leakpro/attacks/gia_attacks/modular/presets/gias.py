#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""GIAS: Gradient Inversion with Generative Image Prior (Jeon et al., NeurIPS 2021)."""

from __future__ import annotations

from typing import Any

from leakpro.attacks.gia_attacks.modular.config.schema import AttackConfig, OptimizerStageConfig
from leakpro.attacks.gia_attacks.modular.config.spec import ComponentSpec


def gias_attack(
    gan_checkpoint: str | None = None,
    huggingface_model: str | None = None,
    latent_dim: int = 512,
    stage1_iterations: int = 1000,
    stage2_iterations: int = 500,
    stage1_lr: float = 0.1,
    stage2_lr: float = 0.01,
    tv_weight: float = 1e-4,
    optimizer_type: str = "adam",
    gan_architecture: str = "stylegan2",
    loss_type: str = "cosine",
    lr_decay: bool = False,
    conditional: bool = False,
    num_classes: int | None = None,
    img_size: int = 32,
    custom_generator: Any = None,  # noqa: ANN401 — accepts any GAN generator duck-type
    stage1_return_best: bool = False,
    stage2_return_best: bool = False,
) -> tuple[AttackConfig, dict[str, Any]]:
    """GIAS — Gradient Inversion with Generative Image Prior, Jeon et al. NeurIPS 2021.

    Two-stage attack:
    1. Stage 1: Optimise latent code **z** in the frozen GAN latent space.
    2. Stage 2: Fine-tune the generator so its manifold better matches the target.

    Returns:
        ``(config, live_overrides)`` — pass both to
        :meth:`~leakpro.attacks.gia_attacks.modular.config.builder.AttackBuilder.build`::

            config, overrides = gias_attack(huggingface_model="brownvc/R3GAN-CIFAR10")
            orch = AttackBuilder.build(config, live_overrides=overrides)

    Reference:
        Jeon et al., "Gradient Inversion with Generative Image Prior", NeurIPS 2021.

    """
    import torch  # noqa: PLC0415

    from leakpro.attacks.gia_attacks.modular.components.representation_strategies import (  # noqa: PLC0415
        GANRepresentation,
        UnfrozenGANRepresentation,
    )
    from leakpro.fl_utils.gan_handler import load_pretrained_gan  # noqa: PLC0415

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if custom_generator is not None:
        generator = custom_generator
        generator.eval()
        if hasattr(custom_generator, "latent_dim"):
            latent_dim = custom_generator.latent_dim
    else:
        generator = load_pretrained_gan(
            checkpoint_path=gan_checkpoint,
            huggingface_model=huggingface_model,
            device=device,
            architecture=gan_architecture,
            latent_dim=latent_dim,
            img_size=img_size,
        )

    stage1_repr = GANRepresentation(
        generator=generator,
        latent_dim=latent_dim,
        conditional=conditional,
        num_classes=num_classes,
    )
    stage2_repr = UnfrozenGANRepresentation(
        generator=generator,
        latent_dim=latent_dim,
        conditional=conditional,
        num_classes=num_classes,
    )

    scheduler_type = "step" if lr_decay else None
    losses = [
        ComponentSpec(type="loss.gradient_matching", params={"loss_type": loss_type, "weight": 1.0}),
    ]
    if tv_weight > 0:
        losses.append(ComponentSpec(type="loss.tv", params={"weight": tv_weight}))

    config = AttackConfig(
        threat_model_type="model_b",
        label_inference=ComponentSpec(type="label_inference.oracle"),
        stages=[
            OptimizerStageConfig(
                representation=ComponentSpec(type="repr.gan_frozen", id="gias_s0_repr"),
                learning_rate=stage1_lr,
                max_iterations=stage1_iterations,
                optimizer_type=optimizer_type,
                scheduler_type=scheduler_type,
                return_best=stage1_return_best,
                losses=losses,
            ),
            OptimizerStageConfig(
                representation=ComponentSpec(type="repr.gan_unfrozen", id="gias_s1_repr"),
                learning_rate=stage2_lr,
                max_iterations=stage2_iterations,
                optimizer_type=optimizer_type,
                scheduler_type=scheduler_type,
                freeze_input=True,
                return_best=stage2_return_best,
                losses=losses,
            ),
        ],
        transitions=[ComponentSpec(type="transition.latent_code")],
    )

    live_overrides = {
        "gias_s0_repr": stage1_repr,
        "gias_s1_repr": stage2_repr,
    }
    return config, live_overrides
