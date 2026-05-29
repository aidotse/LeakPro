#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""GGL: Gradient Gradient Leakage via CMA-ES (Li et al., 2020)."""

from __future__ import annotations

from typing import Any

from leakpro.attacks.gia_attacks.modular.config.schema import AttackConfig, OptimizerStageConfig
from leakpro.attacks.gia_attacks.modular.config.spec import ComponentSpec


def ggl_attack(
    gan_checkpoint: str | None = None,
    huggingface_model: str | None = None,
    latent_dim: int = 128,
    max_iterations: int = 800,
    cost_fn: str = "l2",
    kld: float = 0.1,
    num_samples: int | None = None,
    conditional: bool = False,
    num_classes: int | None = None,
    img_size: int = 64,
    custom_generator: Any = None,  # noqa: ANN401 — accepts any GAN generator duck-type
) -> tuple[AttackConfig, dict[str, Any]]:
    """GGL — Gradient Gradient Leakage, Li et al. 2020.

    Optimises in GAN latent space using CMA-ES (gradient-free evolutionary
    strategy) instead of gradient descent.  Robust to gradient masking defences.

    Returns:
        ``(config, live_overrides)`` — pass both to
        :meth:`~leakpro.attacks.gia_attacks.modular.config.builder.AttackBuilder.build`::

            config, overrides = ggl_attack(huggingface_model="openai/BigGAN-deep-128-ImageNet", ...)
            orch = AttackBuilder.build(config, live_overrides=overrides)

    Reference:
        Li et al., "When Does Data Augmentation Help With Membership Inference
        Attacks?", 2021; used in Fang et al. GIFD 2023 as baseline.

    """
    import torch  # noqa: PLC0415

    from leakpro.attacks.gia_attacks.modular.components.representation_strategies import GANRepresentation  # noqa: PLC0415
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
            architecture="biggan" if conditional else "stylegan2",
            latent_dim=latent_dim,
            img_size=img_size,
            num_classes=num_classes if conditional else None,
        )

    representation = GANRepresentation(
        generator=generator,
        latent_dim=latent_dim,
        conditional=conditional,
        num_classes=num_classes if conditional else None,
    )

    config = AttackConfig(
        threat_model_type="model_b",
        label_inference=ComponentSpec(type="label_inference.oracle"),
        stages=[
            OptimizerStageConfig(
                optimizer_type="cma_es",
                max_iterations=max_iterations,
                cma_population_size=num_samples,
                cma_kld=kld,
                cma_cost_fn=cost_fn,
                representation=ComponentSpec(type="repr.gan_frozen", id="ggl_repr"),
                losses=[
                    ComponentSpec(type="loss.gradient_matching", params={"loss_type": cost_fn, "weight": 1.0}),
                ],
            ),
        ],
    )

    live_overrides = {"ggl_repr": representation}
    return config, live_overrides
