#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""GIFD: Gradient Inversion over Feature Domains (Fang et al., ICCV 2023)."""

from __future__ import annotations

import logging
from typing import Any

from leakpro.attacks.gia_attacks.modular.config.schema import AttackConfig, OptimizerStageConfig
from leakpro.attacks.gia_attacks.modular.config.spec import ComponentSpec

logger = logging.getLogger(__name__)


def gifd_attack(  # noqa: C901, PLR0912, PLR0915
    gan_checkpoint: str | None = None,
    huggingface_model: str | None = None,
    latent_dim: int = 100,
    num_intermediate_stages: int = 3,
    latent_iterations: int = 3000,
    feature_iterations: int = 1000,
    latent_lr: float = 0.1,
    feature_lr: float = 0.1,
    l1_radius: float | list[float] = 1e4,
    tv_weight: float = 1e-4,
    image_norm_weight: float = 1e-6,
    kld_weight: float = 0.0,
    optimizer_type: str = "adam",
    gan_architecture: str = "dcgan",
    conditional: bool = False,
    num_classes: int | None = None,
    img_size: int = 32,
    truncation: float = 0.4,
    custom_generator: Any = None,  # noqa: ANN401 — accepts any GAN generator duck-type
) -> tuple[AttackConfig, dict[str, Any]]:
    """GIFD — Gradient Inversion over Feature Domains, Fang et al. ICCV 2023.

    Progressively shifts the search from latent space to intermediate feature
    domains of the generator, recovering richer structure at each deeper layer.

    Returns:
        ``(config, live_overrides)`` — pass both to
        :meth:`~leakpro.attacks.gia_attacks.modular.config.builder.AttackBuilder.build`::

            config, overrides = gifd_attack(huggingface_model="csinva/cifar10_dcgan")
            orch = AttackBuilder.build(config, live_overrides=overrides, return_best_stage=True)

    Reference:
        Fang, H., et al. "GIFD: A Generative Gradient Inversion Method with Feature
        Domain Optimization." ICCV 2023.

    """
    import torch  # noqa: PLC0415

    from leakpro.attacks.gia_attacks.modular.components.representation_strategies import (  # noqa: PLC0415
        GANRepresentation,
        GIFDIntermediateRepresentation,
    )
    from leakpro.attacks.gia_attacks.modular.components.transition_strategies import GIFDLayerTransition  # noqa: PLC0415
    from leakpro.fl_utils.gan_handler import load_pretrained_gan  # noqa: PLC0415
    from leakpro.fl_utils.gan_models import LayeredGANWrapper  # noqa: PLC0415

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if custom_generator is not None:
        generator = custom_generator
        generator.eval()
        if not hasattr(generator, "num_layer_groups"):
            raise ValueError("custom_generator must expose 'num_layer_groups'. Wrap it in LayeredGANWrapper.")
        if hasattr(custom_generator, "latent_dim") and latent_dim == 100:
            latent_dim = custom_generator.latent_dim
    else:
        loaded = load_pretrained_gan(
            checkpoint_path=gan_checkpoint,
            huggingface_model=huggingface_model,
            device=device,
            architecture=gan_architecture,
            latent_dim=latent_dim,
            img_size=img_size,
            truncation=truncation,
        )
        if hasattr(loaded, "num_layer_groups") and hasattr(loaded, "forward_from"):
            generator = loaded
        else:
            generator = LayeredGANWrapper(loaded)
        generator.to(device).eval()
        if hasattr(loaded, "latent_dim") and latent_dim == 100:
            latent_dim = loaded.latent_dim

    if num_intermediate_stages > generator.num_layer_groups:
        raise ValueError(
            f"num_intermediate_stages={num_intermediate_stages} exceeds "
            f"generator.num_layer_groups={generator.num_layer_groups}."
        )

    # Normalise l1_radius to a per-stage list
    if isinstance(l1_radius, (int, float)):
        l1_radii = [float(l1_radius)] * num_intermediate_stages
    else:
        l1_radii = list(l1_radius)
        if len(l1_radii) != num_intermediate_stages:
            logger.warning(
                f"l1_radius list length ({len(l1_radii)}) != num_intermediate_stages "
                f"({num_intermediate_stages}).  Truncating/padding with last value."
            )
            if len(l1_radii) > num_intermediate_stages:
                l1_radii = l1_radii[:num_intermediate_stages]
            else:
                l1_radii += [l1_radii[-1]] * (num_intermediate_stages - len(l1_radii))

    # Stage 0: latent search
    stage0_losses = [
        ComponentSpec(type="loss.gradient_matching", params={"loss_type": "sim_cmpr0", "weight": 1.0}),
    ]
    if tv_weight > 0:
        stage0_losses.append(ComponentSpec(type="loss.tv", params={"weight": tv_weight}))
    if image_norm_weight > 0:
        stage0_losses.append(ComponentSpec(type="loss.l2", params={"weight": image_norm_weight}))
    if kld_weight > 0:
        stage0_losses.append(ComponentSpec(type="loss.latent_kld", params={"weight": kld_weight}))

    stage0_repr_obj = GANRepresentation(
        generator=generator, latent_dim=latent_dim,
        conditional=conditional, num_classes=num_classes,
    )

    stage_configs = [
        OptimizerStageConfig(
            representation=ComponentSpec(type="repr.gan_frozen", id="gifd_s0_repr"),
            constraint=ComponentSpec(type="constraint.spherical"),
            learning_rate=latent_lr,
            max_iterations=latent_iterations,
            optimizer_type=optimizer_type,
            scheduler_type="cosine_warmup",
            return_best=False,
            losses=stage0_losses,
        ),
    ]

    live_overrides: dict[str, Any] = {"gifd_s0_repr": stage0_repr_obj}
    transition_specs = []

    stage_losses = [
        ComponentSpec(type="loss.gradient_matching", params={"loss_type": "sim_cmpr0", "weight": 1.0}),
    ]
    if tv_weight > 0:
        stage_losses.append(ComponentSpec(type="loss.tv", params={"weight": tv_weight}))
    if image_norm_weight > 0:
        stage_losses.append(ComponentSpec(type="loss.l2", params={"weight": image_norm_weight}))

    for k in range(num_intermediate_stages):
        trans_id = f"gifd_t{k}"
        repr_id = f"gifd_s{k + 1}_repr"

        transition = GIFDLayerTransition(
            generator=generator, layer_idx=k, from_latent=(k == 0)
        )
        repr_obj = GIFDIntermediateRepresentation(generator=generator, start_layer=k + 1)

        live_overrides[trans_id] = transition
        live_overrides[repr_id] = repr_obj

        transition_specs.append(ComponentSpec(type="transition.gifd_layer", id=trans_id))

        radius = l1_radii[k]
        constraint_spec = (
            ComponentSpec(type="constraint.l1_ball", params={"radius": radius})
            if radius > 0 else None
        )

        stage_configs.append(
            OptimizerStageConfig(
                representation=ComponentSpec(type="repr.gifd_intermediate", id=repr_id),
                constraint=constraint_spec,
                learning_rate=feature_lr,
                max_iterations=feature_iterations,
                optimizer_type=optimizer_type,
                scheduler_type="cosine_warmup",
                return_best=False,
                losses=stage_losses,
            ),
        )

    config = AttackConfig(
        threat_model_type="model_b",
        label_inference=ComponentSpec(type="label_inference.oracle"),
        stages=stage_configs,
        transitions=transition_specs,
        return_best_stage=True,
    )
    return config, live_overrides
