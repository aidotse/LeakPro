#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Pydantic configuration schema for the modular GIA framework.

This module is the authoritative source for serialisable attack configuration.
All fields map 1-to-1 to hyperparameters an HPO sweep might want to vary.
Non-serialisable runtime objects (``loss_fn``, ``target_model``, loaded GAN
generators, etc.) belong on :class:`~leakpro.attacks.gia_attacks.modular.core.state.RunContext`
or are passed as ``live_overrides`` to :func:`~leakpro.attacks.gia_attacks.modular.config.builder.AttackBuilder.build`.

This module contains **zero** PyTorch imports or component-instantiation logic.
All build logic lives in :mod:`leakpro.attacks.gia_attacks.modular.config.builder`.

Quickstart::

    from leakpro.attacks.gia_attacks.modular.config.schema import AttackConfig, OptimizerStageConfig
    from leakpro.attacks.gia_attacks.modular.config.spec import ComponentSpec
    from leakpro.attacks.gia_attacks.modular.config.builder import AttackBuilder

    cfg = AttackConfig(
        threat_model_type="model_b",
        stages=[OptimizerStageConfig(
            learning_rate=0.1,
            max_iterations=4800,
            use_gradient_sign=True,
            losses=[
                ComponentSpec(type="loss.gradient_matching", params={"loss_type": "cosine", "weight": 1.0}),
                ComponentSpec(type="loss.tv", params={"weight": 1e-3}),
            ],
        )],
    )
    orch = AttackBuilder.build(cfg, client_observations=obs)

JSON round-trip::

    json_str = cfg.model_dump_json()
    cfg2 = AttackConfig.model_validate_json(json_str)
    assert cfg.model_dump() == cfg2.model_dump()
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, field_validator, model_validator

from leakpro.attacks.gia_attacks.modular.config.spec import ComponentSpec


def _default_device() -> str:
    """Return 'cuda' if a CUDA-capable GPU is available, else 'cpu'."""
    try:
        import torch  # noqa: PLC0415
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class TrainingSimulatorConfig(BaseModel):
    """Settings for the client training simulation.

    These are the *final, intended* values — no ``use_client_*`` flags.
    To apply client observations call
    :func:`~leakpro.attacks.gia_attacks.modular.config.builder.resolve_attack_config`
    before building.
    """

    epochs: int = 1
    compute_mode: Literal["gradients", "updates"] = "gradients"
    model_mode: Literal["eval", "train"] = "eval"
    mode: Literal["attack", "client"] = "attack"
    batch_size: int | None = None
    meta_learning_rate: float = 0.01
    model_config = ConfigDict(extra="forbid")


class FedAvgConfig(BaseModel):
    """FedAvg-specific parameters (Dimitrov attack)."""

    lambda_inv: float = 0.0
    order_invariant_fn: Literal["mean", "sum", "variance"] = "mean"
    matching_metric: Literal["l2", "cosine"] = "l2"
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Stage configs
# ---------------------------------------------------------------------------

class OptimizerStageConfig(BaseModel):
    """Configuration for a single gradient-descent optimization stage.

    All loss components are specified via the ``losses`` list of
    :class:`~leakpro.attacks.gia_attacks.modular.config.spec.ComponentSpec` objects.
    The builder iterates this list and calls ``build_component(spec)`` for
    each entry — adding a new loss requires only a ``@register`` decorator
    on the loss class and a new entry in the preset's losses list.

    Example losses list::

        losses=[
            ComponentSpec(type="loss.gradient_matching", params={"loss_type": "cosine", "weight": 1.0}),
            ComponentSpec(type="loss.tv", params={"weight": 1e-3}),
            ComponentSpec(type="loss.bn_stats", params={"strategy": "running", "weight": 1e-4}),
        ]
    """

    kind: Literal["optimizer"] = "optimizer"

    # Core optimizer settings
    learning_rate: float = 0.1
    label_learning_rate: float | None = None
    max_iterations: int = 1000
    optimizer_type: ComponentSpec | None = Field(
        default_factory=lambda: ComponentSpec(type="optimizer.adam")
    )
    scheduler_type: ComponentSpec | None = None
    patience: int = 10000

    # All loss components specified via ComponentSpec (builder calls build_component for each)
    losses: list[ComponentSpec] = Field(default_factory=list)

    # Component specs for other pipeline elements
    representation: ComponentSpec | None = None
    constraint: ComponentSpec | None = None

    # Step and label strategies always have a default spec; the builder always
    # calls build_component() without special-casing None.
    label_strategy: ComponentSpec = Field(
        default_factory=lambda: ComponentSpec(type="label_strategy.fixed")
    )
    step_strategy: ComponentSpec = Field(
        default_factory=lambda: ComponentSpec(type="step.standard")
    )

    # Gradient settings — injected into step_strategy.params when using the
    # default StandardStepStrategy (via _wire_step_strategy_params validator).
    use_gradient_sign: bool = False
    gradient_noise_std: float = 0.0

    # Misc
    freeze_input: bool = False
    return_best: bool = True
    log_interval: int = 100
    verbose: bool = False

    model_config = ConfigDict(extra="forbid")

    @field_validator("optimizer_type", mode="before")
    @classmethod
    def _coerce_optimizer_type(cls, v: object) -> object:
        """Accept legacy string shortcuts ('adam', 'sgd', 'lbfgs') and convert to ComponentSpec."""
        _shortcuts = {"adam": "optimizer.adam", "sgd": "optimizer.sgd", "lbfgs": "optimizer.lbfgs"}
        if isinstance(v, str):
            key = _shortcuts.get(v, f"optimizer.{v}")
            return ComponentSpec(type=key)
        return v

    @field_validator("scheduler_type", mode="before")
    @classmethod
    def _coerce_scheduler_type(cls, v: object) -> object:
        """Accept legacy string shortcuts and convert to ComponentSpec."""
        _shortcuts = {
            "cosine": "scheduler.cosine",
            "step": "scheduler.step",
            "exponential": "scheduler.exponential",
            "cosine_warmup": "scheduler.cosine_warmup",
        }
        if isinstance(v, str):
            key = _shortcuts.get(v, f"scheduler.{v}")
            return ComponentSpec(type=key)
        return v

    @model_validator(mode="after")
    def _wire_step_strategy_params(self) -> "OptimizerStageConfig":
        """Inject use_gradient_sign / gradient_noise_std into step_strategy params.

        When the caller sets top-level ``use_gradient_sign=True`` (the old API)
        and the step strategy is the default StandardStepStrategy, we propagate
        the values into the spec so the builder can use ``build_component``
        unconditionally.
        """
        if self.step_strategy.type == "step.standard":
            params = dict(self.step_strategy.params)
            if "use_gradient_sign" not in params and self.use_gradient_sign:
                params["use_gradient_sign"] = self.use_gradient_sign
            if "gradient_noise_std" not in params and self.gradient_noise_std:
                params["gradient_noise_std"] = self.gradient_noise_std
            if params != self.step_strategy.params:
                self.step_strategy = ComponentSpec(type="step.standard", params=params)
        return self


class CMAESStageConfig(BaseModel):
    """Configuration for a CMA-ES (gradient-free) optimization stage.

    CMA-ES parameters are now isolated here instead of being mixed into
    :class:`OptimizerStageConfig`, preventing gradient-based and gradient-free
    settings from sharing a single schema.

    Example::

        CMAESStageConfig(
            representation=ComponentSpec(type="repr.gan", id="gan"),
            cma_kld=0.1,
            cma_cost_fn="l2",
            max_iterations=5000,
        )
    """

    kind: Literal["cma_es"] = "cma_es"

    # All loss components specified via ComponentSpec
    losses: list[ComponentSpec] = Field(default_factory=list)

    # Optional latent-space representation
    representation: ComponentSpec | None = None

    # CMA-ES hyper-parameters
    cma_population_size: int | None = None
    cma_kld: float = 0.1
    cma_cost_fn: Literal["l2", "cosine", "sim_cmpr0"] = "l2"

    max_iterations: int = 5000
    log_interval: int = 100

    model_config = ConfigDict(extra="forbid")


class DiffusionStageConfig(BaseModel):
    """Configuration for a diffusion-sampling optimization stage.

    ``mean_strategy`` and ``noise_strategy`` are now :class:`ComponentSpec`
    objects so new strategies can be registered without touching this schema
    (Open-Closed Principle).  Their hyperparameters live inside the spec's
    ``params`` dict rather than as top-level fields.

    Default specs reproduce the original GGDM behaviour::

        mean_strategy=ComponentSpec(
            type="mean.similarity_guidance",
            params={"gamma": 100.0, "grad_clip": 1.0},
        )
        noise_strategy=ComponentSpec(type="noise.standard")

    To use GradInvDiff (AMO + GANI)::

        mean_strategy=ComponentSpec(
            type="mean.adaptive_optimization",
            params={"inner_steps": 5, "inner_lr": 0.01},
        )
        noise_strategy=ComponentSpec(type="noise.gani")
    """

    kind: Literal["diffusion"] = "diffusion"

    # Diffusion model
    diffusion_model_uri: str = "google/ddpm-cifar10-32"

    # Mean adjustment strategy — params forwarded to the registered factory
    mean_strategy: ComponentSpec = Field(
        default_factory=lambda: ComponentSpec(
            type="mean.similarity_guidance",
            params={"gamma": 100.0, "grad_clip": 1.0},
        )
    )

    # Noise injection strategy — params forwarded to the registered factory
    noise_strategy: ComponentSpec = Field(
        default_factory=lambda: ComponentSpec(type="noise.standard")
    )

    # Sampler settings
    sampler: Literal["ddpm", "ddim"] = "ddpm"
    num_steps: int | None = None
    start_t_frac: float | None = None
    stop_t_frac: float | None = None

    # All loss components specified via ComponentSpec
    losses: list[ComponentSpec] = Field(default_factory=list)

    log_interval: int = 100

    model_config = ConfigDict(extra="forbid")


StageConfig = Annotated[
    OptimizerStageConfig | DiffusionStageConfig | CMAESStageConfig,
    Discriminator("kind"),
]


# ---------------------------------------------------------------------------
# Top-level AttackConfig
# ---------------------------------------------------------------------------

class AttackConfig(BaseModel):
    """Top-level serialisable attack configuration.

    Every field maps to a hyperparameter an HPO sweep might vary.
    Non-serialisable runtime objects (``loss_fn``, loaded GAN generators)
    are provided via ``live_overrides`` to
    :func:`~leakpro.attacks.gia_attacks.modular.config.builder.AttackBuilder.build`.

    Attributes:
        schema_version: Bumped on breaking changes; mismatched versions fail
            at parse time.
        stages: One or more stage configs.  Single-stage attacks use a
            one-element list.
        transitions: One :class:`ComponentSpec` per pair of consecutive
            stages (length == max(0, len(stages) - 1)).  Defaults to
            ``ReconstructionTransition`` between every pair.
        training_simulator: Final, intended training settings.  Apply client
            observations first via
            :func:`~leakpro.attacks.gia_attacks.modular.config.builder.resolve_attack_config`.

    """

    schema_version: Literal[1] = 1
    threat_model_type: Literal["model_a", "model_b", "model_c", "model_d", "model_e"] = "model_a"
    device: str = Field(default_factory=_default_device)

    label_inference: ComponentSpec = Field(
        default_factory=lambda: ComponentSpec(type="label_inference.oracle")
    )
    initialization: ComponentSpec = Field(
        default_factory=lambda: ComponentSpec(type="init.random_noise")
    )
    stages: list[StageConfig]
    transitions: list[ComponentSpec] = Field(default_factory=list)
    return_best_stage: bool = False

    num_seeds_per_image: int = 1
    seed_aggregation: ComponentSpec = Field(
        default_factory=lambda: ComponentSpec(type="aggregation.none")
    )
    epoch_handling: Literal["repeated_same", "multi_epoch_separate"] = "repeated_same"

    training_simulator: TrainingSimulatorConfig = Field(
        default_factory=TrainingSimulatorConfig
    )
    fedavg: FedAvgConfig | None = None
    checkpoint_dir: str | None = None

    seed: int = 42

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _check_transitions(self) -> "AttackConfig":
        expected = max(0, len(self.stages) - 1)
        if len(self.transitions) not in (0, expected):
            raise ValueError(
                f"'transitions' must have length {expected} (one per consecutive stage pair) "
                f"or 0 (auto-default to ReconstructionTransition); "
                f"got {len(self.transitions)}."
            )
        return self


__all__ = [
    "AttackConfig",
    "CMAESStageConfig",
    "DiffusionStageConfig",
    "FedAvgConfig",
    "OptimizerStageConfig",
    "StageConfig",
    "TrainingSimulatorConfig",
]
