#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Attack builder — instantiates a ModularGIAOrchestrator from an AttackConfig.

All PyTorch and component-instantiation logic lives here, keeping
:mod:`~leakpro.attacks.gia_attacks.modular.config.schema` as a pure-data module.

Typical usage::

    from leakpro.attacks.gia_attacks.modular.config.builder import AttackBuilder, resolve_attack_config
    from leakpro.attacks.gia_attacks.modular.config.schema import AttackConfig

    cfg = AttackConfig(...)

    # Optionally override training settings with client values before building:
    cfg = resolve_attack_config(cfg, client_observations)

    orchestrator = AttackBuilder.build(cfg)
    state, ctx = orchestrator.run_attack(model, client_observations)

For attacks that require pre-loaded non-serialisable objects (GAN generators,
custom loss functions), pass them via ``live_overrides``::

    gan_repr = GANRepresentation(generator=my_generator, ...)
    orchestrator = AttackBuilder.build(
        cfg,
        live_overrides={"stage0_repr": gan_repr},
    )
"""

from __future__ import annotations

import torch

from leakpro.attacks.gia_attacks.modular.components.cma_es_optimizer import CMAESOptimizer
from leakpro.attacks.gia_attacks.modular.components.composable_optimizer import ComposableOptimizer
from leakpro.attacks.gia_attacks.modular.components.diffusion_optimizer import DiffusionSamplingOptimizer
from leakpro.attacks.gia_attacks.modular.components.label_inference import JointLabelOptimization
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.consensus_strategies import (
    EpochMatchingConsensus,
)
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.epoch_strategies import (
    MultiEpochSeparate,
    SingleStorageReused,
)
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.label_strategies import (
    FixedLabels,
    JointLabelOptimizationStrategy,
    LabelStrategy,
)
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.loss_components import (
    EpochOrderInvariantPrior,
    GradientMatchingLoss,
    GroupConsistencyRegularization,
)
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.step_strategies import (
    StandardStepStrategy,
)
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.training_simulator import (
    MultiEpochTrainingSimulation,
    TrainingSettings,
)
from leakpro.attacks.gia_attacks.modular.components.transition_strategies import ReconstructionTransition
from leakpro.attacks.gia_attacks.modular.config.registry import build_component
from leakpro.attacks.gia_attacks.modular.config.schema import (
    AttackConfig,
    CMAESStageConfig,
    DiffusionStageConfig,
    FedAvgConfig,
    OptimizerStageConfig,
)
from leakpro.attacks.gia_attacks.modular.core.component_base import AggregationStrategy
from leakpro.attacks.gia_attacks.modular.core.threat_model import (
    model_a_eavesdropper,
    model_b_informed,
    model_c_parameter_aware,
    model_d_data_enhanced,
    model_e_statistical,
)
from leakpro.attacks.gia_attacks.modular.orchestrator import ModularGIAOrchestrator
from leakpro.fl_utils.diffusion_handler import load_pretrained_ddpm
from leakpro.fl_utils.fl_client_simulator import ClientObservations

# ---------------------------------------------------------------------------
# Config resolution (client override)
# ---------------------------------------------------------------------------

def resolve_attack_config(
    config: AttackConfig,
    client_observations: ClientObservations,
) -> AttackConfig:
    """Return a copy of *config* with ``training_simulator`` fields overridden from *client_observations*.

    If *client_observations* or its ``training_settings`` are ``None`` the
    original config is returned unchanged.

    Args:
        config: Base :class:`~leakpro.attacks.gia_attacks.modular.config.schema.AttackConfig`.
        client_observations: A :class:`~leakpro.fl_utils.fl_client_simulator.ClientObservations`
            instance (or any object with a ``training_settings`` attribute).

    Returns:
        A new :class:`~leakpro.attacks.gia_attacks.modular.config.schema.AttackConfig`
        whose ``training_simulator`` reflects the client's actual settings.

    """
    if client_observations is None:
        return config
    cs_settings = getattr(client_observations, "training_settings", None)
    if cs_settings is None:
        return config

    updates: dict[str, object] = {}
    if cs_settings.epochs is not None:
        updates["epochs"] = cs_settings.epochs
    if cs_settings.training_batch_size is not None:
        updates["batch_size"] = cs_settings.training_batch_size
    updates["compute_mode"] = cs_settings.compute_mode
    updates["model_mode"] = cs_settings.model_mode

    new_ts = config.training_simulator.model_copy(update=updates)
    return config.model_copy(update={"training_simulator": new_ts})


# ---------------------------------------------------------------------------
# Internal: loss building
# ---------------------------------------------------------------------------

def _build_loss_list(
    specs: list,
    training_simulator: MultiEpochTrainingSimulation,
    seed_aggregation: AggregationStrategy,
    *,
    fedavg: FedAvgConfig | None = None,
    training_settings: TrainingSettings | None = None,
    live_overrides: dict | None = None,
) -> list:
    """Build loss components from ComponentSpec list, injecting runtime objects.

    Args:
        specs: List of :class:`~leakpro.attacks.gia_attacks.modular.config.spec.ComponentSpec`.
        training_simulator: Injected into :class:`GradientMatchingLoss` after construction.
        seed_aggregation: Injected into :class:`GroupConsistencyRegularization` after construction.
        fedavg: Optional :class:`FedAvgConfig` — adds ``EpochOrderInvariantPrior`` when
            ``lambda_inv > 0`` and ``training_settings.epochs > 1``.
        training_settings: Provides ``epochs`` count for the prior above.
        live_overrides: Forwarded to ``build_component``.

    Returns:
        List of instantiated loss component objects.

    """
    losses = []
    for spec in specs:
        loss = build_component(spec, live_overrides=live_overrides)
        if isinstance(loss, GradientMatchingLoss):
            loss.training_simulator = training_simulator
        if isinstance(loss, GroupConsistencyRegularization) and seed_aggregation is not None:
            loss.seed_aggregation = seed_aggregation
        losses.append(loss)

    # FedAvg epoch-order-invariant prior (automatically added from fedavg config)
    if (
        fedavg is not None
        and fedavg.lambda_inv > 0
        and training_settings is not None
        and training_settings.epochs > 1
    ):
        losses.append(
            EpochOrderInvariantPrior(
                order_invariant_function=fedavg.order_invariant_fn,
                distance_function="l2",
                weight=fedavg.lambda_inv,
                epochs=training_settings.epochs,
            )
        )

    return losses


# ---------------------------------------------------------------------------
# AttackBuilder
# ---------------------------------------------------------------------------

class AttackBuilder:
    """Builds a :class:`~leakpro.attacks.gia_attacks.modular.orchestrator.ModularGIAOrchestrator`.

    Converts a serialisable :class:`~leakpro.attacks.gia_attacks.modular.config.schema.AttackConfig`
    into a runnable orchestrator.  All methods are static — this class is a namespace, not a
    stateful object.
    """

    @staticmethod
    def build(
        config: AttackConfig,
        *,
        live_overrides: dict[str, object] | None = None,
        client_observations: ClientObservations | None = None,
    ) -> ModularGIAOrchestrator:
        """Build a :class:`ModularGIAOrchestrator` from *config*.

        Args:
            config: Fully-specified :class:`AttackConfig`.  Call
                :func:`resolve_attack_config` first if you want client
                observations to override training settings.
            live_overrides: Map of ``ComponentSpec.id → pre-built object``.
                Use this to inject already-loaded models (GAN generators,
                custom representations) without serialising them.
            client_observations: Convenience shorthand — if provided,
                :func:`resolve_attack_config` is called internally before
                building.

        Returns:
            A :class:`ModularGIAOrchestrator` ready for :meth:`run_attack`.

        """
        # Apply client overrides if provided
        if client_observations is not None:
            config = resolve_attack_config(config, client_observations)

        # Build threat model
        _threat_factories = {
            "model_a": model_a_eavesdropper,
            "model_b": model_b_informed,
            "model_c": model_c_parameter_aware,
            "model_d": model_d_data_enhanced,
            "model_e": model_e_statistical,
        }
        threat_model = _threat_factories[config.threat_model_type]()

        # Build label inference and default label strategy
        label_inference = build_component(config.label_inference, live_overrides=live_overrides)
        initialization = build_component(config.initialization, live_overrides=live_overrides)
        default_label_strategy: LabelStrategy = (
            JointLabelOptimizationStrategy()
            if isinstance(label_inference, JointLabelOptimization)
            else FixedLabels()
        )

        # Build epoch handling strategy
        ts_cfg = config.training_simulator
        epoch_strat = (
            MultiEpochSeparate()
            if config.epoch_handling == "multi_epoch_separate"
            else SingleStorageReused()
        )

        # Build training simulator
        training_settings = TrainingSettings(
            epochs=ts_cfg.epochs,
            optimizer_type="sgd",
            training_batch_size=ts_cfg.batch_size,
            compute_mode=ts_cfg.compute_mode,
            model_mode=ts_cfg.model_mode,
            shuffle_mode=ts_cfg.mode,
        )
        training_simulator = MultiEpochTrainingSimulation(
            epochs=training_settings.epochs,
            optimizer_type=training_settings.optimizer_type,
            batch_size=training_settings.training_batch_size,
            compute_mode=training_settings.compute_mode,
            model_mode=training_settings.model_mode,
            shuffle_mode="attack",
            epoch_handling_strategy=epoch_strat,
            meta_learning_rate=ts_cfg.meta_learning_rate,
        )

        # Build seed / epoch aggregation
        seed_aggregation = build_component(config.seed_aggregation, live_overrides=live_overrides)
        epoch_aggregation = None
        if training_settings.epochs > 1 and config.epoch_handling == "multi_epoch_separate":
            metric = config.fedavg.matching_metric if config.fedavg else "l2"
            epoch_aggregation = EpochMatchingConsensus(epochs=training_settings.epochs, metric=metric)

        # Pre-load diffusion models (shared across stages with the same URI)
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        preloaded_diffusion: dict[str, tuple] = {}
        for stage in config.stages:
            if isinstance(stage, DiffusionStageConfig):
                uri = stage.diffusion_model_uri
                if uri not in preloaded_diffusion:
                    preloaded_diffusion[uri] = load_pretrained_ddpm(uri, device=_device)

        # Build stages
        built_stages = []
        for stage in config.stages:
            if isinstance(stage, OptimizerStageConfig):
                built_stages.append(
                    AttackBuilder._build_optimizer_stage(
                        stage, config, training_simulator, seed_aggregation,
                        training_settings, default_label_strategy,
                        live_overrides=live_overrides,
                    )
                )
            elif isinstance(stage, DiffusionStageConfig):
                built_stages.append(
                    AttackBuilder._build_diffusion_stage(
                        stage, config, training_simulator, seed_aggregation,
                        training_settings,
                        live_overrides=live_overrides,
                        preloaded_models=preloaded_diffusion,
                    )
                )
            elif isinstance(stage, CMAESStageConfig):
                built_stages.append(
                    AttackBuilder._build_cma_es_stage(
                        stage, config, training_simulator, seed_aggregation,
                        training_settings,
                        live_overrides=live_overrides,
                    )
                )
            else:
                raise TypeError(f"Unknown stage config type: {type(stage)}")

        # Build transitions (default: ReconstructionTransition between every pair)
        if config.transitions:
            built_transitions = [
                build_component(t, live_overrides=live_overrides) for t in config.transitions
            ]
        else:
            built_transitions = [ReconstructionTransition() for _ in range(len(built_stages) - 1)]

        return ModularGIAOrchestrator(
            threat_model=threat_model,
            initialization=initialization,
            stages=built_stages,
            training_simulator=training_simulator,
            label_inference=label_inference,
            transitions=built_transitions,
            seed_aggregation=seed_aggregation,
            epoch_aggregation=epoch_aggregation,
            num_seeds_per_image=config.num_seeds_per_image,
            epoch_handling_strategy=epoch_strat,
            return_best_stage=config.return_best_stage,
            checkpoint_dir=config.checkpoint_dir,
            seed=config.seed,
        )

    @staticmethod
    def _build_optimizer_stage(
        stage: OptimizerStageConfig,
        config: AttackConfig,
        training_simulator: MultiEpochTrainingSimulation,
        seed_aggregation: AggregationStrategy,
        training_settings: TrainingSettings,
        default_label_strategy: LabelStrategy,
        *,
        live_overrides: dict | None = None,
    ) -> ComposableOptimizer:
        """Build one gradient-based optimizer stage from config."""
        losses = _build_loss_list(
            stage.losses, training_simulator, seed_aggregation,
            fedavg=config.fedavg,
            training_settings=training_settings,
            live_overrides=live_overrides,
        )

        representation = (
            build_component(stage.representation, live_overrides=live_overrides)
            if stage.representation else None
        )

        # step_strategy and label_strategy always have a ComponentSpec default
        # (set by OptimizerStageConfig field defaults / validators), so we can
        # always call build_component without special-casing None.
        step_strat = build_component(stage.step_strategy, live_overrides=live_overrides)
        label_strat = build_component(stage.label_strategy, live_overrides=live_overrides)
        constraint = (
            build_component(stage.constraint, live_overrides=live_overrides)
            if stage.constraint else None
        )

        return ComposableOptimizer(
            loss_components=losses,
            representation=representation,
            constraint=constraint,
            label_strategy=label_strat,
            step_strategy=step_strat,
            learning_rate=stage.learning_rate,
            label_learning_rate=stage.label_learning_rate or stage.learning_rate,
            max_iterations=stage.max_iterations,
            optimizer_type=stage.optimizer_type,
            scheduler_type=stage.scheduler_type,
            patience=stage.patience,
            log_interval=stage.log_interval,
            training_simulator=training_simulator,
            seed_aggregation=seed_aggregation if config.num_seeds_per_image > 1 else None,
            freeze_input=stage.freeze_input,
            return_best=stage.return_best,
            verbose=stage.verbose,
        )

    @staticmethod
    def _build_diffusion_stage(
        stage: DiffusionStageConfig,
        config: AttackConfig,
        training_simulator: MultiEpochTrainingSimulation,
        seed_aggregation: AggregationStrategy,
        training_settings: TrainingSettings,
        *,
        live_overrides: dict | None = None,
        preloaded_models: dict | None = None,
    ) -> DiffusionSamplingOptimizer:
        """Build one diffusion stage from config.

        Mean and noise strategies are built via the component registry so that
        new strategies can be registered without editing this method
        (Open-Closed Principle).
        """
        if preloaded_models and stage.diffusion_model_uri in preloaded_models:
            unet, ddpm_scheduler = preloaded_models[stage.diffusion_model_uri]
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            unet, ddpm_scheduler = load_pretrained_ddpm(stage.diffusion_model_uri, device=device)

        losses = _build_loss_list(
            stage.losses, training_simulator, seed_aggregation,
            fedavg=config.fedavg,
            training_settings=training_settings,
            live_overrides=live_overrides,
        )

        mean_strat = build_component(stage.mean_strategy, live_overrides=live_overrides)
        noise_strat = build_component(stage.noise_strategy, live_overrides=live_overrides)

        return DiffusionSamplingOptimizer(
            unet=unet,
            scheduler=ddpm_scheduler,
            mean_strategy=mean_strat,
            noise_strategy=noise_strat,
            loss_components=losses,
            log_interval=stage.log_interval,
            start_t_frac=stage.start_t_frac,
            stop_t_frac=stage.stop_t_frac,
            sampler=stage.sampler,
            num_steps=stage.num_steps,
        )

    @staticmethod
    def _build_cma_es_stage(
        stage: CMAESStageConfig,
        config: AttackConfig,
        training_simulator: MultiEpochTrainingSimulation,
        seed_aggregation: AggregationStrategy,
        training_settings: TrainingSettings,
        *,
        live_overrides: dict | None = None,
    ) -> CMAESOptimizer:
        """Build one CMA-ES stage from its dedicated config."""
        losses = _build_loss_list(
            stage.losses, training_simulator, seed_aggregation,
            fedavg=config.fedavg,
            training_settings=training_settings,
            live_overrides=live_overrides,
        )

        representation = (
            build_component(stage.representation, live_overrides=live_overrides)
            if stage.representation else None
        )

        return CMAESOptimizer(
            loss_components=losses,
            seed_aggregation=seed_aggregation if config.num_seeds_per_image > 1 else None,
            log_interval=stage.log_interval,
            max_iterations=stage.max_iterations,
            representation=representation,
            cma_population_size=stage.cma_population_size,
            cma_kld=stage.cma_kld,
            cma_cost_fn=stage.cma_cost_fn,
        )


__all__ = ["AttackBuilder", "resolve_attack_config"]
