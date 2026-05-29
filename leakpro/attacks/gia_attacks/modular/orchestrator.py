#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Orchestrator for modular gradient inversion attacks."""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from leakpro.attacks.gia_attacks.modular.core.callbacks import Callback, StopStage
from leakpro.attacks.gia_attacks.modular.core.component_base import (
    AggregationStrategy,
    InitializationStrategy,
    LabelInferenceResult,
    LabelInferenceStrategy,
)
from leakpro.attacks.gia_attacks.modular.core.stage import Stage
from leakpro.attacks.gia_attacks.modular.core.state import RunContext, WorkingState
from leakpro.attacks.gia_attacks.modular.core.threat_model import ThreatModel
from leakpro.fl_utils.fl_client_simulator import ClientObservations

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.epoch_strategies import EpochHandlingStrategy
    from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.training_simulator import MultiEpochTrainingSimulation
    from leakpro.attacks.gia_attacks.modular.components.transition_strategies import TransitionStrategy

logger = logging.getLogger(__name__)


class ModularGIAOrchestrator:
    """Coordinate a multi-stage gradient inversion attack.

    Owns the full pipeline:
    - bootstrap (label inference + initialization) → WorkingState
    - stage loop: stage.run(state, ctx, stage_idx, callbacks)
    - transitions between stages
    - checkpoint save/resume between stages
    - return_best_stage tracking (GIFD behavior)
    """

    def __init__(
        self,
        threat_model: ThreatModel,
        initialization: InitializationStrategy,
        stages: list[Stage],
        training_simulator: "MultiEpochTrainingSimulation",
        loss_fn: nn.Module | None = None,
        label_inference: LabelInferenceStrategy | None = None,
        transitions: "list[TransitionStrategy] | None" = None,
        seed_aggregation: AggregationStrategy | None = None,
        epoch_aggregation: AggregationStrategy | None = None,
        num_seeds_per_image: int = 1,
        epoch_handling_strategy: "EpochHandlingStrategy | None" = None,
        return_best_stage: bool = False,
        checkpoint_dir: str | None = None,
        seed: int = 42,
    ) -> None:
        self.threat_model = threat_model
        self.initialization = initialization
        self.stages = stages
        self.training_simulator = training_simulator
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.label_inference = label_inference
        self.transitions = transitions or []
        self.seed_aggregation = seed_aggregation
        self.epoch_aggregation = epoch_aggregation
        self.num_seeds_per_image = num_seeds_per_image
        self.epoch_handling_strategy = epoch_handling_strategy
        self.return_best_stage = return_best_stage
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.seed = seed

        logger.info(f"Initializing ModularGIAOrchestrator: {len(self.stages)} stage(s), threat_model={threat_model.name}")
        self._validate_components()

    def _validate_components(self) -> None:
        components = [self.initialization, *self.stages]
        if self.label_inference:
            components.append(self.label_inference)
        for component in components:
            metadata = component.get_metadata()
            is_allowed, missing = self.threat_model.allows_component(metadata.required_capabilities)
            if not is_allowed:
                raise ValueError(
                    f"Component '{metadata.name}' requires capabilities "
                    f"{missing} not available in threat model '{self.threat_model.name}'."
                )

    # ------------------------------------------------------------------
    # Build RunContext
    # ------------------------------------------------------------------

    def _build_context(
        self,
        target_model: nn.Module,
        client_observations: ClientObservations,
        proxy_dataloader: torch.utils.data.DataLoader | None,
        seed: int,
    ) -> RunContext:
        return RunContext(
            target_model=target_model,
            client_observations=client_observations,
            training_simulator=self.training_simulator,
            loss_fn=self.loss_fn,
            proxy_dataloader=proxy_dataloader,
            seed_aggregation=self.seed_aggregation,
            epoch_aggregation=self.epoch_aggregation,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Bootstrap: label inference + initialization → initial WorkingState
    # ------------------------------------------------------------------

    def _bootstrap(
        self,
        ctx: RunContext,
        input_shape: tuple[int, ...],
        device: torch.device,
    ) -> WorkingState:
        # Label inference
        label_result = None
        if self.label_inference is not None:
            label_result = self.label_inference.infer(ctx)
            logger.info(f"Label inference ({self.label_inference.get_metadata().name}): {label_result.labels.tolist()}")
        elif ctx.client_observations.labels is not None:
            label_result = LabelInferenceResult(
                labels=ctx.client_observations.labels, method="provided"
            )

        # Compute the shape we actually need to initialize
        expected_shape, label_result = self._compute_init_shape(input_shape, ctx, label_result)

        # Query representation strategy for parameter shape (first stage)
        param_shape = expected_shape
        first_stage = self.stages[0] if self.stages else None
        if first_stage is not None and hasattr(first_stage, "representation") and first_stage.representation is not None:
            param_shape = first_stage.representation.get_parameter_shape(expected_shape)

        init_result = self.initialization.initialize(shape=param_shape, device=device)
        logger.info(f"Initialization: shape={init_result.reconstruction.shape}")

        return WorkingState(
            reconstruction=init_result.reconstruction,
            labels=label_result,
        )

    def _compute_init_shape(
        self,
        input_shape: tuple[int, ...],
        ctx: RunContext,
        label_result: LabelInferenceResult | None,
    ) -> tuple[tuple[int, ...], LabelInferenceResult | None]:
        epochs = 1
        if ctx.client_observations.training_settings is not None:
            epochs = ctx.client_observations.training_settings.epochs
        if self.epoch_handling_strategy is None:
            return input_shape, label_result

        num_images = input_shape[0]
        channels, height, width = input_shape[1:]
        expected_shape = self.epoch_handling_strategy.get_expected_reconstruction_shape(
            num_images=num_images,
            num_epochs=epochs,
            num_seeds=self.num_seeds_per_image,
            input_shape=(channels, height, width),
        )
        if label_result is not None:
            label_result = self._reshape_labels_for_epochs(label_result, num_images, epochs)
        return expected_shape, label_result

    def _reshape_labels_for_epochs(
        self,
        label_result: LabelInferenceResult,
        num_images: int,
        epochs: int,
    ) -> LabelInferenceResult:
        label_shape = self.epoch_handling_strategy.get_expected_label_shape(
            num_images=num_images, num_epochs=epochs
        )
        label_type = label_result.label_type
        if label_result.labels.ndim == label_type.bare_ndim:
            target_epochs = label_shape[0]
            if target_epochs > 1:
                label_result.labels = label_type.expand_for_epochs(label_result.labels, target_epochs)
            else:
                label_result.labels = label_type.add_epoch_dim(label_result.labels)
        return label_result

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _checkpoint_path(self, stage_idx: int) -> Path | None:
        if self.checkpoint_dir is None:
            return None
        return self.checkpoint_dir / f"stage_{stage_idx}_output.pt"

    def _save_checkpoint(self, stage_idx: int, state: WorkingState) -> None:
        path = self._checkpoint_path(stage_idx)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        labels_tensor = None
        confidence_tensor = None
        if state.labels is not None:
            labels_tensor = state.labels.labels.detach().cpu()
            if state.labels.confidence is not None:
                confidence_tensor = state.labels.confidence.detach().cpu()
        torch.save({
            "reconstruction": state.reconstruction.detach().cpu() if state.reconstruction is not None else None,
            "optimizable_tensor": state.optimizable_tensor.detach().cpu() if state.optimizable_tensor is not None else None,
            "labels": labels_tensor,
            "confidence": confidence_tensor,
            "stage_idx": stage_idx,
            "loss": state.loss,
            "iteration": state.iteration,
        }, path)
        logger.info("Checkpoint saved: %s", path)

    def _load_checkpoint(self, stage_idx: int, device: torch.device) -> WorkingState | None:
        path = self._checkpoint_path(stage_idx)
        if path is None or not path.exists():
            return None
        saved = torch.load(path, map_location=device, weights_only=False)  # noqa: S614
        labels = None
        if saved.get("labels") is not None:
            confidence = saved.get("confidence")
            labels = LabelInferenceResult(
                labels=saved["labels"].to(device),
                confidence=confidence.to(device) if confidence is not None else None,
                method=f"checkpoint_stage_{stage_idx}",
            )
        return WorkingState(
            reconstruction=saved["reconstruction"].to(device) if saved.get("reconstruction") is not None else None,
            optimizable_tensor=saved["optimizable_tensor"].to(device) if saved.get("optimizable_tensor") is not None else None,
            labels=labels,
            loss=saved.get("loss", float("inf")),
            iteration=saved.get("iteration", 0),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_stage(self, index: int) -> Stage:
        """Return a single stage by index for standalone execution."""
        if index < 0 or index >= len(self.stages):
            raise IndexError(f"Stage index {index} out of range (0..{len(self.stages)-1})")
        return self.stages[index]

    def run_attack(
        self,
        target_model: nn.Module,
        client_observations: ClientObservations,
        device: torch.device | None = None,
        proxy_dataloader: torch.utils.data.DataLoader | None = None,
        input_shape: tuple[int, ...] | None = None,
        callbacks: list[Callback] | None = None,
        seed: int | None = None,
    ) -> tuple[WorkingState, RunContext]:
        """Run the full attack pipeline.

        Returns:
            (final_state, ctx) — WorkingState with the reconstruction and
            RunContext for inspection / replay.
        """
        if device is None:
            device = next(target_model.parameters()).device if len(list(target_model.parameters())) > 0 else torch.device("cpu")

        if input_shape is None:
            if client_observations.input_shape is None:
                raise ValueError("input_shape must be provided either as parameter or via client_observations.input_shape")
            input_shape = client_observations.input_shape

        effective_seed = seed if seed is not None else self.seed
        cbs: list[Callback] = list(callbacks) if callbacks else []
        target_model = deepcopy(target_model).to(device)

        ctx = self._build_context(target_model, client_observations, proxy_dataloader, effective_seed)
        state = self._bootstrap(ctx, input_shape, device)

        for cb in cbs:
            cb.on_attack_start(state, ctx)

        # Find latest checkpoint to resume from
        start_from = 0
        if self.checkpoint_dir is not None:
            for i in range(len(self.stages) - 1, 0, -1):
                if (self.checkpoint_dir / f"stage_{i}_output.pt").exists():
                    start_from = i
                    logger.info("Resuming from stage %d checkpoint", i)
                    break
            if start_from > 0:
                loaded = self._load_checkpoint(start_from, device)
                if loaded is not None:
                    state = loaded

        best_state: WorkingState | None = None

        for i, stage in enumerate(self.stages):
            if i < start_from:
                continue

            for cb in cbs:
                cb.on_stage_start(i, state, ctx)

            # Apply transition between stages
            if i > 0 and i - 1 < len(self.transitions):
                transition = self.transitions[i - 1]
                state = transition.apply(state, ctx)

            # Run the stage
            try:
                state = stage.run(state, ctx, stage_idx=i, callbacks=cbs)
            except StopStage:
                state.converged = False
                logger.info("Stage %d stopped early by callback", i)

            # Save checkpoint after non-final stages
            if i < len(self.stages) - 1:
                self._save_checkpoint(i, state)

            if self.return_best_stage:
                if best_state is None or state.loss < best_state.loss:
                    # Deep-copy labels to prevent joint label optimisation from
                    # mutating the saved best state through the shared tensor reference.
                    best_labels = None
                    if state.labels is not None:
                        best_labels = LabelInferenceResult(
                            labels=state.labels.labels.clone(),
                            confidence=(
                                state.labels.confidence.clone()
                                if state.labels.confidence is not None
                                else None
                            ),
                            method=state.labels.method,
                            label_type=state.labels.label_type,
                        )
                    best_state = WorkingState(
                        reconstruction=state.reconstruction.clone() if state.reconstruction is not None else None,
                        optimizable_tensor=state.optimizable_tensor.clone() if state.optimizable_tensor is not None else None,
                        labels=best_labels,
                        metrics=dict(state.metrics),
                        iteration=state.iteration,
                        loss=state.loss,
                        converged=state.converged,
                    )

            for cb in cbs:
                cb.on_stage_end(i, state, ctx)

            logger.info("Stage %d complete: loss=%.6f, converged=%s", i, state.loss, state.converged)

        final = best_state if (self.return_best_stage and best_state is not None) else state

        for cb in cbs:
            cb.on_attack_end(final, ctx)

        return final, ctx

    def __repr__(self) -> str:
        parts = [
            f"threat_model={self.threat_model.name}",
            f"stages={len(self.stages)}",
            f"initialization={self.initialization.get_metadata().name}",
        ]
        if self.label_inference:
            parts.append(f"label_inference={self.label_inference.get_metadata().name}")
        return f"ModularGIAOrchestrator({', '.join(parts)})"


__all__ = ["ModularGIAOrchestrator"]
