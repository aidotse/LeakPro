"""Orchestrator for modular gradient inversion attacks.

The orchestrator coordinates all components of a gradient inversion attack:
1. Label inference (if needed)
2. Initialization
3. Optimization

It handles the complete attack pipeline and manages component interactions.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING

import torch
from torch import nn

from leakpro.attacks.gia_attacks.modular.core.component_base import (
    InitializationStrategy,
    LabelInferenceResult,
    LabelInferenceStrategy,
    OptimizationState,
    OptimizationStrategy,
)
from leakpro.attacks.gia_attacks.modular.core.threat_model import ThreatModel
from leakpro.fl_utils.fl_client_simulator import ClientObservations

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.epoch_strategies import EpochHandlingStrategy

logger = logging.getLogger(__name__)


class ModularGIAOrchestrator:
    """Orchestrate modular gradient inversion attacks.

    This class coordinates all components of an attack:
    - Validates components against threat model
    - Manages attack pipeline execution
    - Handles component interactions and data flow
    """

    def __init__(
        self,
        threat_model: ThreatModel,
        initialization: InitializationStrategy,
        optimization: OptimizationStrategy,
        label_inference: LabelInferenceStrategy | None = None,
        num_seeds_per_image: int = 1,
        epoch_handling_strategy: "EpochHandlingStrategy | None" = None,
    ) -> None:
        """Initialize orchestrator.

        Args:
            threat_model: Threat model defining attacker capabilities
            initialization: Initialization strategy
            optimization: Optimization strategy
            label_inference: Optional label inference strategy
            num_seeds_per_image: Number of seeds per image for multi-seed optimization
            epoch_handling_strategy: Strategy for handling epochs during reconstruction

        """
        self.threat_model = threat_model
        self.initialization = initialization
        self.optimization = optimization
        self.label_inference = label_inference
        self.epoch_handling_strategy = epoch_handling_strategy
        self.num_seeds_per_image = num_seeds_per_image

        # Validate components against threat model
        logger.info(f"Initializing ModularGIAOrchestrator with threat model: {threat_model.name}")
        self._validate_components()
        logger.info("✓ All components validated against threat model")

    def _validate_components(self) -> None:
        """Validate that all components are compatible with threat model."""
        components = [self.initialization, self.optimization]
        if self.label_inference:
            components.append(self.label_inference)

        for component in components:
            metadata = component.get_metadata()
            is_allowed, missing = self.threat_model.allows_component(metadata.required_capabilities)
            if not is_allowed:
                raise ValueError(
                    f"Component '{metadata.name}' requires capabilities "
                    f"{missing} but threat model only provides {self.threat_model.capabilities}"
                )

    def _validate_input_shape(
        self,
        input_shape: tuple[int, ...] | None,
        client_observations: ClientObservations,
    ) -> tuple[int, ...]:
        """Validate and extract input shape."""
        if input_shape is None:
            if client_observations.input_shape is None:
                raise ValueError(
                    "input_shape must be provided either as parameter or in client_observations.input_shape"
                )
            input_shape = client_observations.input_shape
            logger.info("Using input_shape from client_observations")
        return input_shape

    def _log_attack_start(self, input_shape: tuple[int, ...], device: torch.device) -> None:
        """Log attack initialization information."""
        logger.info("="*60)
        logger.info("Starting gradient inversion attack")
        logger.info(f"Input shape: {input_shape}, Device: {device}")
        logger.info(f"Threat model: {self.threat_model.name}")
        logger.info("="*60)

    def _log_client_observations(self, client_observations: ClientObservations) -> None:
        """Log what observations were provided by client."""
        logger.info("✓ Extracted observations from client")
        if client_observations.post_bn_stats is not None:
            logger.info(f"✓ Client provided post-training BN statistics ({len(client_observations.post_bn_stats)} layers)")
        if client_observations.pre_bn_stats is not None:
            logger.info(f"✓ Client provided pre-training BN statistics ({len(client_observations.pre_bn_stats)} layers)")
        if client_observations.num_images is not None:
            logger.info(f"✓ Client provided num_images: {client_observations.num_images}")

    def _infer_labels(
        self,
        target_model: nn.Module,
        client_observations: ClientObservations,
        num_samples: int,
    ) -> LabelInferenceResult | None:
        """Perform label inference or use provided labels."""
        logger.info("Stage 1/4: Label Inference")

        observed_gradients = client_observations.gradients
        labels = client_observations.labels

        if self.label_inference is not None:
            gradient_list = [observed_gradients[name] for name, _ in target_model.named_parameters()]
            is_update = (
                client_observations.training_settings is not None and
                client_observations.training_settings.compute_mode == "updates"
            )
            logger.debug(f"Using {self.label_inference.get_metadata().name} for label inference")
            label_result = self.label_inference.infer_labels(
                gradients=gradient_list,
                model=target_model,
                num_samples=num_samples,
                true_labels=labels,
                is_update=is_update,
            )
            logger.info(f"✓ Inferred labels: {label_result.labels.tolist()}")
            return label_result

        if labels is not None:
            label_result = LabelInferenceResult(labels=labels, method="provided")
            logger.info(f"✓ Using provided labels: {labels.tolist()}")
            return label_result

        logger.info("No labels available - optimization without label guidance")
        return None

    def _compute_initialization_shape(
        self,
        input_shape: tuple[int, ...],
        client_observations: ClientObservations,
        label_result: LabelInferenceResult | None,
    ) -> tuple[tuple[int, ...], LabelInferenceResult | None]:
        """Compute initialization shape based on epoch handling strategy."""
        epochs = 1
        num_images = input_shape[0]

        if client_observations.training_settings is not None:
            epochs = client_observations.training_settings.epochs
            logger.info(f"Using epochs={epochs} from training_settings")

        if self.epoch_handling_strategy is None:
            return input_shape, label_result

        logger.info(f"Using epoch handling strategy: {self.epoch_handling_strategy.get_name()}")
        channels, height, width = input_shape[1:]

        expected_shape = self.epoch_handling_strategy.get_expected_reconstruction_shape(
            num_images=num_images,
            num_epochs=epochs,
            num_seeds=self.num_seeds_per_image,
            input_shape=(channels, height, width),
        )

        # Reshape labels if needed
        if label_result is not None:
            label_result = self._reshape_labels_for_epochs(
                label_result, num_images, epochs
            )

        return expected_shape, label_result

    def _reshape_labels_for_epochs(
        self,
        label_result: LabelInferenceResult,
        num_images: int,
        epochs: int,
    ) -> LabelInferenceResult:
        """Reshape labels to match epoch handling strategy."""
        label_shape = self.epoch_handling_strategy.get_expected_label_shape(
            num_images=num_images,
            num_epochs=epochs,
        )

        if label_result.labels.ndim == 1 and len(label_shape) == 2:
            if label_shape[0] == epochs and label_shape[0] > 1:
                # MultiEpochSeparate: repeat labels for each epoch [N] -> [E, N]
                label_result.labels = label_result.labels.unsqueeze(0).expand(epochs, -1).contiguous()
            elif label_shape[0] == 1:
                # SingleStorageReused: reshape to [1, N]
                label_result.labels = label_result.labels.unsqueeze(0)

        return label_result

    def _create_attack_config(
        self,
        input_shape: tuple[int, ...],
        optimization_state: OptimizationState,
        label_result: LabelInferenceResult | None,
        labels: torch.Tensor | None,
    ) -> dict:
        """Create final attack configuration dictionary."""
        return {
            "threat_model": self.threat_model.name,
            "input_shape": input_shape,
            "converged": optimization_state.converged,
            "final_loss": optimization_state.loss,
            "iterations": optimization_state.iteration,
            "inferred_labels": label_result.labels.tolist() if label_result is not None else None,
            "true_labels": labels.tolist() if labels is not None else None,
        }

    def run_attack(
        self,
        target_model: nn.Module,
        client_observations: ClientObservations,
        device: torch.device,
        proxy_dataloader: torch.utils.data.DataLoader | None = None,
        input_shape: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Run complete gradient inversion attack server-side.

        Args:
            target_model: Server's model
            client_observations: Observations from client (gradients, BN stats, etc.)
            device: Device for computation
            proxy_dataloader: Optional dataloader for proxy data (if needed)
            input_shape: Shape of input to reconstruct (optional, extracted from client_observations if available)

        Returns:
            Reconstructed data tensor

        """
        # Validate and extract input shape
        input_shape = self._validate_input_shape(input_shape, client_observations)
        self._log_attack_start(input_shape, device)
        self._log_client_observations(client_observations)

        # Copy target model to ensure we don't modify it
        target_model = deepcopy(target_model).to(device)

        # Step 1: Label inference
        label_result = self._infer_labels(target_model, client_observations, input_shape[0])

        # Step 2: Initialization
        logger.info("Stage 2/4: Initialization")
        logger.debug(f"Using {self.initialization.get_metadata().name} for initialization")

        expected_shape, label_result = self._compute_initialization_shape(
            input_shape, client_observations, label_result
        )

        init_result = self.initialization.initialize(
            shape=expected_shape,
            device=device,
        )
        reconstruction = init_result.reconstruction
        logger.info(f"✓ Initialized reconstruction: shape={reconstruction.shape}")

        # Step 3: Optimization
        logger.info("Stage 3/4: Optimization")
        logger.info(f"Using {self.optimization.get_metadata().name} optimizer")
        optimization_state = self.optimization.optimize(
            reconstruction=reconstruction,
            labels=label_result,
            target_model=target_model,
            client_observations=client_observations,
            proxy_dataloader=proxy_dataloader,
        )
        reconstruction = optimization_state.reconstruction
        logger.info(
            f"✓ Optimization complete: "
            f"iterations={optimization_state.iteration}, "
            f"loss={optimization_state.loss:.6f}, "
            f"converged={optimization_state.converged}"
        )


        logger.info("="*60)
        logger.info("Attack complete - returning reconstruction to client")
        logger.info("="*60)

        config = self._create_attack_config(
            input_shape, optimization_state, label_result, client_observations.labels
        )
        return reconstruction, config

    def __repr__(self) -> str:
        """String representation of orchestrator."""
        components = [
            f"threat_model={self.threat_model.name}",
            f"initialization={self.initialization.get_metadata().name}",
            f"optimization={self.optimization.get_metadata().name}",
        ]
        if self.label_inference:
            components.append(f"label_inference={self.label_inference.get_metadata().name}")

        return f"ModularGIAOrchestrator({', '.join(components)})"


__all__ = ["ModularGIAOrchestrator"]
