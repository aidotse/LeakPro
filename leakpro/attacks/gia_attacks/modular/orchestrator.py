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

import torch
from torch import nn

from leakpro.attacks.gia_attacks.modular.core.component_base import (
    InitializationStrategy,
    LabelInferenceStrategy,
    OptimizationStrategy,
)
from leakpro.attacks.gia_attacks.modular.core.threat_model import ThreatModel
from leakpro.fl_utils.fl_client_simulator import ClientObservations

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
    ) -> None:
        """Initialize orchestrator.

        Args:
            threat_model: Threat model defining attacker capabilities
            initialization: Initialization strategy
            optimization: Optimization strategy
            label_inference: Optional label inference strategy

        """
        self.threat_model = threat_model
        self.initialization = initialization
        self.optimization = optimization
        self.label_inference = label_inference

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

    def run_attack(
        self,
        target_model: nn.Module,
        client_observations: ClientObservations,
        input_shape: tuple,
        device: torch.device,
        proxy_dataloader: torch.utils.data.DataLoader | None = None,
    ) -> torch.Tensor:
        """Run complete gradient inversion attack server-side.

        Args:
            target_model: Server's model
            client_observations: Observations from client (gradients, BN stats, etc.)
            input_shape: Shape of input to reconstruct
            device: Device for computation
            proxy_dataloader: Optional dataloader for proxy data (if needed)

        Returns:
            Reconstructed data tensor

        """
        logger.info("="*60)
        logger.info("Starting gradient inversion attack")
        logger.info(f"Input shape: {input_shape}, Device: {device}")
        logger.info(f"Threat model: {self.threat_model.name}")
        logger.info("="*60)

        # Extract necessary observations from client_observations
        observed_gradients = client_observations.gradients
        labels = client_observations.labels
        logger.info("✓ Extracted observations from client")

        # Copy target model to ensure we don't modify it
        target_model = deepcopy(target_model).to(device)

        # Log what BN statistics were provided by client
        if client_observations.post_bn_stats is not None:
            logger.info(f"✓ Client provided post-training BN statistics ({len(client_observations.post_bn_stats)} layers)")
        if client_observations.pre_bn_stats is not None:
            logger.info(f"✓ Client provided pre-training BN statistics ({len(client_observations.pre_bn_stats)} layers)")
        if client_observations.batch_size is not None:
            logger.info(f"✓ Client provided batch size: {client_observations.batch_size}")
        if client_observations.spatial_sizes is not None:
            logger.info("✓ Client provided spatial sizes for variance correction")

        # Step 1: Label inference
        logger.info("Stage 1/4: Label Inference")
        inferred_labels = None
        if self.label_inference is not None:
            # Convert dict of gradients to list matching model parameters order
            gradient_list = [observed_gradients[name] for name, _ in target_model.named_parameters()]
            logger.debug(f"Using {self.label_inference.get_metadata().name} for label inference")
            label_result = self.label_inference.infer_labels(
                gradients=gradient_list,
                model=target_model,
                num_samples=input_shape[0],
                true_labels=labels,  # Pass through for OracleLabels
            )
            inferred_labels = label_result.labels
            logger.info(f"✓ Inferred labels: {inferred_labels.tolist()}")
        elif labels is not None:
            inferred_labels = labels
            logger.info(f"✓ Using provided labels: {labels.tolist()}")
        else:
            logger.info("No labels available - optimization without label guidance")

        # Step 2: Initialization
        logger.info("Stage 2/4: Initialization")
        logger.debug(f"Using {self.initialization.get_metadata().name} for initialization")
        init_result = self.initialization.initialize(
            shape=input_shape,
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

        config = {
                "threat_model": self.threat_model.name,
                "input_shape": input_shape,
                "converged": optimization_state.converged,
                "final_loss": optimization_state.loss,
                "iterations": optimization_state.iteration,
                "inferred_labels": inferred_labels.tolist() if inferred_labels is not None else None,
                "true_labels": labels.tolist() if labels is not None else None,
            }
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
