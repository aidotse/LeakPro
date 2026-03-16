"""Training simulator component for gradient inversion attacks.

This module provides different strategies for simulating client-side training
and computing gradients or parameter updates used for matching in the attack.
"""

from __future__ import annotations

from abc import abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from leakpro.attacks.gia_attacks.modular.core.component_base import Component, ComponentMetadata
from leakpro.fl_utils.gia_module_to_functional import MetaModule
from leakpro.fl_utils.gia_optimizers import MetaAdam, MetaSGD

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.epoch_strategies import EpochHandlingStrategy


@dataclass
class TrainingSettings:
    """Configuration for training simulation.

    This dataclass encapsulates all parameters needed to recreate a training simulator,
    allowing attacks to configure training simulation independently from the client.
    """

    epochs: int
    """Number of training epochs"""

    optimizer_type: str
    """Optimizer to use ('sgd' or 'adam')"""

    training_batch_size: int | None
    """Mini-batch size for gradient updates during training. If None, uses full batch."""

    compute_mode: str
    """'updates' to compute parameter updates (Δθ), 'gradients' to compute gradients (∂L/∂θ)"""

    model_mode: str
    """Mode to set the model to ('train' or 'eval')"""

    shuffle_mode: str = "attack"
    """Training mode: 'attack' (deterministic) or 'client' (realistic with shuffling)"""

    @classmethod
    def from_simulator(cls, simulator: "TrainingSimulator") -> "TrainingSettings":
        """Extract training settings from an existing TrainingSimulator instance.

        Args:
            simulator: TrainingSimulator instance to extract settings from

        Returns:
            TrainingSettings with the same configuration

        """
        if isinstance(simulator, MultiEpochTrainingSimulation):
            return cls(
                epochs=simulator.epochs,
                optimizer_type=simulator.optimizer_type,
                training_batch_size=simulator.batch_size,
                compute_mode=simulator.compute_mode,
                model_mode=simulator.model_mode,
                shuffle_mode=simulator.shuffle_mode,
            )
        raise ValueError(f"Unknown simulator type: {type(simulator)}")


class TrainingSimulator(Component):
    """Base class for training simulators.

    A training simulator computes either gradients or parameter updates
    from a model and input data, preserving the computational graph for
    backpropagation through the computation.
    """

    @abstractmethod
    def simulate_training(
        self,
        model: nn.Module,
        input_data: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: nn.Module,
    ) -> Dict[str, torch.Tensor] | Dict[str, list[torch.Tensor]]:
        """Compute gradients or updates from the model.

        Args:
            model: Target model
            input_data: Reconstructed input data. Shape can be:
                - [N, C, H, W] for client training
                - [E, N, G, C, H, W] for multi-seed optimization (G seeds per image)
            labels: Labels used for the loss (shape [N] or [N, num_classes])
            loss_fn: Loss function to use

        Returns:
            Dictionary mapping parameter names to gradient/update tensors.
            For multi-seed input, returns dict with 'seeds' key containing
            list of G gradient dicts (one per seed).

        """
        pass


class MultiEpochTrainingSimulation(TrainingSimulator):
    """Simulate multi-epoch client training and compute parameter updates.

    Uses meta-optimizers from gia_train.py to simulate FL client training
    while preserving the computational graph. Computes Δθ = θ_new - θ_old.
    """

    def __init__(
            self,
            epochs: int = 1,
            optimizer_type: str = "sgd",
            batch_size: int | None = None,
            compute_mode: str = "updates",
            model_mode: str = "train",
            shuffle_mode: str = "attack",
            epoch_handling_strategy: "EpochHandlingStrategy | None" = None,
        ) -> None:
        """Initialize training simulator.

        Args:
            epochs: Number of training epochs to simulate
            optimizer_type: Optimizer to use ("sgd" or "adam")
            batch_size: Batch size for training simulation. If None, uses full batch of N images.
            compute_mode: "updates" to compute parameter updates (Δθ),
                "gradients" to compute gradients (∂L/∂θ)
            model_mode: Mode to set the model to ("train" or "eval")
            shuffle_mode: Training mode:
                - "attack": Deterministic, no batch shuffling (for reconstruction)
                - "client": Realistic client simulation with batch shuffling
            epoch_handling_strategy: Strategy for how to handle reconstruction data across epochs.
                Controls whether we use separate images per epoch, repeat same images, etc.

        """
        self.epochs = epochs
        self.optimizer_type = optimizer_type
        self.batch_size = batch_size
        self.compute_mode = compute_mode
        self.model_mode = model_mode
        self.shuffle_mode = shuffle_mode
        self.epoch_handling_strategy = epoch_handling_strategy

        if shuffle_mode not in ["attack", "client"]:
            raise ValueError(f"Unknown shuffle mode: {shuffle_mode}. Must be 'attack' or 'client'")

        # Initialize meta-optimizer
        if optimizer_type == "sgd":
            self.meta_optimizer = MetaSGD(lr=0.01)
        elif optimizer_type == "adam":
            self.meta_optimizer = MetaAdam(lr=0.001)
        else:
            raise ValueError(f"Unknown optimizer_type: {optimizer_type}")

    def get_metadata(self) -> ComponentMetadata:
        """Get component metadata."""
        hyper_parameters_required = self.epochs > 1
        return ComponentMetadata(
            name="MultiEpochTrainingSimulation",
            display_name="Multi-Epoch Training Simulation",
            required_capabilities={"has_local_hyperparameters": hyper_parameters_required},
            description=f"Simulate {self.epochs}-epoch client training with {self.optimizer_type}",
        )

    def _validate_gradient_mode(self) -> None:
        """Validate that gradient mode is only used with single epoch."""
        if self.compute_mode == "gradients" and self.epochs != 1:
            raise ValueError("Multi-epoch gradient computation not supported")

    def simulate_training(
        self,
        model: nn.Module,
        input_data: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: nn.Module,
    ) -> Dict[str, torch.Tensor] | Dict[str, list[torch.Tensor]]:
        """Simulate training and compute parameter updates.

        Args:
            model: Target model
            input_data: Input data in one of two formats:
                - Attack reconstruction: [E, N, G, C, H, W]
                  * E: Number of epochs
                  * N: Images per epoch
                  * G: Number of random seeds (1 for single-seed)
                  * C, H, W: Image dimensions
                - Client data: [N, C, H, W] (real client training)
            labels: Labels:
                - Attack: [E, N] or [E, N, K]
                - Client: [N] or [N, K]
            loss_fn: Loss function to use

        Returns:
            Dictionary mapping parameter names to update tensors (Δθ)
            or gradient tensors (∂L/∂θ) depending on compute_mode.
            For multi-seed (G > 1), returns dict with 'seed_results' key containing list of dicts.

        """
        if self.model_mode == "eval":
            model.eval()
        else:
            model.train()
        # Detect mode from input shape
        is_client_data = (input_data.ndim == 4)

        if is_client_data:
            # Client real data: [N, C, H, W]
            return self._compute_for_client(model, input_data, labels, loss_fn)

        if input_data.ndim == 6:
            # Attack reconstruction: [E, N, G, C, H, W]
            return self._compute_for_attack(model, input_data, labels, loss_fn)

        raise ValueError(
            f"Unexpected input dimensionality: {input_data.ndim}D. "
            f"Expected 6D [E,N,G,C,H,W] for attack reconstruction or 4D [N,C,H,W] for client data"
        )

    def _compute_for_client(
        self,
        model: nn.Module,
        input_data: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """Compute gradients/updates for client training (real data).

        Args:
            model: Target model
            input_data: [N, C, H, W] real client data
            labels: [N] or [N, K] labels
            loss_fn: Loss function

        Returns:
            Dictionary mapping parameter names to gradients or updates

        """
        device = input_data.device

        # Gradient mode: compute gradients from single forward pass
        if self.compute_mode == "gradients":
            self._validate_gradient_mode()
            dataset = TensorDataset(input_data, labels)
            batch_size = self.batch_size or len(input_data)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(self.shuffle_mode == "client"))
            return self._compute_gradients(model, dataloader, loss_fn, device)

        # Update mode: simulate multi-epoch training with same data
        return self._compute_updates_for_epochs(
            model=model,
            input_data=input_data,  # [N, C, H, W]
            labels=labels,  # [N] or [N, K]
            loss_fn=loss_fn,
            device=device,
            use_epoch_strategy=False,  # Client reuses same data
        )

    def _compute_for_attack(
        self,
        model: nn.Module,
        input_data: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: nn.Module,
    ) -> Dict[str, torch.Tensor] | Dict[str, list[torch.Tensor]]:
        """Compute gradients/updates for attack reconstruction.

        Args:
            model: Target model
            input_data: [E, N, G, C, H, W] reconstruction
            labels: [E, N] or [E, N, K] labels
            loss_fn: Loss function

        Returns:
            For single seed (G=1): Dict mapping param names to gradients/updates
            For multi-seed (G>1): Dict with 'seed_results' and 'num_seeds' keys

        """
        input_epochs, num_images, num_seeds = input_data.shape[:3]

        # Process each seed independently
        seed_results = []
        for seed_idx in range(num_seeds):
            seed_data = input_data[:, :, seed_idx]  # [E, N, C, H, W]
            seed_result = self._compute_for_single_seed(
                model=model,
                input_data=seed_data,
                labels=labels,
                loss_fn=loss_fn,
            )
            seed_results.append(seed_result)

        # For single seed, return result directly
        if num_seeds == 1:
            return seed_results[0]

        # For multi-seed, return list of results
        return {"seed_results": seed_results, "num_seeds": num_seeds}

    def _compute_for_single_seed(
        self,
        model: nn.Module,
        input_data: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """Compute gradients/updates for a single seed of attack reconstruction.

        Args:
            model: Target model
            input_data: [E, N, C, H, W] single-seed reconstruction
            labels: [E, N] or [E, N, K] labels
            loss_fn: Loss function

        Returns:
            Dictionary mapping parameter names to gradients or updates

        """
        device = input_data.device

        # Gradient mode: use first epoch only
        if self.compute_mode == "gradients":
            self._validate_gradient_mode()
            epoch_data = self.epoch_handling_strategy.get_data_for_epoch(input_data, 0)  # [N, C, H, W]
            epoch_labels = self.epoch_handling_strategy.get_label_for_epoch(labels, 0)  # [N] or [N, K]

            dataset = TensorDataset(epoch_data, epoch_labels)
            batch_size = self.batch_size or len(epoch_data)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            return self._compute_gradients(model, dataloader, loss_fn, device)

        # Update mode: use epoch strategy for each epoch
        return self._compute_updates_for_epochs(
            model=model,
            input_data=input_data,  # [E, N, C, H, W]
            labels=labels,  # [E, N] or [E, N, K]
            loss_fn=loss_fn,
            device=device,
            use_epoch_strategy=True,  # Attack uses strategy
        )

    def _compute_updates_for_epochs(
        self,
        model: nn.Module,
        input_data: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: nn.Module,
        device: torch.device,
        use_epoch_strategy: bool,
    ) -> Dict[str, torch.Tensor]:
        """Compute updates by iterating through epochs.

        Args:
            model: Target model
            input_data:
                - If use_epoch_strategy=True: [E, N, C, H, W] attack reconstruction
                - If use_epoch_strategy=False: [N, C, H, W] client data
            labels:
                - If use_epoch_strategy=True: [E, N] or [E, N, K]
                - If use_epoch_strategy=False: [N] or [N, K]
            loss_fn: Loss function
            device: Device
            use_epoch_strategy: True for attack (use strategy), False for client (reuse same data)

        Returns:
            Dictionary mapping parameter names to update tensors (Δθ)

        """
        # Convert model to functional form
        patched_model = MetaModule(model)

        # Store original parameters
        original_params = OrderedDict(
            (name, param.clone()) for name, param in model.named_parameters()
        )

        # Iterate through epochs
        for epoch_idx in range(self.epochs):
            # Get data for this epoch
            if use_epoch_strategy:
                # Attack: ask strategy for epoch-specific data
                epoch_data = self.epoch_handling_strategy.get_data_for_epoch(input_data, epoch_idx)  # [N, C, H, W]
                epoch_labels = self.epoch_handling_strategy.get_label_for_epoch(labels, epoch_idx)  # [N] or [N, K]
            else:
                # Client: reuse same data for all epochs
                epoch_data = input_data  # [N, C, H, W]
                epoch_labels = labels  # [N] or [N, K]

            # Create dataloader for this epoch
            batch_size = self.batch_size or len(epoch_data)
            dataset = TensorDataset(epoch_data, epoch_labels)

            # Only shuffle in client mode (attack mode is always deterministic)
            # Attack always uses use_epoch_strategy=True, client uses False
            shuffle_batches = (self.shuffle_mode == "client" and not use_epoch_strategy)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_batches)

            # Train on batches for this epoch
            for batch_data, batch_labels in dataloader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)

                # Forward pass with current parameters
                outputs = patched_model(batch_data, patched_model.parameters)
                loss = loss_fn(outputs, batch_labels)

                # Meta-optimizer step (creates new parameter set)
                patched_model.parameters = self.meta_optimizer.step(
                    loss, patched_model.parameters
                )

        # Compute parameter updates (delta)
        updates = OrderedDict()
        for (name, new_param), (_, orig_param) in zip(
            patched_model.parameters.items(), original_params.items()
        ):
            updates[name] = new_param - orig_param

        return updates

    def _compute_gradients(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Compute gradients.

        Returns ∂L/∂θ accumulated across all batches (or from single batch).
        """
        accumulated_loss = 0.0
        num_batches = 0

        for inputs, batch_labels in dataloader:
            inputs = inputs.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(inputs)
            batch_loss = loss_fn(outputs, batch_labels)
            accumulated_loss = accumulated_loss + batch_loss
            num_batches += 1

        # Average loss across batches
        final_loss = accumulated_loss / num_batches

        # Compute gradients from final loss
        gradients = torch.autograd.grad(
            final_loss,
            model.parameters(),
            create_graph=True,
            retain_graph=True,
        )

        # Convert to named dictionary
        param_names = [name for name, _ in model.named_parameters()]
        return OrderedDict(zip(param_names, gradients))
