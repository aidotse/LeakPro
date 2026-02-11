"""Training simulator component for gradient inversion attacks.

This module provides different strategies for simulating client-side training
and computing gradients or parameter updates used for matching in the attack.
"""

from __future__ import annotations

from abc import abstractmethod
from collections import OrderedDict
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from leakpro.attacks.gia_attacks.modular.core.component_base import Component, ComponentMetadata
from leakpro.fl_utils.gia_module_to_functional import MetaModule
from leakpro.fl_utils.gia_optimizers import MetaAdam, MetaSGD


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
                - [B, C, H, W] for single-seed optimization
                - [B, G, C, H, W] for multi-seed optimization (G seeds per image)
            labels: Labels used for the loss (shape [B] or [B, num_classes])
            loss_fn: Loss function to use

        Returns:
            Dictionary mapping parameter names to gradient/update tensors.
            For multi-seed input, returns dict with 'seeds' key containing
            list of G gradient dicts (one per seed).

        """
        pass


class DirectGradientComputation(TrainingSimulator):
    """Compute raw gradients from single forward/backward pass.

    This is the standard unrealistic approach used by most attacks.
    Simply computes ∂L/∂θ without any training simulation.
    """

    def __init__(self, model_mode: str = "eval") -> None:
        """Initialize direct gradient computation."""
        self.model_mode = model_mode

    def get_metadata(self) -> ComponentMetadata:
        """Get component metadata."""
        return ComponentMetadata(
            name="DirectGradientComputation",
            display_name="Direct Gradient Computation",
            required_capabilities={},
            description="Compute raw gradients from single forward/backward pass",
        )

    def simulate_training(
        self,
        model: nn.Module,
        input_data: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: nn.Module,
    ) -> Dict[str, torch.Tensor] | Dict[str, list[torch.Tensor]]:
        """Compute gradients from single forward/backward pass.

        Args:
            model: Target model
            input_data: Reconstructed input data. Shape can be:
                - [B, C, H, W] for single-seed optimization
                - [B, G, C, H, W] for multi-seed optimization
            labels: Labels used for the loss (shape [B] or [B, num_classes])
            loss_fn: Loss function to use

        Returns:
            Dictionary mapping parameter names to gradient tensors.
            For multi-seed, returns dict with 'seeds' key containing list of dicts.

        """
        # Check if multi-seed input
        if input_data.ndim == 5:
            # Multi-seed: [B, G, C, H, W]
            batch_size, num_seeds = input_data.shape[:2]

            # Process each seed separately
            seed_gradients = []
            for g in range(num_seeds):
                seed_data = input_data[:, g, ...]  # [B, C, H, W]
                seed_grads = self._compute_single_seed(model, seed_data, labels, loss_fn)
                seed_gradients.append(seed_grads)

            return {"seeds": seed_gradients, "num_seeds": num_seeds}

        # Single seed: [B, C, H, W]
        return self._compute_single_seed(model, input_data, labels, loss_fn)

    def _compute_single_seed(
        self,
        model: nn.Module,
        input_data: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """Compute gradients for a single seed.

        Args:
            model: Target model
            input_data: Shape [B, C, H, W]
            labels: Labels (shape [B] or [B, num_classes])
            loss_fn: Loss function

        Returns:
            Dictionary mapping parameter names to gradient tensors

        """
        if self.model_mode == "eval":
            model.eval()
        else:
            model.train()

        # Forward pass
        outputs = model(input_data)
        loss = loss_fn(outputs, labels)

        # Compute gradients with graph
        gradients = torch.autograd.grad(
            loss,
            model.parameters(),
            create_graph=True,
            retain_graph=True,
        )

        # Convert to named dictionary
        param_names = [name for name, _ in model.named_parameters()]
        return OrderedDict(zip(param_names, gradients))


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
            model_mode: str = "train"
        ) -> None:
        """Initialize training simulator. No config.

        Args:
            epochs: Number of training epochs to simulate
            optimizer_type: Optimizer to use ("sgd" or "adam")
            batch_size: Batch size for training simulation. If None, uses full batch.
            compute_mode: "updates" to compute parameter updates (Δθ),
                "gradients" to compute gradients (∂L/∂θ)
            model_mode: Mode to set the model to ("train" or "eval")

        """
        self.epochs = epochs
        self.optimizer_type = optimizer_type
        self.batch_size = batch_size
        self.compute_mode = compute_mode
        self.model_mode = model_mode
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
            input_data: Reconstructed input data. Shape can be:
                - [B, C, H, W] for single-seed optimization
                - [B, G, C, H, W] for multi-seed optimization
            labels: Labels used for the loss (shape [B] or [B, num_classes])
            loss_fn: Loss function to use

        Returns:
            Dictionary mapping parameter names to update tensors (Δθ)
            or gradient tensors (∂L/∂θ) depending on compute_mode.
            For multi-seed, returns dict with 'seeds' key containing list of dicts.

        """
        # Check if multi-seed input
        if input_data.ndim == 5:
            # Multi-seed: [B, G, C, H, W]
            batch_size, num_seeds = input_data.shape[:2]

            # Process each seed separately
            seed_results = []
            for g in range(num_seeds):
                seed_data = input_data[:, g, ...]  # [B, C, H, W]
                seed_result = self._simulate_single_seed(model, seed_data, labels, loss_fn)
                seed_results.append(seed_result)

            return {"seeds": seed_results, "num_seeds": num_seeds}

        # Single seed: [B, C, H, W]
        return self._simulate_single_seed(model, input_data, labels, loss_fn)

    def _simulate_single_seed(
        self,
        model: nn.Module,
        input_data: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """Simulate training for a single seed.

        Args:
            model: Target model
            input_data: Shape [B, C, H, W]
            labels: Labels (shape [B] or [B, num_classes])
            loss_fn: Loss function

        Returns:
            Dictionary mapping parameter names to update/gradient tensors

        """
        if self.model_mode == "eval":
            model.eval()
        else:
            model.train()
        device = input_data.device

        # Create dataloader for training simulation
        dataset = TensorDataset(input_data, labels)
        batch_size = self.batch_size or len(input_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        if self.compute_mode == "updates":
            # Simulate training and return parameter updates
            return self._compute_updates(model, dataloader, loss_fn, device)

        # Compute gradients (with multi-epoch forward passes if epochs > 1)
        return self._compute_gradients(model, dataloader, loss_fn, device)

    def _compute_updates(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Compute parameter updates after training simulation.

        Returns Δθ = θ_new - θ_old
        """
        # Convert model to functional form
        patched_model = MetaModule(model)

        # Store original parameters
        original_params = OrderedDict(
            (name, param.clone()) for name, param in model.named_parameters()
        )

        # Simulate training epochs
        for _ in range(self.epochs):
            for inputs, batch_labels in dataloader:
                inputs = inputs.to(device)
                batch_labels = batch_labels.to(device)

                # Forward pass with current parameters
                outputs = patched_model(inputs, patched_model.parameters)
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

        Returns ∂L/∂θ evaluated at the final loss.
        """
        if self.epochs != 1:
            raise ValueError("Multi-epoch gradient computation is not possible.")

        for inputs, batch_labels in dataloader:
            inputs = inputs.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(inputs)
            final_loss = loss_fn(outputs, batch_labels)

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
