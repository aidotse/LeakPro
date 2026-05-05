#
# Copyright 2023-2026 AI Sweden
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Loss components for gradient inversion optimization."""

from __future__ import annotations

from abc import abstractmethod
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader

from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.bn_statistics_strategies import (
    BNStatisticsStrategy,
)
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.training_simulator import (
    MultiEpochTrainingSimulation,
    TrainingSimulator,
)
from leakpro.attacks.gia_attacks.modular.core.component_base import (
    AggregationStrategy,
    Component,
    ComponentMetadata,
)
from leakpro.fl_utils.fl_client_simulator import ClientObservations


class LossComponent(Component):
    """Base class for all loss components."""

    def __init__(self, weight: float = 1.0) -> None:
        """Initialize loss component.

        Args:
            weight: Weight of this loss component in the total loss

        """
        self._weight = weight

    @property
    def weight(self) -> float:
        """Weight of this loss component."""
        return self._weight

    @abstractmethod
    def compute(
        self,
        reconstruction: torch.Tensor,
        model: nn.Module,
        labels: torch.Tensor,
        target_gradients: List[torch.Tensor],
        loss_fn: nn.Module,
    ) -> torch.Tensor:
        """Compute the loss.

        Args:
            reconstruction: Current reconstruction
            model: Target model
            labels: Labels for reconstruction
            target_gradients: True gradients to match
            loss_fn: Loss function for gradient computation

        Returns:
            Loss tensor (scalar)

        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this loss component."""
        pass

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Return metadata for this loss component.

        By default, loss components have no special requirements.
        Override this method if the loss component needs specific capabilities.
        """
        return ComponentMetadata(
            name=cls.__name__,
            display_name=cls.__name__,
            description="Loss component for gradient inversion",
            required_capabilities={},
        )


class GradientMatchingLoss(LossComponent):
    """Gradient matching loss - core loss for gradient inversion attacks.

    Supports matching either raw gradients or parameter updates depending on
    the training simulator used.
    """

    def __init__(
        self,
        loss_type: str = "l2",
        weight: float = 1.0,
        training_simulator: TrainingSimulator | None = None,
    ) -> None:
        """Initialize gradient matching loss.

        Args:
            loss_type: Distance metric ("l2", "cosine", or "fisher")
            weight: Loss weight
            training_simulator: TrainingSimulator instance for computing values.

        """
        super().__init__(weight)
        self.loss_type = loss_type
        self.training_simulator = training_simulator or MultiEpochTrainingSimulation(
            epochs=1, compute_mode="gradients", model_mode="eval"
        )

        if loss_type not in ["l2", "cosine", "fisher"]:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    @property
    def name(self) -> str:
        """Name of this loss component."""
        return f"GradientMatchingLoss({self.loss_type})"

    def compute(
        self,
        reconstruction: torch.Tensor,
        model: nn.Module,
        labels: torch.Tensor,
        target_gradients: List[torch.Tensor],
        loss_fn: nn.Module,
    ) -> torch.Tensor:
        """Compute gradient matching loss.

        Args:
            reconstruction: Current reconstruction from optimizer [E, B, G, C, H, W]
            model: Target model
            labels: Labels for reconstruction [B] or [B, K]
            target_gradients: True gradients/updates to match
            loss_fn: Loss function for gradient computation

        Returns:
            Loss tensor (scalar)

        """

        # Compute reconstructed values using training simulator
        reconstructed_values_dict = self.training_simulator.simulate_training(
            model=model,
            input_data=reconstruction,
            labels=labels,
            loss_fn=loss_fn,
        )

        # Check if multi-seed results
        if "seed_results" in reconstructed_values_dict:
            # Multi-seed: compute loss for each seed and average
            seed_gradients = reconstructed_values_dict["seed_results"]
            num_seeds = reconstructed_values_dict["num_seeds"]

            total_loss = 0.0
            for seed_grads in seed_gradients:
                reconstructed_values = list(seed_grads.values())
                seed_loss = self._compute_matching_loss(
                    reconstructed_values, target_gradients
                )
                total_loss += seed_loss

            # Average across seeds
            total_loss = total_loss / num_seeds
        else:
            # Single seed: compute loss directly
            reconstructed_values = list(reconstructed_values_dict.values())
            total_loss = self._compute_matching_loss(
                reconstructed_values, target_gradients
            )

        return total_loss * self.weight

    def _compute_matching_loss(
        self,
        reconstructed_values: List[torch.Tensor],
        target_gradients: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute matching loss between reconstructed and target gradients.

        Args:
            reconstructed_values: List of reconstructed gradient tensors
            target_gradients: List of target gradient tensors

        Returns:
            Matching loss (scalar)

        """
        # Match values using specified distance metric
        if self.loss_type == "l2":
            total_loss = sum(
                (g_rec - g_target).pow(2).sum()
                for g_rec, g_target in zip(reconstructed_values, target_gradients)
            )
        elif self.loss_type == "cosine":
            rec_flat = torch.cat([g.flatten() for g in reconstructed_values])
            target_flat = torch.cat([g.flatten() for g in target_gradients])
            cos_sim = torch.nn.functional.cosine_similarity(
                rec_flat.unsqueeze(0), target_flat.unsqueeze(0)
            )
            total_loss = 1.0 - cos_sim
        elif self.loss_type == "fisher":
            total_loss = 0.0
            for g_rec, g_target in zip(reconstructed_values, target_gradients):
                g_target = g_target.detach()
                fisher_weight = 1/ (torch.abs(g_target).pow(2) + 1e-8)
                total_loss +=  (fisher_weight * (g_rec - g_target).pow(2)).sum()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return total_loss


class TVRegularization(LossComponent):
    """Total Variation regularization for smooth reconstructions."""

    def __init__(self, weight: float = 1e-4) -> None:
        """Initialize TV regularization."""
        super().__init__(weight)

    @property
    def name(self) -> str:
        """Name of this loss component."""
        return "TVRegularization"

    def compute(
        self,
        reconstruction: torch.Tensor,
        model: nn.Module,
        labels: torch.Tensor,
        target_gradients: List[torch.Tensor],
        loss_fn: nn.Module,
    ) -> torch.Tensor:
        """Compute TV loss.

        Args:
            reconstruction: Current reconstruction [E, N, G, C, H, W]
            model: Not used
            labels: Not used
            target_gradients: Not used
            loss_fn: Not used

        Returns:
            TV regularization loss

        """
        _ = (model, labels, target_gradients, loss_fn)  # Unused parameters

        # Expect [E, N, G, C, H, W] format
        num_epochs, num_images, num_seeds = reconstruction.shape[:3]

        # Apply TV to each epoch and seed independently, then average
        total_tv = 0.0
        for e in range(num_epochs):
            for g in range(num_seeds):
                seed_rec = reconstruction[e, :, g, ...]  # [N, C, H, W]
                dx = torch.abs(seed_rec[:, :, :, :-1] - seed_rec[:, :, :, 1:])
                dy = torch.abs(seed_rec[:, :, :-1, :] - seed_rec[:, :, 1:, :])
                total_tv += dx.mean() + dy.mean()

        return self.weight * total_tv / (num_epochs * num_seeds)


class L2Regularization(LossComponent):
    """L2 regularization on reconstruction pixel values.

    Penalizes large pixel values by adding L2 norm to the loss:
        L_l2 = weight * ||reconstruction||²

    This encourages smaller, more constrained reconstructions and can
    help prevent extreme pixel values.

    Reference:
        Yin et al., "See through Gradients", CVPR 2021
    """

    def __init__(self, weight: float = 1e-4) -> None:
        """Initialize L2 regularization."""
        super().__init__(weight)

    @property
    def name(self) -> str:
        """Name of this loss component."""
        return "L2Regularization"

    def compute(
        self,
        reconstruction: torch.Tensor,
        model: nn.Module,
        labels: torch.Tensor,
        target_gradients: List[torch.Tensor],
        loss_fn: nn.Module,
    ) -> torch.Tensor:
        """Compute L2 regularization loss.

        Args:
            reconstruction: Current reconstruction [E, B, G, C, H, W]
            model: Not used
            labels: Not used
            target_gradients: Not used
            loss_fn: Not used

        Returns:
            L2 regularization loss

        """
        _ = (model, labels, target_gradients, loss_fn)  # Unused parameters

        # Compute squared L2 norm
        return self.weight * reconstruction.pow(2).mean()


class LabelEntropyRegularization(LossComponent):
    """Entropy regularization on label probabilities to promote confidence.

    This encourages the model to be confident in its label predictions by
    minimizing the entropy of the label distribution. Lower entropy means
    more peaked/confident distributions, higher entropy means more uniform
    (uncertain) distributions.

    Entropy: H(p) = -sum(p * log(p))
    - Low entropy (e.g., [0.99, 0.01]) → confident
    - High entropy (e.g., [0.5, 0.5]) → uncertain

    Note: This operates on the probabilities (after softmax), not the logits.
    """

    def __init__(self, weight: float = 1e-2) -> None:
        """Initialize label entropy regularization.

        Args:
            weight: Regularization weight. Higher values encourage more confident labels.

        """
        super().__init__(weight)

    @property
    def name(self) -> str:
        """Name of this loss component."""
        return "LabelEntropyRegularization"

    def compute(
        self,
        reconstruction: torch.Tensor,
        model: nn.Module,
        labels: torch.Tensor,
        target_gradients: List[torch.Tensor],
        loss_fn: nn.Module,
    ) -> torch.Tensor:
        """Compute entropy regularization loss on label probabilities.

        The labels tensor contains probabilities (after softmax from the label strategy).
        We compute the entropy H(p) = -sum(p * log(p)) and minimize it to encourage
        confident (peaked) label distributions.

        Args:
            reconstruction: Current reconstruction (not used)
            model: Target model (not used)
            labels: Label probabilities (batch_size, num_classes)
            target_gradients: True gradients (not used)
            loss_fn: Not used

        Returns:
            Entropy loss on labels (averaged across batch)

        """
        _ = (reconstruction, model, target_gradients, loss_fn)  # Unused parameters
        # Compute entropy: -sum(p * log(p))
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        entropy = -(labels * torch.log(labels + eps)).sum(dim=1)

        # Return mean entropy across batch
        return self.weight * entropy.mean()


class BNStatisticsRegularization(LossComponent):
    """Batch normalization statistics regularization.

    This loss component regularizes reconstructions to match batch statistics
    (mean and variance) from normalization layers. Supports three strategies:

    1. RunningBNStatisticsStrategy: Uses model's running statistics (Huang et al.)
    2. InferredBNStatisticsStrategy: Infers statistics from momentum updates
    3. ProxyBNStatisticsStrategy: Estimates from proxy/surrogate data

    The strategy must be configured and setup before optimization begins.
    """

    def __init__(
        self,
        strategy: BNStatisticsStrategy,
        weight: float = 1e-4,
    ) -> None:
        """Initialize BN statistics regularization.

        Args:
            strategy: BN statistics strategy to use (must be imported from bn_statistics_strategies)
            weight: Loss weight

        """
        super().__init__(weight)
        self.strategy = strategy
        self._is_setup = False

    @property
    def name(self) -> str:
        """Name of this loss component."""
        metadata = self.strategy.get_metadata()
        return f"BNStatisticsRegularization({metadata.name})"

    def setup(
        self,
        model: nn.Module,
        reconstruction: torch.Tensor | None = None,
        client_observations: "ClientObservations" | None = None,
        training_simulator: TrainingSimulator | None = None,
        proxy_dataloader: "DataLoader" | None = None,
    ) -> None:
        """Setup the BN statistics strategy.

        This should be called before optimization begins, typically in the
        orchestrator or optimizer initialization.

        Args:
            model: Target model
            reconstruction: Current reconstruction tensor
            client_observations: ClientObservations from FL client
            training_simulator: Training simulator for forward passes
            proxy_dataloader: Optional server-side dataloader for ProxyBNStatisticsStrategy

        """
        self.strategy.setup(
            model=model,
            reconstruction=reconstruction,
            client_observations=client_observations,
            training_simulator=training_simulator,
            proxy_dataloader=proxy_dataloader,
        )
        self._is_setup = True

    def compute(
        self,
        reconstruction: torch.Tensor,
        model: nn.Module,
        labels: torch.Tensor,
        target_gradients: List[torch.Tensor],
        loss_fn: nn.Module,
    ) -> torch.Tensor:
        """Compute BN statistics regularization loss.

        Args:
            reconstruction: Current reconstruction
            model: Target model
            labels: Labels (not used)
            target_gradients: Target gradients (not used)
            loss_fn: Not used

        Returns:
            BN statistics mismatch loss


        """
        _ = (labels, target_gradients, loss_fn)  # Unused parameters
        if not self._is_setup:
            raise RuntimeError(
                "BNStatisticsRegularization must be setup() before use. "
                "Call bn_loss.setup(model, **strategy_kwargs) before optimization."
            )

        reg_loss = self.strategy.compute_regularization(model, reconstruction)
        return self.weight * reg_loss

    def cleanup(self) -> None:
        """Clean up strategy resources (e.g., remove hooks)."""
        if self._is_setup:
            self.strategy.cleanup()
            self._is_setup = False

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Return metadata for BN statistics regularization.

        Note: This returns empty requirements. The actual requirements
        come from the strategy instance, which is checked at runtime.
        """
        return ComponentMetadata(
            name="BNStatisticsRegularization",
            display_name="BN Statistics Regularization",
            description="Regularizes reconstruction to match BN statistics",
            required_capabilities={},
        )

    def get_strategy_requirements(self) -> dict:
        """Get requirements from the BN statistics strategy."""
        metadata = self.strategy.get_metadata()
        return metadata.required_capabilities


class GroupConsistencyRegularization(LossComponent):
    """Group consistency regularization for multi-seed optimization.

    Encourages multiple seeds for the same image to converge towards a
    consensus, preventing excessive divergence while allowing exploration.

    For reconstruction with shape [B, G, C, H, W]:
    - Computes consensus image for each of B images across its G seeds
    - Penalizes each seed for deviating from the consensus

    Loss formula (from See Through Gradients paper):
        L_group = (1/G) * Σ_g ||x̂_g - E(x̂_group)||²

    Where E(x̂_group) is the consensus (typically mean) across seeds.

    Reference:
        Yin et al., "See through Gradients: Image Batch Recovery via
        GradInversion", CVPR 2021, Equation 11
    """

    def __init__(
        self,
        seed_aggregation: AggregationStrategy,
        weight: float = 0.01,
    ) -> None:
        """Initialize group consistency regularization.

        Args:
            seed_aggregation: Strategy for computing consensus across seeds
            weight: Regularization weight (α_group in paper)

        """
        super().__init__(weight)
        self.seed_aggregation = seed_aggregation

    @property
    def name(self) -> str:
        """Name of this loss component."""
        return "GroupConsistencyRegularization"

    def compute(
        self,
        reconstruction: torch.Tensor,
        model: nn.Module,
        labels: torch.Tensor,
        target_gradients: List[torch.Tensor],
        loss_fn: nn.Module,
    ) -> torch.Tensor:
        """Compute group consistency regularization loss.

        Args:
            reconstruction: Current reconstruction, shape [E, B, G, C, H, W]
            model: Not used
            labels: Not used
            target_gradients: Not used
            loss_fn: Not used

        Returns:
            Group consistency loss (scalar)

        """
        _ = (model, labels, target_gradients, loss_fn)  # Unused parameters

        # Expect [E, B, G, C, H, W] format
        num_epochs, num_images, num_seeds = reconstruction.shape[:3]

        if num_seeds == 1:
            # Single seed, no consistency needed
            return torch.tensor(0.0, device=reconstruction.device)

        # Compute consensus for each image across its seeds
        # consensus shape: [E, B, C, H, W]
        consensus = self.seed_aggregation.compute_consensus(reconstruction)

        # Compute deviation of each seed from its group consensus
        total_loss = torch.tensor(0.0, device=reconstruction.device)

        for g in range(num_seeds):
            seed_reconstruction = reconstruction[:, :, g, ...]  # [E, B, C, H, W]
            deviation = (seed_reconstruction - consensus).pow(2).mean()
            total_loss += deviation

        # Average across seeds
        total_loss = total_loss / num_seeds

        return self.weight * total_loss

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Return metadata for group consistency regularization."""
        return ComponentMetadata(
            name="GroupConsistencyRegularization",
            display_name="Group Consistency Regularization",
            description="Regularizes multiple seeds to stay near consensus",
            required_capabilities={},
            paper_reference="Yin et al., See Through Gradients, CVPR 2021",
        )


class EpochOrderInvariantPrior(LossComponent):
    """Epoch order-invariant prior for FedAvg multi-epoch attacks.

    Enforces that order-invariant functions over all images in different epochs
    should produce the same result, since each image appears exactly once per epoch.

    Implements Equation 3 from Dimitrov et al.:
    L_inv = (1/E²) Σ_{e1,e2} D_inv(g(X̃_{e1}), g(X̃_{e2}))

    Reference:
        Dimitrov et al., "Data Leakage in Federated Averaging", TMLR 2022
    """

    def __init__(
        self,
        order_invariant_function: str = "mean",
        distance_function: str = "l2",
        weight: float = 0.1,
        epochs: int = 3,
    ) -> None:
        """Initialize epoch order-invariant prior.

        Args:
            order_invariant_function: Function to aggregate images within an epoch
                - "mean": Mean of all images
                - "sum": Sum of all images
                - "variance": Variance across images
            distance_function: Distance to compare aggregated epoch representations
                - "l2": L2 distance
                - "l1": L1 distance
            weight: Loss weight (λ_inv in paper)
            epochs: Number of epochs (E)

        """
        super().__init__(weight)
        self.order_invariant_function = order_invariant_function
        self.distance_function = distance_function
        self.epochs = epochs

        if order_invariant_function not in ["mean", "sum", "variance"]:
            raise ValueError(f"Unknown order_invariant_function: {order_invariant_function}")

        if distance_function not in ["l2", "l1"]:
            raise ValueError(f"Unknown distance_function: {distance_function}")

    @property
    def name(self) -> str:
        """Name of this loss component."""
        return f"EpochOrderInvariantPrior(g={self.order_invariant_function})"

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Return metadata for this loss component."""
        return ComponentMetadata(
            name=cls.__name__,
            display_name="Epoch Order-Invariant Prior (FedAvg)",
            description="Enforce order-invariant properties across epochs",
            required_capabilities={},
            paper_reference="Dimitrov et al., Data Leakage in Federated Averaging, TMLR 2022",
        )

    def compute(
        self,
        reconstruction: torch.Tensor,
        model: nn.Module,
        labels: torch.Tensor,
        target_gradients: List[torch.Tensor],
        loss_fn: nn.Module,
    ) -> torch.Tensor:
        """Compute epoch order-invariant prior loss.

        Args:
            reconstruction: Reconstruction in [E, B, G, C, H, W] format
            model: Target model (unused)
            labels: Labels (unused)
            target_gradients: Target gradients (unused)
            loss_fn: Loss function (unused)

        Returns:
            Loss tensor (scalar)

        """
        _ = (model, labels, target_gradients, loss_fn)  # Unused

        num_epochs, num_images, num_seeds = reconstruction.shape[:3]

        # Validate dimensions
        if num_epochs != self.epochs:
            raise ValueError(f"Expected E={self.epochs} epochs, got {num_epochs}")

        # No prior applies for single epoch
        if num_epochs == 1:
            return torch.tensor(0.0, device=reconstruction.device, dtype=reconstruction.dtype)

        # Process each seed separately and average
        total_loss = sum(
            self._compute_single_seed_prior(reconstruction[:, :, g, ...])
            for g in range(num_seeds)
        ) / num_seeds

        return total_loss * self.weight

    def _compute_single_seed_prior(self, reconstruction: torch.Tensor) -> torch.Tensor:
        """Compute prior for a single seed.

        Args:
            reconstruction: Shape [E, B, C, H, W]

        Returns:
            Prior loss (scalar)

        """
        num_epochs = reconstruction.shape[0]

        # Compute order-invariant representation for each epoch
        epoch_representations = []
        for e in range(num_epochs):
            epoch_images = reconstruction[e]  # [B, C, H, W]

            # Apply order-invariant function
            if self.order_invariant_function == "mean":
                epoch_repr = epoch_images.mean(dim=0)
            elif self.order_invariant_function == "sum":
                epoch_repr = epoch_images.sum(dim=0)
            elif self.order_invariant_function == "variance":
                epoch_repr = epoch_images.var(dim=0)

            epoch_representations.append(epoch_repr)

        # Compute pairwise distances (upper triangle only)
        total_loss = 0.0
        for e1 in range(num_epochs):
            for e2 in range(e1 + 1, num_epochs):
                repr1, repr2 = epoch_representations[e1], epoch_representations[e2]
                dist = (repr1 - repr2).pow(2).sum() if self.distance_function == "l2" else (repr1 - repr2).abs().sum()
                total_loss += dist

        # Normalize by number of pairs: E*(E-1)/2
        return total_loss / (num_epochs * (num_epochs - 1) / 2) if num_epochs > 1 else total_loss


__all__ = [
    "LossComponent",
    "GradientMatchingLoss",
    "TVRegularization",
    "L2Regularization",
    "LabelEntropyRegularization",
    "BNStatisticsRegularization",
    "GroupConsistencyRegularization",
    "EpochOrderInvariantPrior",
]
