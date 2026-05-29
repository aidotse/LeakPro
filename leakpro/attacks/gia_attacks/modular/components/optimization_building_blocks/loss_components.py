#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Loss components for gradient inversion optimization."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, List

import torch

from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.bn_statistics_strategies import (
    BNStatisticsStrategy,
)
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.training_simulator import (
    MultiEpochTrainingSimulation,
    TrainingSimulator,
)
from pydantic import validate_call

from leakpro.attacks.gia_attacks.modular.config.registry import register
from leakpro.attacks.gia_attacks.modular.core.component_base import (
    AggregationStrategy,
    Component,
    ComponentMetadata,
)

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.state import RunContext


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

    def setup(
        self,
        ctx: "RunContext",
        reconstruction: torch.Tensor | None = None,
    ) -> None:
        """One-time setup called before the optimisation loop begins.

        The default implementation is a no-op.  Override in subclasses that
        need to extract model statistics, install hooks, or perform any other
        per-stage initialisation before :meth:`compute` is ever called.

        Args:
            ctx: Immutable run context (model, observations, simulator, dataloader).
            reconstruction: Initial reconstruction tensor — useful when
                setup depends on the starting point (e.g. batch-size inference).
        """

    @abstractmethod
    def compute(
        self,
        reconstruction: torch.Tensor,
        labels: torch.Tensor,
        target_gradients: List[torch.Tensor],
        ctx: "RunContext",
    ) -> torch.Tensor:
        """Compute the loss.

        Args:
            reconstruction: Current reconstruction in data space
            labels: Labels for this reconstruction
            target_gradients: Pre-extracted ordered target gradient list
            ctx: Run context (provides target_model and loss_fn)

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
            required_capabilities={},
        )


@register("loss.gradient_matching")
class GradientMatchingLoss(LossComponent):
    """Gradient matching loss - core loss for gradient inversion attacks.

    Supports matching either raw gradients or parameter updates depending on
    the training simulator used.
    """

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        loss_type: str = "l2",
        weight: float = 1.0,
        training_simulator: TrainingSimulator | None = None,
    ) -> None:
        """Initialize gradient matching loss.

        Args:
            loss_type: Distance metric. Options:
                - "l2": Euclidean distance (MSE)
                - "cosine": 1 - cosine_similarity (PyTorch implementation)
                - "sim": Cosine similarity (original GIFD manual implementation)
                - "sim_cmpr0": Compressed cosine similarity with 0% pruning (GIFD paper)
                - "fisher": Fisher-weighted L2 distance
            weight: Loss weight
            training_simulator: TrainingSimulator instance for computing values.

        """
        super().__init__(weight)
        self.loss_type = loss_type
        self.training_simulator = training_simulator or MultiEpochTrainingSimulation(
            epochs=1, compute_mode="gradients", model_mode="eval"
        )

        if loss_type not in ["l2", "cosine", "fisher", "sim", "sim_cmpr0"]:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    @property
    def name(self) -> str:
        """Name of this loss component."""
        return f"GradientMatchingLoss({self.loss_type})"

    def compute(
        self,
        reconstruction: torch.Tensor,
        labels: torch.Tensor,
        target_gradients: List[torch.Tensor],
        ctx: "RunContext",
    ) -> torch.Tensor:
        """Compute gradient matching loss.

        Args:
            reconstruction: Current reconstruction from optimizer [E, B, G, C, H, W]
            labels: Labels for reconstruction [B] or [B, K]
            target_gradients: Pre-extracted ordered target gradient list
            ctx: Run context providing target_model and loss_fn

        Returns:
            Loss tensor (scalar)

        """

        reconstructed_values_dict = self.training_simulator.simulate_training(
            model=ctx.target_model,
            input_data=reconstruction,
            labels=labels,
            loss_fn=ctx.loss_fn,
        )

        if "seed_results" in reconstructed_values_dict:
            seed_gradients = reconstructed_values_dict["seed_results"]
            num_seeds = reconstructed_values_dict["num_seeds"]

            total_loss = 0.0
            for seed_grads in seed_gradients:
                reconstructed_values = list(seed_grads.values())
                seed_loss = self._compute_matching_loss(
                    reconstructed_values, target_gradients
                )
                total_loss += seed_loss

            total_loss = total_loss / num_seeds
        else:
            reconstructed_values = list(reconstructed_values_dict.values())
            total_loss = self._compute_matching_loss(
                reconstructed_values, target_gradients
            )

        return total_loss * self.weight

    def compute_per_seed_losses(
        self,
        reconstruction: torch.Tensor,
        labels: torch.Tensor,
        target_gradients: List[torch.Tensor],
        ctx: "RunContext",
    ) -> torch.Tensor:
        """Compute gradient matching loss separately for each seed.

        Args:
            reconstruction: Current reconstruction [E, N, G, C, H, W]
            labels: Labels for reconstruction
            target_gradients: Pre-extracted ordered target gradient list
            ctx: Run context providing target_model and loss_fn

        Returns:
            Per-seed losses [E, N, G]

        """
        E, N, G = reconstruction.shape[:3]

        reconstructed_values_dict = self.training_simulator.simulate_training(
            model=ctx.target_model,
            input_data=reconstruction,
            labels=labels,
            loss_fn=ctx.loss_fn,
        )

        if "seed_results" in reconstructed_values_dict:
            seed_gradients = reconstructed_values_dict["seed_results"]
            num_seeds = len(seed_gradients)

            if num_seeds != G:
                raise ValueError(
                    f"Mismatch between reconstruction seeds (G={G}) and "
                    f"training simulator results (num_seeds={num_seeds})"
                )

            seed_losses_list = []
            for seed_grads in seed_gradients:
                reconstructed_values = list(seed_grads.values())
                seed_loss = self._compute_matching_loss(
                    reconstructed_values, target_gradients
                )
                seed_losses_list.append(seed_loss)

            seed_losses = torch.stack(seed_losses_list)
            seed_losses = seed_losses.view(1, 1, G).expand(E, N, G).contiguous()
        else:
            reconstructed_values = list(reconstructed_values_dict.values())
            loss = self._compute_matching_loss(reconstructed_values, target_gradients)
            seed_losses = loss.view(1, 1, 1).expand(E, N, G).contiguous()

        return seed_losses

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
        elif self.loss_type == "sim":
            # Original GIFD 'sim' loss: manual cosine similarity computation
            # Equivalent to cosine but matches original implementation exactly
            pnorm = [torch.tensor(0.0, device=reconstructed_values[0].device),
                     torch.tensor(0.0, device=target_gradients[0].device)]
            costs = torch.tensor(0.0, device=reconstructed_values[0].device)

            for g_rec, g_target in zip(reconstructed_values, target_gradients):
                costs -= (g_rec * g_target).sum()
                pnorm[0] += g_rec.pow(2).sum()
                pnorm[1] += g_target.pow(2).sum()

            total_loss = 1.0 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()
        elif self.loss_type == "sim_cmpr0":
            # Original GIFD 'sim_cmpr0' loss: compressed similarity with 0% pruning
            # This is equivalent to 'sim' but uses the compressed gradient implementation
            # With ratio=0.0, all gradients are kept (no compression)
            ratio = 0.0
            pnorm = [torch.tensor(0.0, device=reconstructed_values[0].device),
                     torch.tensor(0.0, device=target_gradients[0].device)]
            costs = torch.tensor(0.0, device=reconstructed_values[0].device)

            for g_rec, g_target in zip(reconstructed_values, target_gradients):
                # Compute number of elements to keep
                k = int(g_target.numel() * (1 - ratio))
                k = max(k, 1)

                # Flatten and select top-k by absolute value from target
                target_flat = g_target.flatten()
                target_threshold = torch.min(
                    torch.topk(torch.abs(target_flat), k, 0, largest=True, sorted=False)[0]
                )
                target_mask = torch.ge(torch.abs(target_flat), target_threshold)
                target_compressed = target_flat * target_mask

                # Apply same mask to reconstructed gradients
                rec_flat = g_rec.flatten()
                rec_compressed = rec_flat * target_mask

                # Compute cosine similarity on compressed gradients
                costs -= (rec_compressed * target_compressed).sum()
                pnorm[0] += rec_compressed.pow(2).sum()
                pnorm[1] += target_compressed.pow(2).sum()

            total_loss = 1.0 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()
        elif self.loss_type == "fisher":
            total_loss = 0.0
            for g_rec, g_target in zip(reconstructed_values, target_gradients):
                g_target = g_target.detach()
                fisher_weight = 1/ (torch.abs(g_target).pow(2) + 1e-8)
                total_loss +=  (fisher_weight * (g_rec - g_target).pow(2)).sum()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return total_loss


@register("loss.tv")
class TVRegularization(LossComponent):
    """Total Variation regularization for smooth reconstructions."""

    @validate_call
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
        labels: torch.Tensor,
        target_gradients: List[torch.Tensor],
        ctx: "RunContext",
    ) -> torch.Tensor:
        """Compute TV loss on reconstruction [E, N, G, C, H, W]."""
        _ = (labels, target_gradients, ctx)  # Unused

        num_epochs, num_images, num_seeds = reconstruction.shape[:3]

        total_tv = 0.0
        for e in range(num_epochs):
            for g in range(num_seeds):
                seed_rec = reconstruction[e, :, g, ...]  # [N, C, H, W]
                dx = torch.abs(seed_rec[:, :, :, :-1] - seed_rec[:, :, :, 1:])
                dy = torch.abs(seed_rec[:, :, :-1, :] - seed_rec[:, :, 1:, :])
                tv_value = dx.mean() + dy.mean()

                total_tv += tv_value

        return self.weight * total_tv / (num_epochs * num_seeds)


@register("loss.l2")
class L2Regularization(LossComponent):
    """L2 regularization on reconstruction pixel values.

    Penalizes large pixel values by adding L2 norm to the loss.
    Computes Frobenius norm normalized by spatial resolution:
        L_l2 = weight * ||reconstruction||_2 / (H * W)

    Where ||·||_2 is the L2/Frobenius norm: sqrt(sum(x²))

    This matches the paper's image_norm regularization which encourages
    smaller, more constrained reconstructions.

    Reference:
        Yin et al., "See through Gradients", CVPR 2021
        Fang et al., "GIFD", ICCV 2023 (configs: image_norm)
    """

    @validate_call
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
        labels: torch.Tensor,
        target_gradients: List[torch.Tensor],
        ctx: "RunContext",
    ) -> torch.Tensor:
        """Compute L2 regularization loss on reconstruction [E, B, G, C, H, W]."""
        _ = (labels, target_gradients, ctx)  # Unused

        H, W = reconstruction.shape[-2:]
        spatial_area = H * W

        E, N, G, C, H, W = reconstruction.shape
        per_seed = reconstruction.reshape(E * N * G, -1)
        l2_norms = torch.norm(per_seed, p=2, dim=1)
        total_norm = l2_norms.sum() / spatial_area

        return self.weight * total_norm


@register("loss.latent_kld")
class LatentKLDivergenceRegularization(LossComponent):
    """KL-divergence prior that keeps the latent code close to N(0, 1).

    Used in the GIFD paper (Fang et al., ICCV 2023) during the stage-0
    latent optimisation with BigGAN (``KLD=0.1``).  It is only applied in
    stage 0 and disabled for all subsequent feature-domain stages.

    The formula matches the paper's closure exactly::

        KLD = -0.5 * (1 + log(σ² + ε) - μ² - σ²)

    where μ and σ are the **sample** mean and std of the full (flattened)
    latent vector, treating all ``latent_dim`` coordinates as i.i.d. draws.
    This is the KL divergence from a fitted 1-D Gaussian to ``N(0,1)``.

    The component reads the latent code via a mutable reference that must be
    set before each optimisation step by calling :meth:`set_latent`.  The
    :class:`~leakpro.attacks.gia_attacks.modular.components.composable_optimizer
    .ComposableOptimizer` does this automatically when the component is included
    in the stage's loss list.

    Args:
        weight: Scaling factor applied to the KLD value (paper default: 0.1).

    """

    @validate_call
    def __init__(self, weight: float = 0.1) -> None:
        """Initialise KLD prior."""
        super().__init__(weight)
        self._latent_ref: torch.Tensor | None = None

    def set_latent(self, latent: torch.Tensor) -> None:
        """Store a reference to the current optimisable latent tensor.

        Called by :class:`ComposableOptimizer` immediately before loss
        computation so that ``compute`` sees the up-to-date ``z``.

        Args:
            latent: Optimisable latent codes ``[E, N, G, latent_dim]`` or
                    any shape; the tensor is used as-is (no copy made).

        """
        self._latent_ref = latent

    @property
    def name(self) -> str:
        """Name of this loss component."""
        return "LatentKLDivergenceRegularization"

    def compute(
        self,
        reconstruction: torch.Tensor,
        labels: torch.Tensor,
        target_gradients: List[torch.Tensor],
        ctx: "RunContext",
    ) -> torch.Tensor:
        """Compute KLD regularisation on the stored latent reference.

        The latent reference must be set before each step via :meth:`set_latent`.
        ``reconstruction``, ``labels``, ``target_gradients``, and ``ctx`` are unused.

        Returns:
            ``weight × KLD`` scalar tensor on the same device as the latent.

        """
        _ = (reconstruction, labels, target_gradients, ctx)

        if self._latent_ref is None:
            raise RuntimeError(
                "Latent reference not set. Call set_latent() before compute()."
            )

        z = self._latent_ref.flatten()

        # unbiased=False matches the paper's torch.std call
        mu = torch.mean(z)
        sigma = torch.std(z, unbiased=False)
        sigma2 = sigma.pow(2)

        # -0.5 * (1 + log(σ² + ε) − μ² − σ²)
        kld = -0.5 * (1.0 + torch.log(sigma2 + 1e-10) - mu.pow(2) - sigma2)
        return self.weight * kld


@register("loss.label_entropy")
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

    @validate_call
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
        labels: torch.Tensor,
        target_gradients: List[torch.Tensor],
        ctx: "RunContext",
    ) -> torch.Tensor:
        """Compute entropy regularization loss on label probabilities.

        Args:
            reconstruction: Current reconstruction (not used)
            labels: Label probabilities (batch_size, num_classes)
            target_gradients: True gradients (not used)
            ctx: Run context (not used)

        Returns:
            Entropy loss on labels (averaged across batch)

        """
        _ = (reconstruction, target_gradients, ctx)
        eps = 1e-8
        entropy = -(labels * torch.log(labels + eps)).sum(dim=1)
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
        ctx: "RunContext",
        reconstruction: torch.Tensor | None = None,
    ) -> None:
        """Setup the BN statistics strategy.

        Args:
            ctx: Run context providing model, observations, training simulator, and dataloader
            reconstruction: Current reconstruction tensor

        """
        self.strategy.setup(ctx=ctx, reconstruction=reconstruction)
        self._is_setup = True

    def compute(
        self,
        reconstruction: torch.Tensor,
        labels: torch.Tensor,
        target_gradients: List[torch.Tensor],
        ctx: "RunContext",
    ) -> torch.Tensor:
        """Compute BN statistics regularization loss.

        Args:
            reconstruction: Current reconstruction
            labels: Labels (not used)
            target_gradients: Target gradients (not used)
            ctx: Run context (not directly used; required by LossComponent ABC).

        Returns:
            BN statistics mismatch loss

        """
        _ = (labels, target_gradients, ctx)
        if not self._is_setup:
            raise RuntimeError(
                "BNStatisticsRegularization must be setup() before use. "
                "Call bn_loss.setup(ctx) before optimization."
            )

        reg_loss = self.strategy.compute_regularization(reconstruction)
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
            required_capabilities={},
        )

    def get_strategy_requirements(self) -> dict:
        """Get requirements from the BN statistics strategy."""
        metadata = self.strategy.get_metadata()
        return metadata.required_capabilities


@register("loss.group_consistency")
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
        seed_aggregation: AggregationStrategy | None = None,
        weight: float = 0.01,
    ) -> None:
        """Initialize group consistency regularization.

        Args:
            seed_aggregation: Strategy for computing consensus across seeds.
                When built via the registry this is ``None`` and injected
                by :func:`~leakpro.attacks.gia_attacks.modular.config.builder._build_loss_list`.
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
        labels: torch.Tensor,
        target_gradients: List[torch.Tensor],
        ctx: "RunContext",
    ) -> torch.Tensor:
        """Compute group consistency regularization loss.

        Args:
            reconstruction: Current reconstruction, shape [E, B, G, C, H, W]
            labels: Not used
            target_gradients: Not used
            ctx: Not used

        Returns:
            Group consistency loss (scalar)

        """
        _ = (labels, target_gradients, ctx)

        num_epochs, num_images, num_seeds = reconstruction.shape[:3]

        if num_seeds == 1:
            return torch.tensor(0.0, device=reconstruction.device)

        consensus = self.seed_aggregation.compute_consensus(reconstruction)

        total_loss = torch.tensor(0.0, device=reconstruction.device)

        for g in range(num_seeds):
            seed_reconstruction = reconstruction[:, :, g, ...]
            deviation = (seed_reconstruction - consensus).pow(2).mean()
            total_loss += deviation

        total_loss = total_loss / num_seeds

        return self.weight * total_loss

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Return metadata for group consistency regularization."""
        return ComponentMetadata(
            name="GroupConsistencyRegularization",
            required_capabilities={},
        )


@register("loss.epoch_order_invariant")
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
            required_capabilities={},
        )

    def compute(
        self,
        reconstruction: torch.Tensor,
        labels: torch.Tensor,
        target_gradients: List[torch.Tensor],
        ctx: "RunContext",
    ) -> torch.Tensor:
        """Compute epoch order-invariant prior loss.

        Args:
            reconstruction: Reconstruction in [E, B, G, C, H, W] format
            labels: Unused
            target_gradients: Unused
            ctx: Unused

        Returns:
            Loss tensor (scalar)

        """
        _ = (labels, target_gradients, ctx)

        num_epochs, num_images, num_seeds = reconstruction.shape[:3]

        if num_epochs != self.epochs:
            raise ValueError(f"Expected E={self.epochs} epochs, got {num_epochs}")

        if num_epochs == 1:
            return torch.tensor(0.0, device=reconstruction.device, dtype=reconstruction.dtype)

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

        total_loss = 0.0
        for e1 in range(num_epochs):
            for e2 in range(e1 + 1, num_epochs):
                repr1, repr2 = epoch_representations[e1], epoch_representations[e2]
                dist = (repr1 - repr2).pow(2).sum() if self.distance_function == "l2" else (repr1 - repr2).abs().sum()
                total_loss += dist

        # Normalize by number of pairs: E*(E-1)/2
        return total_loss / (num_epochs * (num_epochs - 1) / 2) if num_epochs > 1 else total_loss


@register("loss.bn_stats")
def _bn_stats_loss_factory(
    strategy: str = "running",
    weight: float = 1.0,
    momentum: float = 0.1,
) -> "BNStatisticsRegularization":
    """Registry factory for :class:`BNStatisticsRegularization`.

    Args:
        strategy: ``"running"``, ``"inferred"``, or ``"proxy"``.
        weight: Loss weight.
        momentum: EMA momentum for the ``"inferred"`` strategy.

    Returns:
        Configured :class:`BNStatisticsRegularization` instance.
    """
    from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.bn_statistics_strategies import (
        InferredBNStatisticsStrategy,
        ProxyBNStatisticsStrategy,
        RunningBNStatisticsStrategy,
    )

    _strategies = {
        "running": lambda: RunningBNStatisticsStrategy(),
        "inferred": lambda: InferredBNStatisticsStrategy(momentum=momentum),
        "proxy": lambda: ProxyBNStatisticsStrategy(),
    }
    if strategy not in _strategies:
        raise ValueError(f"Unknown bn_stats strategy '{strategy}'. Choose from: {list(_strategies)}")
    return BNStatisticsRegularization(strategy=_strategies[strategy](), weight=weight)


__all__ = [
    "LossComponent",
    "GradientMatchingLoss",
    "TVRegularization",
    "L2Regularization",
    "LatentKLDivergenceRegularization",
    "LabelEntropyRegularization",
    "BNStatisticsRegularization",
    "GroupConsistencyRegularization",
    "EpochOrderInvariantPrior",
]
