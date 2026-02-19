"""Batch normalization statistics strategies for gradient inversion attacks.

This module provides different strategies for regularizing reconstructions based on
batch normalization statistics, implementing the approaches from:
- Huang et al. (uses running BN statistics)
- GIA Running (infers BN statistics from momentum updates)
- GIA Estimate (estimates BN statistics from proxy data)

These are thin wrappers around the existing hook classes in fl_utils.model_utils.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from copy import deepcopy
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.convnext import LayerNorm2d

from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.training_simulator import (
    TrainingSimulator,
)
from leakpro.attacks.gia_attacks.modular.core.component_base import (
    Component,
    ComponentMetadata,
)
from leakpro.fl_utils.fl_client_simulator import ClientObservations
from leakpro.fl_utils.model_utils import (
    BNFeatureHook,
    InferredBNFeatureHook,
    InferredIN2dFeatureHook,
    InferredLN2dFeatureHook,
    InferredLNFeatureHook,
)

logger = logging.getLogger(__name__)


class BNStatisticsStrategy(Component):
    """Base class for batch normalization statistics strategies.

    Different strategies for obtaining or estimating the batch statistics
    that should be matched during reconstruction optimization.
    """

    def __init__(self) -> None:
        """Initialize common state for all strategies."""
        self.feature_hooks: List = []

    @abstractmethod
    def setup(
        self,
        model: nn.Module,
        reconstruction: torch.Tensor | None = None,
        client_observations: ClientObservations | None = None,
        training_simulator: TrainingSimulator | None = None,
        proxy_dataloader: DataLoader | None = None,
    ) -> None:
        """Setup the strategy (e.g., register hooks, compute statistics).

        Args:
            model: The target model
            reconstruction: Current reconstruction tensor (optional, strategy-specific)
            client_observations: ClientObservations from FL client (optional, strategy-specific)
            training_simulator: Training simulator for forward passes (optional, strategy-specific)
            proxy_dataloader: Server-side dataloader (optional, strategy-specific)

        """
        pass

    def compute_regularization(
        self,
        _model: nn.Module,
        reconstruction: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the BN statistics regularization loss.

        Note: This should be called AFTER a forward pass has been done
        (e.g., during gradient computation), as the hooks compute r_feature
        during the forward pass.

        Args:
            _model: The target model (not used, hooks already attached)
            reconstruction: Current reconstruction (used for device detection)

        Returns:
            Regularization loss (scalar)

        """
        # Simply sum the r_feature values computed by hooks during forward pass
        if self.feature_hooks:
            return sum(hook.r_feature for hook in self.feature_hooks)
        return torch.tensor(0.0, device=reconstruction.device)

    def cleanup(self) -> None:
        """Clean up any hooks or resources."""
        for hook in self.feature_hooks:
            hook.close()
        self.feature_hooks = []

    @classmethod
    @abstractmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Return metadata describing this strategy's requirements."""
        pass


class RunningBNStatisticsStrategy(BNStatisticsStrategy):
    """Use running BN statistics from the model (Huang et al. approach).

    This strategy uses the model's stored running_mean and running_var
    from BatchNorm layers as targets for regularization. This assumes
    the attacker has access to these statistics.

    This is a modular wrapper around the logic from huang.py.

    Reference:
        Huang et al. "Evaluating Gradient Inversion Attacks and Defenses
        in Federated Learning." NeurIPS 2021.
    """

    def __init__(self) -> None:
        """Initialize running BN statistics strategy."""
        super().__init__()
        self.name = self.__class__.get_metadata().name

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Return metadata describing this strategy's requirements."""
        return ComponentMetadata(
            name="running_bn_statistics",
            display_name="Running BN Statistics",
            description="Uses model's running BN statistics (Huang et al.)",
            required_capabilities={"has_bn_statistics": True},
            paper_reference="Huang et al., NeurIPS 2021",
        )

    def setup(
        self,
        model: nn.Module,
        reconstruction: torch.Tensor | None = None,
        client_observations: ClientObservations | None = None,
        training_simulator: TrainingSimulator | None = None,
        proxy_dataloader: DataLoader | None = None,
    ) -> None:
        """Setup hooks to track BN statistics during forward pass.

        Args:
            model: The target model
            reconstruction: Not used by this strategy
            client_observations: ClientObservations from FL client (contains post_bn_stats)
            training_simulator: Not used by this strategy
            proxy_dataloader: Not used by this strategy

        """
        _ = (reconstruction, training_simulator, proxy_dataloader)  # Unused args
        self.cleanup()  # Remove any existing hooks
        self.feature_hooks = []

        # Check if client provided post-training BN statistics
        post_train_running_stats = None
        if client_observations is not None:
            post_train_running_stats = client_observations.post_bn_stats

        if post_train_running_stats is None:
            # No BN statistics provided by client - skip setup
            # Strategy will return 0 loss in compute_regularization
            logger.warning(
                f"{self.name}: Client did not provide post-training BN statistics. "
                "BN regularization will be disabled (loss = 0)."
            )
            return

        # Load the post-training statistics into the model's running_mean/var
        # This is what the client had after training
        bn_layer_idx = 0
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d) and bn_layer_idx < len(post_train_running_stats):
                    running_mean, running_var = post_train_running_stats[bn_layer_idx]
                    module.running_mean.data.copy_(running_mean.to(module.running_mean.device))
                    module.running_var.data.copy_(running_var.to(module.running_var.device))
                    bn_layer_idx += 1

        # Register BNFeatureHook on all BatchNorm2d layers (same as huang.py)
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.feature_hooks.append(BNFeatureHook(module))

        # Freeze BN running statistics by setting momentum=0
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.momentum = 0


class InferredBNStatisticsStrategy(BNStatisticsStrategy):
    """Infer BN statistics from running statistics momentum (GIA Running approach).

    This strategy infers the client's batch statistics by observing how
    the running statistics changed during client training, using the
    momentum-based update equation.

    Running statistics update: running_stat = (1-m)*running_stat + m*batch_stat
    Therefore: batch_stat = (running_stat_new - (1-m)*running_stat_old) / m

    This is a modular wrapper around the logic from gia_running.py.

    Reference:
        GIA Running implementation (gia_running.py).
    """

    def __init__(self, momentum: float = 0.1) -> None:
        """Initialize inferred BN statistics strategy.

        Args:
            momentum: BN momentum parameter (default PyTorch value is 0.1)

        """
        super().__init__()
        self.name = self.__class__.get_metadata().name
        self.momentum = momentum

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Return metadata describing this strategy's requirements."""
        return ComponentMetadata(
            name="inferred_bn_statistics",
            display_name="Inferred BN Statistics",
            description="Infers BN statistics from momentum updates (GIA Running)",
            required_capabilities={
                "has_bn_statistics": True,
                "has_local_hyperparameters": True,
            },
            paper_reference="GIA Running implementation",
        )

    def setup(
        self,
        model: nn.Module,
        reconstruction: torch.Tensor | None = None,
        client_observations: ClientObservations | None = None,
        training_simulator: TrainingSimulator | None = None,
        proxy_dataloader: DataLoader | None = None,
    ) -> None:
        """Setup by inferring batch statistics from running statistics changes.

        Args:
            model: The target model
            reconstruction: Current reconstruction (used to infer num_images if client didn't provide it)
            client_observations: ClientObservations from FL client (contains pre/post_bn_stats, num_images)
            training_simulator: Optional training simulator (not used in this strategy)
            proxy_dataloader: Optional proxy dataloader (not used in this strategy)

        """
        _ = training_simulator, proxy_dataloader  # Unused args
        self.cleanup()
        self.feature_hooks = []

        # Extract BN statistics from client observations
        pre_train_running_stats = None
        post_train_running_stats = None
        num_images = None

        if client_observations is not None:
            pre_train_running_stats = client_observations.pre_bn_stats
            post_train_running_stats = client_observations.post_bn_stats
            num_images = client_observations.num_images

        # Check if client provided necessary statistics
        if pre_train_running_stats is None or post_train_running_stats is None:
            # Client did not provide pre/post BN statistics - skip setup
            # Strategy will return 0 loss in compute_regularization
            logger.warning(
                f"{self.name}: Client did not provide pre/post-training BN statistics. "
                "BN regularization will be disabled (loss = 0). "
                f"(pre_stats={'provided' if pre_train_running_stats else 'missing'}, "
                f"post_stats={'provided' if post_train_running_stats else 'missing'})"
            )
            return

        # Get num_images - fallback to reconstruction shape if not provided by client
        if num_images is None and reconstruction is not None:
            num_images = reconstruction.shape[0]
        elif num_images is None:
            # Cannot proceed without num_images
            logger.warning(
                f"{self.name}: Cannot infer num_images (no client num_images and no reconstruction). "
                "BN regularization will be disabled (loss = 0)."
            )
            return

        # Infer batch statistics from momentum updates (from gia_running.py)
        used_statistics = []
        for (rm_pre, rv_pre), (rm_post, rv_post) in zip(
            pre_train_running_stats, post_train_running_stats
        ):
            # Infer what batch statistics were used during training
            # Batch stat = (running_new - (1-m)*running_old) / m
            # For default momentum=0.1: batch_stat = 10*running_new - 9*running_old
            used_mean = (rm_post - (1 - self.momentum) * rm_pre) / self.momentum
            used_var = (rv_post - (1 - self.momentum) * rv_pre) / self.momentum
            used_statistics.append((used_mean, used_var))

        # Apply bias correction for variance (PyTorch uses biased variance internally)
        client_statistics = []
        for i in range(len(used_statistics)):
            used_mean, used_var = used_statistics[i]
            correction_factor = (num_images - 1) / num_images
            client_statistics.append((used_mean, used_var * correction_factor))

        # Register InferredBNFeatureHook on all BatchNorm2d layers (same as gia_running.py)
        start_idx = 0
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.feature_hooks.append(InferredBNFeatureHook(
                    module,
                    client_statistics[start_idx][0],
                    client_statistics[start_idx][1]))
                start_idx += 1

        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.momentum = 0


class ProxyBNStatisticsStrategy(BNStatisticsStrategy):
    """Estimate BN statistics from proxy data (GIA Estimate approach).

    This strategy uses a proxy dataset (from the same or similar domain)
    to estimate what the batch statistics might look like. This is more
    realistic than having access to exact running statistics.

    Supports multiple normalization types:
    - BatchNorm2d
    - LayerNorm
    - LayerNorm2d (ConvNeXt)
    - InstanceNorm2d

    This is a modular wrapper around the logic from gia_estimate.py.

    Reference:
        GIA Estimate implementation (gia_estimate.py).
    """

    def __init__(self) -> None:
        """Initialize proxy BN statistics strategy."""
        super().__init__()
        self.name = self.__class__.get_metadata().name

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Return metadata describing this strategy's requirements."""
        return ComponentMetadata(
            name="proxy_bn_statistics",
            display_name="Proxy BN Statistics",
            description="Estimates BN statistics from proxy data (GIA Estimate)",
            required_capabilities={"has_surrogate_data": True},
            paper_reference="GIA Estimate implementation",
        )

    def _create_statistics_hooks(self, proxy_statistics: List) -> Dict:
        """Create hook functions for collecting normalization statistics.

        Args:
            proxy_statistics: List to append collected statistics to

        Returns:
            Dictionary mapping module types to hook functions

        """

        def bn_forward_hook(_module: nn.Module, input: tuple, _output: torch.Tensor) -> None:
            """Collect BN statistics."""
            x = input[0]
            batch_mean = x.mean(dim=(0, 2, 3))
            batch_var = x.var(dim=(0, 2, 3), unbiased=False)
            proxy_statistics.append((batch_mean.detach(), batch_var.detach()))

        def ln_forward_hook(_module: nn.Module, input: tuple, _output: torch.Tensor) -> None:
            """Collect LayerNorm statistics."""
            x = input[0]
            m = x.mean()
            v = x.var(unbiased=False)
            proxy_statistics.append((m.detach(), v.detach()))

        def ln2d_forward_hook(_module: nn.Module, input: tuple, _output: torch.Tensor) -> None:
            """Collect LayerNorm2d statistics."""
            x = input[0]
            m = x.mean(dim=1).mean()
            v = x.var(dim=1, unbiased=False).mean()
            proxy_statistics.append((m.detach(), v.detach()))

        def in2d_forward_hook(_module: nn.Module, input: tuple, _output: torch.Tensor) -> None:
            """Collect InstanceNorm2d statistics."""
            x = input[0]  # NCHW
            m = x.mean(dim=(2, 3))  # [N, C]
            v = x.var(dim=(2, 3), unbiased=False)  # [N, C]
            proxy_statistics.append((m.mean(dim=1).mean().detach(), v.mean(dim=1).mean().detach()))

        return {
            nn.BatchNorm2d: bn_forward_hook,
            nn.LayerNorm: ln_forward_hook,
            LayerNorm2d: ln2d_forward_hook,
            nn.InstanceNorm2d: in2d_forward_hook,
        }

    def _collect_proxy_statistics(
        self, proxy_model: nn.Module, proxy_dataloader: DataLoader
    ) -> List:
        """Collect statistics from proxy data using forward hooks.

        Args:
            proxy_model: Copy of the model to use for statistics collection
            proxy_dataloader: DataLoader with proxy data

        Returns:
            List of (mean, variance) tuples for each normalization layer

        """
        proxy_statistics = []
        hook_map = self._create_statistics_hooks(proxy_statistics)

        # Register temporary hooks to collect statistics
        hooks = []
        for module in proxy_model.modules():
            for module_type, hook_fn in hook_map.items():
                if isinstance(module, module_type):
                    hooks.append(module.register_forward_hook(hook_fn))
                    break

        # Run forward pass on one batch to collect statistics
        with torch.no_grad():
            for inputs, _labels in proxy_dataloader:
                device = next(proxy_model.parameters()).device
                _ = proxy_model(inputs.to(device))
                break  # Only need one batch

        # Remove hooks
        for h in hooks:
            h.remove()

        return proxy_statistics

    def _register_feature_hooks(self, model: nn.Module, proxy_statistics: List) -> None:
        """Register feature hooks on model normalization layers.

        Args:
            model: The target model
            proxy_statistics: List of (mean, variance) tuples from proxy data

        """
        start_idx = 0
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.feature_hooks.append(
                    InferredBNFeatureHook(
                        module, proxy_statistics[start_idx][0], proxy_statistics[start_idx][1]
                    )
                )
                start_idx += 1
            elif isinstance(module, nn.LayerNorm):
                self.feature_hooks.append(
                    InferredLNFeatureHook(
                        module, proxy_statistics[start_idx][0], proxy_statistics[start_idx][1]
                    )
                )
                start_idx += 1
            elif isinstance(module, LayerNorm2d):
                self.feature_hooks.append(
                    InferredLN2dFeatureHook(
                        module, proxy_statistics[start_idx][0], proxy_statistics[start_idx][1]
                    )
                )
                start_idx += 1
            elif isinstance(module, nn.InstanceNorm2d):
                self.feature_hooks.append(
                    InferredIN2dFeatureHook(
                        module, proxy_statistics[start_idx][0], proxy_statistics[start_idx][1]
                    )
                )
                start_idx += 1

    def setup(
        self,
        model: nn.Module,
        reconstruction: torch.Tensor | None = None,
        client_observations: ClientObservations | None = None,
        training_simulator: TrainingSimulator | None = None,
        proxy_dataloader: DataLoader | None = None,
    ) -> None:
        """Setup by computing statistics from proxy data.

        Args:
            model: The target model
            reconstruction: Not used by this strategy
            client_observations: Not used by this strategy (server uses own proxy data)
            training_simulator: Training simulator for forward passes
            proxy_dataloader: DataLoader with proxy data (server's own data, not from client)

        """
        _ = (reconstruction, client_observations, training_simulator)  # Unused args
        self.cleanup()
        self.feature_hooks = []

        # Check if proxy data is provided
        if proxy_dataloader is None:
            logger.warning(
                f"{self.name}: No proxy dataloader provided. "
                "BN regularization will be disabled (loss = 0). "
                "Pass 'proxy_dataloader' to run_attack() to enable this strategy."
            )
            return

        # Collect statistics from proxy data
        proxy_model = deepcopy(model)
        proxy_statistics = self._collect_proxy_statistics(proxy_model, proxy_dataloader)

        # Register feature hooks on the target model
        self._register_feature_hooks(model, proxy_statistics)

        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.momentum = 0


__all__ = [
    "BNStatisticsStrategy",
    "RunningBNStatisticsStrategy",
    "InferredBNStatisticsStrategy",
    "ProxyBNStatisticsStrategy",
]

