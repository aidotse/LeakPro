#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Label inference strategies for gradient inversion attacks.

Label inference determines or optimizes labels from gradients. This is often the first
step in a GIA attack, as knowing the correct labels significantly improves reconstruction.

Strategies:
    - IDLGLabelInference: Analytical label inference using gradient signs (iDLG paper)
    - AnalyticalLabelInference: Closed-form solution for single samples
    - BatchIDLGLabelInference: iDLG extended for batch scenarios
    - JointLabelOptimization: Optimize labels jointly with reconstruction
    - OracleLabels: Use ground-truth labels (for debugging/upper bound)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn.functional import one_hot, softmax

from leakpro.attacks.gia_attacks.modular.config.registry import register
from leakpro.attacks.gia_attacks.modular.core.component_base import (
    ComponentMetadata,
    LabelInferenceResult,
    LabelInferenceStrategy,
)
from leakpro.attacks.gia_attacks.modular.core.label_types import LabelType

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.state import RunContext


@register("label_inference.idlg")
class IDLGLabelInference(LabelInferenceStrategy):
    """Improved Deep Leakage from Gradients (iDLG) label inference.

    Uses analytical gradient sign analysis to infer labels. For cross-entropy loss
    with softmax, the bias gradient of the last layer directly reveals label information:
    grad_b = sum_i(softmax(logit_i) - one_hot(y_i))

    The true label positions have the most negative gradient values.

    Reference:
        Zhao et al., "iDLG: Improved Deep Leakage from Gradients", 2020
    """

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Get metadata for iDLG label inference strategy."""
        return ComponentMetadata(
            name="idlg",
            required_capabilities={"has_gradients": True},
        )

    def infer(self, ctx: "RunContext") -> LabelInferenceResult:
        """Infer labels from last layer bias gradient.

        Args:
            ctx: Run context with target_model and client_observations.

        Returns:
            LabelInferenceResult with inferred labels

        """
        obs = ctx.client_observations
        gradients = [
            obs.gradients[name]
            for name, _ in ctx.target_model.named_parameters()
            if name in obs.gradients
        ]
        num_samples = obs.input_shape[0]
        is_update = (
            obs.training_settings is not None
            and obs.training_settings.compute_mode == "updates"
        )

        last_bias_grad = None
        last_weight_grad = None

        for grad in reversed(gradients):
            if grad is not None and isinstance(grad, torch.Tensor):
                if grad.dim() == 1 and last_bias_grad is None:
                    last_bias_grad = grad
                elif grad.dim() == 2 and last_weight_grad is None:
                    last_weight_grad = grad

                if last_bias_grad is not None:
                    break

        if last_bias_grad is None and last_weight_grad is not None:
            last_bias_grad = last_weight_grad.sum(dim=1)

        if last_bias_grad is None:
            raise ValueError(
                "Could not find last layer bias or weight gradient. "
                "iDLG requires gradients from the final layer."
            )

        # Updates: Δθ = -lr * ∇L, so negate to get gradient direction
        if is_update:
            last_bias_grad = -last_bias_grad

        # For cross-entropy with softmax: grad_b = softmax - one_hot
        # Negative positions indicate true labels
        if num_samples == 1:
            label = torch.argmin(last_bias_grad).unsqueeze(0)
            confidence = softmax(-last_bias_grad, dim=0).unsqueeze(0)
        else:
            sorted_indices = torch.argsort(last_bias_grad)
            label = sorted_indices[:num_samples]
            confidence = softmax(
                -last_bias_grad.unsqueeze(0).expand(num_samples, -1), dim=-1
            )

        return LabelInferenceResult(
            labels=label,
            confidence=confidence,
            method="idlg",
            metadata={"gradient": last_bias_grad.detach()},
        )


@register("label_inference.joint")
class JointLabelOptimization(LabelInferenceStrategy):
    """Optimize labels jointly with reconstruction (DLG style).

    Instead of inferring labels analytically, initialize soft labels that will
    be optimized together with the reconstruction during the attack.
    """

    def __init__(self, num_classes: int = 10, init_strategy: str = "uniform") -> None:
        """Initialize joint label optimization strategy.

        Args:
            num_classes: Number of output classes
            init_strategy: Initialization strategy ("uniform", "random", or "idlg_warm")

        """
        self.num_classes = num_classes
        self.init_strategy = init_strategy

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Get metadata for joint label optimization strategy."""
        return ComponentMetadata(
            name="joint_optimization",
            required_capabilities={"has_gradients": True},
        )

    def infer(self, ctx: "RunContext") -> LabelInferenceResult:
        """Initialize soft labels for optimization."""
        obs = ctx.client_observations
        model = ctx.target_model
        num_samples = obs.input_shape[0]
        gradients = [
            obs.gradients[name]
            for name, _ in model.named_parameters()
            if name in obs.gradients
        ]
        device = gradients[0].device if gradients else torch.device("cpu")

        num_classes = self.num_classes
        for m in reversed(list(model.modules())):
            if hasattr(m, "out_features"):
                num_classes = m.out_features
                break

        if self.init_strategy == "idlg_warm":
            idlg = IDLGLabelInference()
            try:
                idlg_result = idlg.infer(ctx)
                soft_labels = one_hot(idlg_result.labels, num_classes).float()
            except (ValueError, RuntimeError):
                soft_labels = torch.ones(num_samples, num_classes, device=device) / num_classes
        elif self.init_strategy == "uniform":
            soft_labels = torch.ones(num_samples, num_classes, device=device) / num_classes
        elif self.init_strategy == "random":
            soft_labels = torch.randn(num_samples, num_classes, device=device)
            soft_labels = softmax(soft_labels, dim=-1)
        else:
            soft_labels = torch.ones(num_samples, num_classes, device=device) / num_classes

        hard_labels = torch.argmax(soft_labels, dim=-1)

        return LabelInferenceResult(
            labels=hard_labels,
            confidence=soft_labels,
            method="joint_optimization",
            metadata={"soft_labels": soft_labels, "requires_optimization": True},
        )


@register("label_inference.oracle")
class OracleLabels(LabelInferenceStrategy):
    """Use ground-truth labels (for debugging/upper bound analysis).

    This is not a realistic attack scenario, but useful for:
    - Debugging reconstruction quality independent of label inference
    - Establishing upper bounds on reconstruction performance
    - Testing optimization strategies
    """

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Get metadata for oracle label inference strategy."""
        return ComponentMetadata(
            name="oracle",
            required_capabilities={},  # No requirements - uses provided labels
        )

    def infer(self, ctx: "RunContext") -> LabelInferenceResult:
        """Return provided ground-truth labels."""
        true_labels = ctx.client_observations.labels
        if true_labels is None:
            raise ValueError(
                "OracleLabels requires labels in client_observations"
            )

        if not isinstance(true_labels, torch.Tensor):
            true_labels = torch.tensor(true_labels)

        label_type = LabelType.auto_detect(true_labels)
        confidence = label_type.to_confidence(true_labels)

        return LabelInferenceResult(
            labels=true_labels,
            confidence=confidence,
            method="oracle",
            label_type=label_type,
            metadata={"ground_truth": True},
        )

@register("label_inference.geng")
class GengLabelInference(LabelInferenceStrategy):
    """Geng et al. label inference for single-step gradient attacks (FedSGD).

    This is a special case of DimitrovLabelInference with epochs=1.
    Uses network statistics (softmax outputs and ReLU activations) to analytically
    estimate label counts from gradients.

    Reference:
        Geng et al., "Towards General Deep Leakage in Federated Learning"
    """

    def __init__(
        self,
        num_dummy_samples: int = 100,
    ) -> None:
        """Initialize Geng label inference.

        Args:
            num_dummy_samples: Number of random samples for statistics estimation

        """
        # Geng is just Dimitrov with epochs=1
        self._dimitrov = DimitrovLabelInference(
            epochs=1,
            batch_size=1,  # Will be overridden based on num_samples
            learning_rate=0.1,  # Not used when epochs=1
            num_dummy_samples=num_dummy_samples,
            strategy="start",  # Doesn't matter when epochs=1
        )

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Get metadata for Geng label inference."""
        return ComponentMetadata(
            name="geng",
            required_capabilities={"has_gradients": True},
        )

    def infer(self, ctx: "RunContext") -> LabelInferenceResult:
        """Infer label counts using Geng's analytical method.

        Delegates to DimitrovLabelInference with epochs=1, which is
        mathematically equivalent to Geng's formula.

        Args:
            ctx: Run context with target_model and client_observations.

        Returns:
            LabelInferenceResult with inferred labels

        """
        self._dimitrov.batch_size = ctx.client_observations.input_shape[0]
        result = self._dimitrov.infer(ctx)
        result.method = "geng"
        return result


@register("label_inference.dimitrov")
class DimitrovLabelInference(LabelInferenceStrategy):
    """Dimitrov label inference for multi-epoch FedAvg attacks.

    This method uses interpolated network statistics (softmax outputs and
    ReLU activations) between the server and client models to analytically
    estimate label counts in the batch. It's specifically designed for
    multi-epoch federated averaging scenarios.

    The algorithm:
    1. Computes statistics at server (pre-update) and client (post-update) models
    2. Interpolates statistics across training steps
    3. Uses the formula: counts = K*ps - (K*dW)/(O*s)
       where ps = softmax probabilities, dW = weight gradient sum,
       O = ReLU activation sum, s = total steps, K = batch size
    4. Rounds fractional counts to integers

    Reference:
        Dimitrov et al., "Data Leakage in Federated Averaging", TMLR 2022
    """

    def __init__(
        self,
        epochs: int = 3,
        batch_size: int = 1,
        learning_rate: float = 0.1,
        num_dummy_samples: int = 100,
        strategy: str = "avg"
    ) -> None:
        """Initialize Dimitrov label inference.

        Args:
            epochs: Number of training epochs
            batch_size: Batch size used in training
            learning_rate: Learning rate used in client training
            num_dummy_samples: Number of random samples for statistics estimation
            strategy: Estimation strategy ('start', 'end', or 'avg')

        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_dummy_samples = num_dummy_samples
        self.strategy = strategy

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Get metadata for Dimitrov label inference."""
        return ComponentMetadata(
            name="dimitrov",
            required_capabilities={"has_gradients": True, "has_local_hyperparameters": True},
        )

    def _calc_label_stats(
        self,
        model: nn.Module,
        params: dict[str, torch.Tensor],
        input_shape: tuple[int, ...],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate label statistics from model.

        Args:
            model: The neural network model
            params: Model parameters
            input_shape: Shape of input images
            device: Computation device

        Returns:
            Tuple of (dW, O, ps) where:
            - dW: Sum of weight gradients in last layer
            - O: Mean sum of last ReLU activations
            - ps: Mean softmax probabilities

        """
        # Generate dummy inputs
        dummy_inputs = torch.randn(
            self.num_dummy_samples, *input_shape,
            device=device
        )

        # Temporarily load parameters into model
        original_state = model.state_dict()

        # Create full state dict with both trainable params and buffers (BN stats)
        new_state = {}
        for name, value in original_state.items():
            new_state[name] = params.get(name, value)
        model.load_state_dict(new_state)
        model.eval()

        # Forward pass with hooks to capture last ReLU
        last_relu_outputs = []

        def hook_fn(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
            _ = (module, input) # Not used
            # Capture the ReLU output for all samples in batch
            last_relu_outputs.append(output.detach())

        # Find last ReLU layer
        last_relu = None
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                last_relu = module

        if last_relu is not None:
            handle = last_relu.register_forward_hook(hook_fn)

        # Get softmax probabilities
        with torch.no_grad():
            outputs = model(dummy_inputs)
            softmax_probs = torch.softmax(outputs, dim=1)

        if last_relu is not None:
            handle.remove()

        ps = softmax_probs.mean(dim=0)

        output_sum = last_relu_outputs[0].sum(dim=-1).mean() if last_relu_outputs else outputs.abs().sum(dim=-1).mean()

        # Restore original parameters
        model.load_state_dict(original_state)

        return ps, output_sum

    def _calc_raw_counts(
        self,
        ps: torch.Tensor,
        output_sum: torch.Tensor,
        weight_grad_sum: torch.Tensor,
        total_steps: int,
        batch_size_k: int,
    ) -> torch.Tensor:
        """Calculate raw (fractional) label counts.

        Formula: counts = K*ps - (K*dW)/(O*s)

        Args:
            ps: Softmax probabilities
            output_sum: ReLU activation sum
            weight_grad_sum: Weight gradient sum
            total_steps: Total training steps
            batch_size_k: Batch size

        Returns:
            Raw label counts (may be fractional)

        """
        return batch_size_k * ps - (batch_size_k * weight_grad_sum) / (output_sum * total_steps)

    def _round_label_counts(self, counts: torch.Tensor, batch_size_k: int) -> torch.Tensor:
        """Round fractional label counts to integers summing to K.

        Args:
            counts: Fractional label counts
            batch_size_k: Target sum (number of samples)

        Returns:
            Integer label counts summing to K

        """
        # Floor all counts
        counts_floor = torch.clamp(torch.floor(counts), min=0).long()
        counts_rem = batch_size_k - counts_floor.sum()

        # Fractional remainders
        remainders = counts - counts_floor.float()

        if counts_rem >= 0:
            # Need to add counts_rem more labels
            # Add to positions with largest remainders
            _, top_indices = torch.topk(remainders, min(int(counts_rem), len(remainders)))
            counts_floor[top_indices] += 1
        else:
            # Need to remove -counts_rem labels
            max_remove = -int(counts_rem)

            # First, mark positions that can't be removed (already 0 or very small)
            can_remove = counts_floor >= 1
            num_can_remove = can_remove.sum().item()

            if max_remove <= num_can_remove:
                # Remove from positions with smallest remainders among those that can be removed
                masked_remainders = remainders.clone()
                masked_remainders[~can_remove] = float("inf")
                _, bottom_indices = torch.topk(-masked_remainders, max_remove)
                counts_floor[bottom_indices] -= 1
            else:
                # Remove all we can, then remove from largest counts
                counts_floor[can_remove] -= 1
                remaining_to_remove = max_remove - num_can_remove
                # Ensure we don't try to remove more than available
                remaining_to_remove = min(remaining_to_remove, len(counts_floor))
                if remaining_to_remove > 0:
                    _, top_indices = torch.topk(counts_floor.float(), remaining_to_remove)
                    counts_floor[top_indices] -= 1

        return torch.clamp(counts_floor, min=0)

    def _compute_label_counts_by_strategy(
        self,
        ps_start: torch.Tensor,
        output_sum_start: torch.Tensor,
        ps_end: torch.Tensor,
        output_sum_end: torch.Tensor,
        weight_grad_sum: torch.Tensor,
        total_steps: int,
        num_samples: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute label counts based on the selected strategy.

        Args:
            ps_start: Softmax probabilities at start
            output_sum_start: ReLU activation sum at start
            ps_end: Softmax probabilities at end
            output_sum_end: ReLU activation sum at end
            weight_grad_sum: Weight gradient sum
            total_steps: Total training steps
            num_samples: Number of samples
            device: Computation device

        Returns:
            Raw label counts

        """
        if self.strategy == "start":
            return self._calc_raw_counts(
                ps_start, output_sum_start, weight_grad_sum, total_steps, num_samples
            )
        if self.strategy == "end":
            return self._calc_raw_counts(
                ps_end, output_sum_end, weight_grad_sum, total_steps, num_samples
            )
        # "avg" strategy
        # Average over interpolated points
        num_interp = total_steps
        coefs = torch.linspace(0, 1, num_interp, device=device)

        all_raw_counts = []
        for coef in coefs:
            ps_interp = (1 - coef) * ps_start + coef * ps_end
            output_sum_interp = (1 - coef) * output_sum_start + coef * output_sum_end
            raw_counts_i = self._calc_raw_counts(
                ps_interp, output_sum_interp, weight_grad_sum, total_steps, num_samples
            )
            all_raw_counts.append(raw_counts_i)

        return torch.stack(all_raw_counts).mean(dim=0)

    def _infer_input_shape(self, model: nn.Module) -> tuple[int, ...]:
        """Infer input shape from model architecture.

        Args:
            model: The neural network model

        Returns:
            Input shape tuple

        Raises:
            ValueError: If input shape cannot be inferred

        """
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                # Assume square images
                in_channels = module.in_channels
                # Default to 32x32 for CIFAR-like datasets
                return (in_channels, 32, 32)
            if isinstance(module, nn.Linear):
                in_features = module.in_features
                return (in_features,)

        raise ValueError("Could not infer input shape from model")

    def _extract_weight_gradient_sum(self, gradients: list[torch.Tensor]) -> torch.Tensor:
        """Extract and sum last layer weight gradient.

        Args:
            gradients: List of gradient tensors

        Returns:
            Summed weight gradient

        Raises:
            ValueError: If weight gradient cannot be found

        """
        last_weight_grad = None
        for grad in reversed(gradients):
            if grad is not None and isinstance(grad, torch.Tensor) and grad.dim() == 2:
                last_weight_grad = grad
                break

        if last_weight_grad is None:
            raise ValueError("Could not find last layer weight gradient")

        return last_weight_grad.sum(dim=1)

    def _compute_client_params(
        self,
        gradients: list[torch.Tensor],
        model: nn.Module,
        server_params: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute client parameters after update.

        Args:
            gradients: List of gradient tensors
            model: The neural network model
            server_params: Server parameters before update

        Returns:
            Client parameters after update

        """
        grad_list = list(gradients)
        param_list = list(model.named_parameters())

        if len(grad_list) != len(param_list):
            # Gradients list might be shorter or have different structure
            return server_params

        client_params = {}
        for (name, param), grad in zip(param_list, grad_list):
            if grad is not None and isinstance(grad, torch.Tensor):
                client_params[name] = param.data - self.learning_rate * grad
            else:
                client_params[name] = param.data.clone()

        return client_params

    def _create_confidence_matrix(
        self,
        labels_tensor: torch.Tensor,
        num_samples: int,
        num_classes: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create confidence matrix from label tensor.

        Args:
            labels_tensor: Predicted labels
            num_samples: Number of samples
            num_classes: Number of classes
            device: Computation device

        Returns:
            Confidence matrix

        """
        confidence = torch.zeros(num_samples, num_classes, device=device)
        for i, label in enumerate(labels_tensor):
            confidence[i, label] = 1.0
        return confidence

    def infer(self, ctx: "RunContext") -> LabelInferenceResult:
        """Infer label counts using Dimitrov's analytical method.

        Args:
            ctx: Run context with target_model and client_observations.

        Returns:
            LabelInferenceResult with inferred labels

        """
        obs = ctx.client_observations
        model = ctx.target_model
        num_samples = obs.input_shape[0]
        gradients = [
            obs.gradients[name]
            for name, _ in model.named_parameters()
            if name in obs.gradients
        ]
        is_update = (
            obs.training_settings is not None
            and obs.training_settings.compute_mode == "updates"
        )

        if is_update:
            gradients = [-g for g in gradients]

        device = gradients[0].device

        # Use actual input shape from observations (C, H, W); fall back to model inference
        input_shape = obs.input_shape[1:] if len(obs.input_shape) > 1 else self._infer_input_shape(model)
        weight_grad_sum = self._extract_weight_gradient_sum(gradients)

        server_params = {name: param.data.clone() for name, param in model.named_parameters()}
        client_params = self._compute_client_params(gradients, model, server_params)

        ps_start, output_sum_start = self._calc_label_stats(model, server_params, input_shape, device)
        ps_end, output_sum_end = self._calc_label_stats(model, client_params, input_shape, device)

        k_batches = (num_samples + self.batch_size - 1) // self.batch_size
        total_steps = k_batches * self.epochs

        raw_counts = self._compute_label_counts_by_strategy(
            ps_start, output_sum_start, ps_end, output_sum_end,
            weight_grad_sum, total_steps, num_samples, device
        )

        label_counts = self._round_label_counts(raw_counts, num_samples)

        labels = []
        for label_idx, count in enumerate(label_counts):
            labels.extend([label_idx] * int(count.item()))

        labels_tensor = torch.tensor(labels[:num_samples], dtype=torch.long, device=device)

        num_classes = len(label_counts)
        confidence = self._create_confidence_matrix(labels_tensor, num_samples, num_classes, device)

        return LabelInferenceResult(
            labels=labels_tensor,
            confidence=confidence,
            method="dimitrov",
            metadata={
                "raw_counts": raw_counts.detach(),
                "rounded_counts": label_counts,
                "strategy": self.strategy,
            },
        )


__all__ = [
    "IDLGLabelInference",
    "JointLabelOptimization",
    "OracleLabels",
    "GengLabelInference",
    "DimitrovLabelInference",
]
