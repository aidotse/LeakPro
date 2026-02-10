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

import torch
from torch import nn
from torch.nn.functional import one_hot, softmax

from leakpro.attacks.gia_attacks.modular.core.component_base import (
    ComponentMetadata,
    LabelInferenceResult,
    LabelInferenceStrategy,
)


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
            display_name="iDLG Label Inference",
            description="Analytical label inference using gradient signs (iDLG)",
            required_capabilities={"has_gradients": True},
            paper_reference="Zhao et al., iDLG, 2020",
        )

    def infer_labels(
        self,
        gradients: list[torch.Tensor],
        model: nn.Module,
        num_samples: int,
        true_labels: torch.Tensor | None = None,
    ) -> LabelInferenceResult:
        """Infer labels from last layer bias gradient.

        Args:
            gradients: List of gradient tensors (ordered as model.parameters())
            model: Target model (not used but kept for interface consistency)
            num_samples: Number of samples in batch
            true_labels: Not used by this strategy

        Returns:
            LabelInferenceResult with inferred labels

        """

        _ = (model, true_labels)  # Unused parameters, but kept for interface consistency

        # Find the last bias gradient (1D tensor)
        last_bias_grad = None
        for grad in reversed(gradients):
            if grad is not None and grad.dim() == 1:
                last_bias_grad = grad
                break

        if last_bias_grad is None:
            raise ValueError(
                "Could not find last layer bias gradient. "
                "iDLG requires a bias in the final layer."
            )

        # For cross-entropy with softmax: grad_b = softmax - one_hot
        # Negative positions indicate true labels
        if num_samples == 1:
            # Single sample: most negative position is the label
            label = torch.argmin(last_bias_grad).unsqueeze(0)
            confidence = softmax(-last_bias_grad, dim=0).unsqueeze(0)
        else:
            # Batch: take top num_samples most negative positions
            sorted_indices = torch.argsort(last_bias_grad)
            label = sorted_indices[:num_samples]
            # Confidence based on negative gradient magnitude
            confidence = softmax(
                -last_bias_grad.unsqueeze(0).expand(num_samples, -1), dim=-1
            )

        return LabelInferenceResult(
            labels=label,
            confidence=confidence,
            method="idlg",
            metadata={"gradient": last_bias_grad.detach()},
        )


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
            display_name="Joint Label Optimization",
            description="Initialize soft labels for joint optimization with reconstruction",
            required_capabilities={"has_gradients": True},
        )

    def infer_labels(
        self,
        gradients: list[torch.Tensor],
        model: nn.Module,
        num_samples: int,
        true_labels: torch.Tensor | None = None,
    ) -> LabelInferenceResult:
        """Initialize soft labels for optimization."""
        device = gradients[0].device

        _ = true_labels  # Not used for initialization, but kept for interface consistency

        if self.init_strategy == "idlg_warm":
            # Warm-start using iDLG
            idlg = IDLGLabelInference()
            try:
                idlg_result = idlg.infer_labels(gradients, model, num_samples)
                # Convert hard labels to soft (one-hot)
                soft_labels = one_hot(idlg_result.labels, self.num_classes).float()
            except (ValueError, RuntimeError):
                # Fallback to uniform if iDLG fails
                soft_labels = torch.ones(num_samples, self.num_classes, device=device) / self.num_classes
        elif self.init_strategy == "uniform":
            soft_labels = torch.ones(num_samples, self.num_classes, device=device) / self.num_classes
        elif self.init_strategy == "random":
            soft_labels = torch.randn(num_samples, self.num_classes, device=device)
            soft_labels = softmax(soft_labels, dim=-1)
        else:
            soft_labels = torch.ones(num_samples, self.num_classes, device=device) / self.num_classes

        # Get hard labels for initial reconstruction
        hard_labels = torch.argmax(soft_labels, dim=-1)

        return LabelInferenceResult(
            labels=hard_labels,
            confidence=soft_labels,
            method="joint_optimization",
            metadata={"soft_labels": soft_labels, "requires_optimization": True},
        )


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
            display_name="Oracle Labels (Ground Truth)",
            description="Use ground-truth labels - not a real attack",
            required_capabilities={},  # No requirements - uses provided labels
        )

    def infer_labels(
        self,
        gradients: list[torch.Tensor],
        model: nn.Module,
        num_samples: int,
        true_labels: torch.Tensor | None = None,
    ) -> LabelInferenceResult:
        """Return provided ground-truth labels."""
        _ = (gradients, model, num_samples)  # Unused parameters
        if true_labels is None:
            raise ValueError(
                "OracleLabels requires 'true_labels' parameter"
            )

        if not isinstance(true_labels, torch.Tensor):
            true_labels = torch.tensor(true_labels, dtype=torch.long)

        # Create uniform confidence (we know labels are correct)
        # Infer num_classes from the labels themselves
        num_classes = int(true_labels.max().item()) + 1
        confidence = one_hot(true_labels, num_classes).float()

        return LabelInferenceResult(
            labels=true_labels,
            confidence=confidence,
            method="oracle",
            metadata={"ground_truth": True},
        )

__all__ = [
    "IDLGLabelInference",
    "JointLabelOptimization",
    "OracleLabels",
]
