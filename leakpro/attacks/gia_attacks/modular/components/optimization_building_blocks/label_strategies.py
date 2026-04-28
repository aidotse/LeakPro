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
"""Label handling strategies for optimization."""

from abc import abstractmethod
from typing import List, Tuple

import torch

from leakpro.attacks.gia_attacks.modular.core.component_base import Component, ComponentMetadata, LabelInferenceResult


class LabelStrategy(Component):
    """Base class for label handling strategies."""

    @abstractmethod
    def setup(
        self,
        initial_labels: LabelInferenceResult,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Setup labels for optimization.

        Returns:
            Tuple of (effective_labels, optimizable_params)

        """
        pass

    @abstractmethod
    def get_labels_for_forward(
        self,
        optimizable_params: List[torch.Tensor],
        fixed_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Get labels for current forward pass."""
        pass

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Return metadata for this label strategy.

        By default, label strategies have no special requirements.
        """
        return ComponentMetadata(
            name=cls.__name__,
            display_name=cls.__name__,
            description="Label handling strategy",
            required_capabilities={},
        )


class FixedLabels(LabelStrategy):
    """Use fixed pre-inferred labels (iDLG, inverting gradients)."""

    def setup(
        self,
        initial_labels: LabelInferenceResult,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Use initial labels as-is."""
        return initial_labels.labels, []

    def get_labels_for_forward(
        self,
        _optimizable_params: List[torch.Tensor],
        fixed_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Return fixed labels."""
        return fixed_labels


class JointLabelOptimizationStrategy(LabelStrategy):
    """Jointly optimize labels with reconstruction (DLG)."""

    def __init__(self, temperature: float = 1.0) -> None:
        """Initialize joint label optimization."""
        self.temperature = temperature

    def setup(
        self,
        initial_labels: LabelInferenceResult,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Setup soft labels for optimization.

        Args:
            initial_labels: Can be either:
                - Hard labels: Will be converted to soft labels
                - Soft labels: Will be used directly

        Returns:
            Tuple of (soft_labels, [soft_labels]) where soft_labels are optimizable

        """
        soft_labels = initial_labels.confidence.clone().detach()
        soft_labels.requires_grad = True

        return soft_labels, [soft_labels]

    def get_labels_for_forward(
        self,
        optimizable_params: List[torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Return soft labels as probabilities.

        Converts the optimizable label logits to probabilities via softmax.
        """
        if optimizable_params:
            soft_labels = optimizable_params[0]
            return torch.softmax(soft_labels, dim=1)
        return labels


__all__ = ["LabelStrategy", "FixedLabels", "JointLabelOptimizationStrategy"]
