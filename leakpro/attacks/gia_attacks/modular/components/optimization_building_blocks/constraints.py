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
"""Constraint strategies for optimization."""

from abc import abstractmethod

import torch

from leakpro.attacks.gia_attacks.modular.core.component_base import Component, ComponentMetadata


class ConstraintStrategy(Component):
    """Base class for constraint strategies."""

    @abstractmethod
    def apply(self, reconstruction: torch.Tensor, data_mean: torch.Tensor, data_std: torch.Tensor) -> torch.Tensor:
        """Apply constraint to reconstruction.

        Args:
            reconstruction: Current reconstruction
            data_mean: Mean used for normalization
            data_std: Std used for normalization

        Returns:
            Constrained reconstruction

        """
        pass

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Return metadata for this constraint strategy.

        By default, constraint strategies have no special requirements.
        """
        return ComponentMetadata(
            name=cls.__name__,
            display_name=cls.__name__,
            description="Constraint strategy for optimization",
            required_capabilities={},
        )


class NoConstraint(ConstraintStrategy):
    """No constraints - pass through."""

    def apply(self, reconstruction: torch.Tensor, data_mean: torch.Tensor, data_std: torch.Tensor) -> torch.Tensor:
        """Return reconstruction unchanged."""
        _ = (data_mean, data_std)  # Unused parameters, but kept for interface consistency
        return reconstruction


class ClipConstraint(ConstraintStrategy):
    """Clip pixel values to valid range."""

    def apply(self, reconstruction: torch.Tensor, data_mean: torch.Tensor, data_std: torch.Tensor) -> torch.Tensor:
        """Clip values to valid range based on data mean and std."""
        min_val = (0 - data_mean) / data_std
        max_val = (1 - data_mean) / data_std
        return torch.clamp(reconstruction, min=min_val, max=max_val)


__all__ = ["ConstraintStrategy", "NoConstraint", "ClipConstraint"]
