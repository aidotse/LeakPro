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
