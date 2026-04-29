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
"""Step execution strategies for optimization.

Different optimizers require different step execution patterns:
- Standard: Simple zero_grad, backward, step
- Allows for extensions that use different stepping
"""

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Tuple

import torch

from leakpro.attacks.gia_attacks.modular.core.component_base import Component, ComponentMetadata

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.components.composable_optimizer import InternalOptimizerState


class StepStrategy(Component):
    """Base class for optimization step strategies."""

    @abstractmethod
    def execute_step(
        self,
        state: "InternalOptimizerState",
        compute_loss_fn: Callable[[], Tuple[torch.Tensor, dict[str, float]]],
        apply_constraints_fn: Callable[[Any], None],
    ) -> Tuple[float, dict[str, float]]:
        """Execute one optimization step.

        Args:
            state: Internal optimizer state (contains optimizer, reconstruction, etc.)
            compute_loss_fn: Function that computes (total_loss, losses)
            apply_constraints_fn: Function that applies constraints to state

        Returns:
            Tuple of (total_loss_value, losses_dict)

        """
        pass
    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Return metadata for this step strategy.

        By default, step strategies have no special requirements.
        """
        return ComponentMetadata(
            name=cls.__name__,
            display_name=cls.__name__,
            description="Optimization step strategy",
            required_capabilities={},
        )

class StandardStepStrategy(StepStrategy):
    """Standard step execution with closure."""

    def __init__(self, use_gradient_sign: bool = False, gradient_noise_std: float = 0.0) -> None:
        """Initialize step strategy.

        Args:
            use_gradient_sign: If True, use sign(grad) instead of grad for updates
            gradient_noise_std: Standard deviation of Gaussian noise to add to gradients (0.0 = no noise)

        """
        self.use_gradient_sign = use_gradient_sign
        self.gradient_noise_std = gradient_noise_std

    def _add_gradient_noise(self, state: "InternalOptimizerState") -> None:
        """Add Gaussian noise to gradients."""
        if self.gradient_noise_std <= 0:
            return

        for param_group in state.optimizer.param_groups:
            for param in param_group["params"]:
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * self.gradient_noise_std
                    param.grad.data.add_(noise)

    def _apply_gradient_sign(self, state: "InternalOptimizerState") -> None:
        """Apply sign operation to gradients."""
        if not self.use_gradient_sign:
            return

        for param_group in state.optimizer.param_groups:
            for param in param_group["params"]:
                if param.grad is not None:
                    param.grad.data = torch.sign(param.grad.data)

    def execute_step(
        self,
        state: "InternalOptimizerState",
        compute_loss_fn: Callable[[], Tuple[torch.Tensor, dict[str, torch.Tensor]]],
        apply_constraints_fn: Callable[[Any], None],
    ) -> Tuple[float, dict[str, torch.Tensor]]:
        """Execute standard optimization step."""
        captured_losses = {"total_loss": 0.0, "losses": {}}

        def closure() -> dict[str, float]:
            """Closure for optimization step."""
            state.optimizer.zero_grad()
            total_loss, losses = compute_loss_fn()
            total_loss.backward()

            # Add noise and apply sign to gradients if requested
            self._add_gradient_noise(state)
            self._apply_gradient_sign(state)

            captured_losses["total_loss"] = total_loss
            captured_losses["losses"] = losses
            return total_loss

        state.optimizer.step(closure)
        apply_constraints_fn(state)

        if state.scheduler is not None:
            state.scheduler.step()

        return (
            captured_losses["total_loss"].item(),
            {
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in captured_losses["losses"].items()
            }
        )

__all__ = [
    "StepStrategy",
    "StandardStepStrategy",
]
