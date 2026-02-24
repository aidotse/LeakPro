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

    def __init__(self, use_gradient_sign: bool = False) -> None:
        """Initialize step strategy.

        Args:
            use_gradient_sign: If True, use sign(grad) instead of grad for updates

        """
        self.use_gradient_sign = use_gradient_sign

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

            # Apply sign to gradients if requested
            if self.use_gradient_sign:
                for param in state.optimizer.param_groups[0]["params"]:
                    if param.grad is not None:
                        param.grad.data = torch.sign(param.grad.data)
                # Also apply to label parameters if they exist
                if len(state.optimizer.param_groups) > 1:
                    for param in state.optimizer.param_groups[1]["params"]:
                        if param.grad is not None:
                            param.grad.data = torch.sign(param.grad.data)

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
