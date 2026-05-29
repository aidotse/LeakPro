#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Constraint strategies for optimization."""

from abc import abstractmethod
from typing import TYPE_CHECKING

import torch

from leakpro.attacks.gia_attacks.modular.config.registry import register
from leakpro.attacks.gia_attacks.modular.core.component_base import Component, ComponentMetadata

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.state import RunContext


def _project_onto_l1_ball(x: torch.Tensor, radius: float) -> torch.Tensor:
    """Project each row of *x* onto the L1 ball of the given *radius*.

    Implements the algorithm from Duchi et al. (2008).
    Each sample (first dimension) is projected independently.

    Args:
        x: Tensor ``[batch, n]`` – each row is a vector to project.
        radius: L1 ball radius ``r > 0``.

    Returns:
        Projected tensor, same shape as *x*.
    """
    # Fast path: already inside the ball
    norms = torch.norm(x, p=1, dim=1)  # [batch]
    mask_in = norms <= radius  # [batch]

    if mask_in.all():
        return x

    abs_x = x.abs()
    # Sort descending along feature dimension
    mu, _ = torch.sort(abs_x, dim=1, descending=True)
    cumsum = mu.cumsum(dim=1)  # [batch, n]
    n = x.shape[1]
    arange = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)  # [n]
    # rho[i] = largest index where mu[i] - (cumsum[i] - radius) / index > 0
    condition = mu - (cumsum - radius) / arange > 0  # [batch, n]
    rho = condition.long().sum(dim=1, keepdim=True).clamp(min=1)  # [batch, 1]
    # theta = (cumsum[rho-1] - radius) / rho
    rho_idx = (rho - 1).expand_as(cumsum[:, :1])  # [batch, 1]
    cumsum_rho = cumsum.gather(1, rho_idx)  # [batch, 1]
    theta = (cumsum_rho - radius) / rho.float()  # [batch, 1]
    proj = torch.clamp(abs_x - theta, min=0) * x.sign()

    # Keep original where already inside the ball
    proj = torch.where(mask_in.unsqueeze(1), x, proj)
    return proj


class ConstraintStrategy(Component):
    """Base class for constraint strategies."""

    @abstractmethod
    def apply(self, reconstruction: torch.Tensor, ctx: "RunContext") -> torch.Tensor:
        """Apply constraint to reconstruction.

        Args:
            reconstruction: Current reconstruction
            ctx: Run context (provides data_mean/data_std via client_observations)

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
            required_capabilities={},
        )


@register("constraint.clip")
class ClipConstraint(ConstraintStrategy):
    """Clip pixel values to valid range."""

    def apply(self, reconstruction: torch.Tensor, ctx: "RunContext") -> torch.Tensor:
        """Clip values to valid range based on data mean and std."""
        data_mean = ctx.client_observations.data_mean
        data_std = ctx.client_observations.data_std
        min_val = (0 - data_mean) / data_std
        max_val = (1 - data_mean) / data_std
        return torch.clamp(reconstruction, min=min_val, max=max_val)


class FeatureSpaceConstraint(ConstraintStrategy):
    """Marker base class for constraints that operate on the *optimisable tensor*.

    Ordinary constraints (e.g. :class:`ClipConstraint`) are applied to the
    **pixel-space reconstruction** and are therefore skipped when a
    representation strategy is active.  Subclasses of
    ``FeatureSpaceConstraint`` signal that they should be applied directly
    to the optimisable tensor (latent codes, intermediate features, …)
    regardless of whether a representation is present.

    Subclasses must still implement :meth:`apply`.  The signature is the same
    as for :class:`ConstraintStrategy` but *data_mean* / *data_std* may be
    ignored when they are irrelevant (e.g. for feature-space projections).
    """

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Return metadata."""
        return ComponentMetadata(
            name=cls.__name__,
            required_capabilities={},
        )


@register("constraint.l1_ball")
class L1BallProjectionConstraint(FeatureSpaceConstraint):
    """Project intermediate features onto an L1 ball around their initial value.

    This implements the per-stage L1-ball constraint from **GIFD** (Fang et al.
    NeurIPS 2023, Algorithm 1, line 9):

    .. math::

        h_k \\leftarrow h^0_k + \\Pi_{\\|\\cdot\\|_1 \\le r_k}(h_k - h^0_k)

    where :math:`h^0_k` is the initial value of the feature tensor at the
    start of stage *k* (set once via :meth:`set_initial_point`), and
    :math:`r_k` is the radius.

    The projection is applied *per-sample* in the batch (each image's
    flattened feature vector is projected independently).

    Args:
        radius: L1-ball radius ``r > 0``.  A radius of 0 disables the
                projection (pass-through).

    Usage::

        constraint = L1BallProjectionConstraint(radius=5.0)
        # In composable_optimizer._setup_optimization():
        constraint.set_initial_point(initial_features)
        # In the step loop, apply_constraints_fn calls:
        features = constraint.apply(features, data_mean, data_std)
    """

    def __init__(self, radius: float = 5.0) -> None:
        """Initialise with projection radius."""
        self.radius = radius
        self._initial_point: torch.Tensor | None = None

    def set_initial_point(self, initial: torch.Tensor) -> None:
        """Store the initial feature tensor (called at the start of each stage).

        Args:
            initial: Initial optimisable tensor
                     ``[E, N, G, C, H, W]`` (or any shape).
        """
        self._initial_point = initial.detach().clone()

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Return metadata."""
        return ComponentMetadata(
            name="l1_ball_projection",
            required_capabilities={},
        )

    def apply(
        self,
        reconstruction: torch.Tensor,
        ctx: "RunContext",
    ) -> torch.Tensor:
        """Project *reconstruction* (= optimisable features) onto the L1 ball.

        *ctx* is accepted for API compatibility but not used — this constraint
        operates purely in feature space.

        Args:
            reconstruction: Current optimisable features (any shape).
            ctx: Ignored.

        Returns:
            Projected features, same shape as *reconstruction*.
        """
        if self._initial_point is None or self.radius <= 0:
            return reconstruction

        h0 = self._initial_point.to(reconstruction.device)
        deviation = reconstruction - h0  # Δh = h_k − h^0_k

        orig_shape = deviation.shape

        # Treat the first 3 dims (E, N, G) as the batch dimension when the
        # tensor is ≥ 4-D (e.g. [E, N, G, C, H, W] or [E, N, G, latent_dim]).
        # Each (E, N, G) combination is projected independently.
        batch_size = orig_shape[0] * orig_shape[1] * orig_shape[2] if len(orig_shape) >= 4 else orig_shape[0]

        dev_flat = deviation.reshape(batch_size, -1)
        proj_flat = _project_onto_l1_ball(dev_flat, self.radius)
        proj_dev = proj_flat.reshape(orig_shape)
        return h0 + proj_dev


@register("constraint.spherical")
class SphericalConstraint(FeatureSpaceConstraint):
    """Spherical constraint — element-wise magnitude preservation.

    Exactly replicates the GIFD paper's ``SphericalOptimizer`` (Fang et al., ICCV 2023).
    Applied AFTER each optimization step.

    This is a :class:`FeatureSpaceConstraint` because it operates on the
    **optimisable latent tensor** (z), not on the pixel-space reconstruction.
    Without this inheritance the constraint would be silently skipped when a
    GAN representation is active (stage 0 of GIFD).

    **Paper behaviour (inversefed/reconstruction_algorithms.py, ~line 100):**

    For a BigGAN latent ``z`` of shape ``[N, 128]`` (ndim=2), the paper uses::

        radii = {param: (param.pow(2).sum(tuple(range(2, param.ndim)), keepdim=True) + 1e-9).sqrt()
                 for param in params}

        def step():
            for param in params:
                param.data.div_((param.pow(2).sum(tuple(range(2, param.ndim)), keepdim=True) + 1e-9).sqrt())
                param.mul_(radii[param])

    When ``ndim == 2``, ``range(2, 2)`` is empty, so ``tensor.sum((), keepdim=True)``
    returns the tensor **unchanged** (no reduction).  The result is an
    **element-wise** constraint: ``radii[i, j] ≈ |z_init[i, j]|``, and each
    step resets every scalar to ``sign(z[i,j]) * |z_init[i,j]|``.

    **LeakPro adaptation:**

    Our framework stores ``z`` as ``[E, N, G, latent_dim]`` (ndim=4).  We apply
    the same element-wise logic: ``radii`` and ``current_norm`` are computed
    *per scalar element* (no dimensional reduction), so every element is
    projected back to its initial absolute value.

    Note:
        Applied only to latent codes (z) during GIFD stage 0.  Not used for
        intermediate features.

    Reference:
        Fang et al., GIFD (ICCV 2023) — inversefed/reconstruction_algorithms.py
        ``SphericalOptimizer`` class.
    """

    def __init__(self) -> None:
        """Initialise spherical constraint."""
        self._initial_radii: torch.Tensor | None = None

    def set_initial_point(self, initial: torch.Tensor) -> None:
        """Store element-wise radii from the initial tensor.

        Computes ``sqrt(z_init_i^2 + eps)`` element-wise, matching the paper's
        ``SphericalOptimizer.__init__`` for a latent tensor of shape ``[N, latent_dim]``.

        The tensor is reshaped to ``[batch, latent_dim]`` before computation so that
        the CUDA kernel operates on the same 2-D layout as the paper (where the latent
        code is ``[N=1, latent_dim=128]``).  This eliminates the ~1-ULP FP difference
        that arises when operating on ``[E, N, G, latent_dim]`` (4-D) vs ``[N, latent_dim]``.

        Args:
            initial: Initial optimisable tensor (latent codes), shape ``[E, N, G, latent_dim]``.

        """
        with torch.no_grad():
            # Flatten all batch dims (E, N, G) into one so CUDA sees [batch, latent_dim].
            # This matches the paper's z=[N, latent_dim] convention exactly.
            z_2d = initial.reshape(-1, initial.shape[-1])  # [E*N*G, latent_dim]
            self._initial_radii = (z_2d.pow(2).sum((), keepdim=True) + 1e-9).sqrt()
            self._original_shape = initial.shape

    @classmethod
    def get_metadata(cls) -> ComponentMetadata:
        """Return metadata."""
        return ComponentMetadata(
            name="spherical_constraint",
            required_capabilities={},
        )

    def apply(
        self,
        reconstruction: torch.Tensor,
        ctx: "RunContext",
    ) -> torch.Tensor:
        """Apply spherical constraint: normalize each element and scale by initial radius.

        Replicates the paper's step function::

            param.data.div_(current_norm)
            param.mul_(initial_radii)

        where both ``current_norm`` and ``initial_radii`` are computed element-wise.

        Args:
            reconstruction: Current latent tensor, any shape.
            ctx: Ignored (feature-space constraint).

        Returns:
            Constrained latent tensor with element-wise radii preserved.

        """
        _ = ctx  # Unused, kept for interface consistency

        if self._initial_radii is None:
            return reconstruction

        with torch.no_grad():
            original_shape = reconstruction.shape
            z_2d = reconstruction.reshape(-1, original_shape[-1])
            current_norm = (z_2d.pow(2).sum((), keepdim=True) + 1e-9).sqrt()
            z_2d.div_(current_norm)
            z_2d.mul_(self._initial_radii.to(reconstruction.device))
            return reconstruction



__all__ = [
    "ConstraintStrategy",
    "ClipConstraint",
    "FeatureSpaceConstraint",
    "L1BallProjectionConstraint",
    "SphericalConstraint",
]
