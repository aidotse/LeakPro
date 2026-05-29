#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Mean adjustment strategies for diffusion-based gradient inversion.

These strategies control how gradient information from the gradient-matching
loss modifies the DDPM posterior mean ``mu_theta`` at each reverse-process
timestep, steering the denoising trajectory toward the true private data.

Classes
-------
* :class:`MeanAdjustmentStrategy` — abstract base
* :class:`SimilarityGuidance` — add ``-gamma * sigma_t * grad_{x_t} sim``
  to ``mu_theta`` (GGDM, Gu et al. 2024)
* :class:`AdaptiveMeanOptimization` — K inner Adam steps on a learnable
  ``mu*``, then blend with ``mu_theta`` via a schedule
  (GradInvDiff, Wang et al. 2024)

References
----------
* Gu et al., "Federated Learning Vulnerabilities: Privacy Attacks with
  Denoising Diffusion Probabilistic Models", WWW 2024.
* Wang et al., "GradInvDiff: Stealing Medical Privacy in Federated
  Learning via Diffusion-Based Gradient Inversion", 2024.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List

import torch

from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.blending_schedules import (
    BlendingSchedule,
    ConstantSchedule,
    CosineDecaySchedule,
    LinearDecaySchedule,
)
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.diffusion_utils import (
    denorm_ddpm_to_model_space as _denorm_ddpm_to_model_space,
)
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.optimizer_utils import (
    compute_loss_components,
    run_inner_adam_loop,
)
from pydantic import validate_call

from leakpro.attacks.gia_attacks.modular.config.registry import register

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.loss_components import (
        LossComponent,
    )
    from leakpro.attacks.gia_attacks.modular.core.state import RunContext


class MeanAdjustmentStrategy(ABC):
    """How gradient information modifies the DDPM posterior mean ``mu_theta``.

    Called once per reverse-process timestep.  Returns the adjusted mean,
    the direction ``delta_mu = mu_adjusted - mu_theta``, and a dict of
    logging info.

    The ``delta_mu`` vector is consumed by
    :class:`~leakpro.attacks.gia_attacks.modular.components.
    optimization_building_blocks.noise_strategies.GradientAlignedNoiseInjection`
    to project the noise onto the gradient-residual direction (GANI).
    """

    @abstractmethod
    def adjust(
        self,
        mu_theta: torch.Tensor,
        sigma_t: torch.Tensor | float,
        xt: torch.Tensor,
        x0_pred: torch.Tensor,
        scheduler: Any,
        t: int,
        T: int,
        ctx: "RunContext",
        labels: torch.Tensor,
        target_grads: List[torch.Tensor],
        loss_components: List[LossComponent],
        n_images: int = 1,
        n_seeds: int = 1,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Adjust the posterior mean using gradient information.

        Args:
            mu_theta: DDPM posterior mean ``mu_theta(x_t, t)``  [N*G, C, H, W].
            sigma_t: Posterior standard deviation sigma_t (scalar-like).
            xt: Current noisy image ``x_t``  [N*G, C, H, W].
            x0_pred: UNet's clean-image prediction ``x_hat_0`` [N*G, C, H, W].
            scheduler: DDPMSchedulerWrapper.
            t: Current timestep (counts down from T-1 to 0).
            T: Total number of timesteps.
            ctx: Run context (provides target_model, loss_fn, data_mean, data_std).
            labels: Inferred or oracle labels [N*G].
            target_grads: Target (observed) gradients.
            loss_components: Loss components that compute the guidance scalar.
            n_images: Number of distinct images N (batch size without seeds).
            n_seeds: Number of seeds per image G.

        Returns:
            ``(adjusted_mean, delta_mu, info_dict)``:

            * ``adjusted_mean`` — modified mean [N*G, C, H, W].
            * ``delta_mu`` — direction ``mu_adjusted - mu_theta`` [N*G, C, H, W].
            * ``info_dict`` — logging info, e.g. ``{"loss": 0.05}``.
        """
        ...


@register("mean.similarity_guidance")
class SimilarityGuidance(MeanAdjustmentStrategy):
    """GGDM-style guidance: shift ``mu_theta`` by the gradient of the similarity loss.

    Implements a corrected variant of Algorithm 1 from Gu et al. (WWW 2024).

    The original paper feeds noisy ``x_t`` directly into the FL model, but
    at high noise levels ``x_t`` is nearly pure Gaussian and produces
    uninformative gradients.  We instead compute gradient matching on the
    UNet's clean prediction ``x_hat_0`` and backpropagate to ``x_t`` space.

    The reparameterisation is::

        x_hat_0 = (x_t - sqrt(1 - alpha_bar_t) * eps_theta(x_t, t)) / sqrt(alpha_bar_t)

    The **exact** chain rule gives::

        grad_{x_t} sim  =  grad_{x_hat_0} sim  *  d(x_hat_0) / d(x_t)
                        =  grad_{x_hat_0} sim  *  (I - sqrt(1-alpha_bar_t) * J_eps)
                                                /  sqrt(alpha_bar_t)

    where ``J_eps = d eps_theta / d x_t`` is the UNet's input Jacobian.
    When ``backprop_through_unet=True`` (default), the UNet is re-run with
    ``x_t.requires_grad=True`` so that ``torch.autograd.grad`` automatically
    includes the ``J_eps`` term — yielding the exact gradient.

    When ``backprop_through_unet=False``, the fast approximation is used
    instead (treats ``eps_theta`` as constant w.r.t. ``x_t``)::

        grad_{x_t} sim  ≈  grad_{x_hat_0} sim  /  sqrt(alpha_bar_t)

    This produces a guidance term
    ``-gamma * sigma_t * grad_{x_t} sim``  with a cleaner signal than
    feeding noisy ``x_t`` into the model.

    Args:
        gamma: Guidance scale (default: 100).
        grad_clip: Clamp guidance gradients to ``[-clip, clip]``.
        use_x0_pred: If ``True`` (default), compute gradient matching on the
            UNet's x0 prediction.  Set to ``False`` to use the original
            paper's noisy-``x_t`` approach.
        backprop_through_unet: If ``True`` (default), re-run the UNet with
            ``requires_grad`` enabled on ``x_t`` to compute the exact
            gradient through the denoising network.  Set to ``False`` for the
            fast approximation that ignores the UNet Jacobian.
            Ignored when ``use_x0_pred=False``.
    """

    @validate_call
    def __init__(
        self,
        gamma: float = 100.0,
        grad_clip: float = 1.0,
        use_x0_pred: bool = True,
        backprop_through_unet: bool = True,
    ) -> None:
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.use_x0_pred = use_x0_pred
        self.backprop_through_unet = backprop_through_unet

    def adjust(
        self,
        mu_theta: torch.Tensor,
        sigma_t: torch.Tensor | float,
        xt: torch.Tensor,
        x0_pred: torch.Tensor,
        scheduler: Any,
        t: int,
        T: int,
        ctx: "RunContext",
        labels: torch.Tensor,
        target_grads: List[torch.Tensor],
        loss_components: List[LossComponent],
        n_images: int = 1,
        n_seeds: int = 1,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        C, H, W = xt.shape[-3], xt.shape[-2], xt.shape[-1]
        data_mean = ctx.client_observations.data_mean
        data_std = ctx.client_observations.data_std

        if self.use_x0_pred:
            lbl_6d = labels[:n_images].unsqueeze(0) if labels.ndim == 1 else labels[:, :n_images]

            if self.backprop_through_unet:
                # ---- Exact guidance: backprop through UNet ----------------
                unet = kwargs.get("unet")
                if unet is None:
                    raise ValueError(
                        "SimilarityGuidance with backprop_through_unet=True "
                        "requires 'unet' to be forwarded as a keyword argument "
                        "to adjust().  Pass unet=self.unet from the optimizer."
                    )
                xt_var = xt.detach().clone().requires_grad_(True)
                t_tensor = torch.full(
                    (xt_var.shape[0],), t, device=xt_var.device, dtype=torch.long
                )
                eps_var = unet(xt_var, t_tensor).sample
                x0_pred_var = scheduler.predict_x0_from_eps(xt_var, t, eps_var).clamp(-1, 1)
                x0_6d = x0_pred_var.reshape(1, n_images, n_seeds, C, H, W)
                x_model = _denorm_ddpm_to_model_space(x0_6d, data_mean, data_std)
                total_loss, component_losses = compute_loss_components(
                    loss_components, x_model, lbl_6d, target_grads, ctx
                )
                grad_xt = torch.autograd.grad(total_loss, xt_var)[0]
                grad_xt = grad_xt.clamp(-self.grad_clip, self.grad_clip)
                guidance = -self.gamma * (sigma_t ** 2) * grad_xt.detach()
            else:
                # ---- Fast approximation: treat eps_theta as constant ------
                x0_var = x0_pred.detach().clone().requires_grad_(True)
                x0_6d = x0_var.reshape(1, n_images, n_seeds, C, H, W)
                x_model = _denorm_ddpm_to_model_space(x0_6d, data_mean, data_std)
                total_loss, component_losses = compute_loss_components(
                    loss_components, x_model, lbl_6d, target_grads, ctx
                )
                grad_x0 = torch.autograd.grad(total_loss, x0_var)[0]
                alpha_bar = scheduler.alphas_cumprod[t].float()
                grad_xt = grad_x0 / alpha_bar.sqrt().clamp(min=1e-4)
                grad_xt = grad_xt.clamp(-self.grad_clip, self.grad_clip)
                guidance = -self.gamma * (sigma_t ** 2) * grad_xt.detach()
        else:
            # ---- Original GGDM: gradient matching on noisy x_t -----------
            xt_var = xt.detach().clone().requires_grad_(True)
            xt_6d = xt_var.reshape(1, n_images, n_seeds, C, H, W)
            x_model = _denorm_ddpm_to_model_space(xt_6d, data_mean, data_std)
            lbl_6d = labels[:n_images].unsqueeze(0) if labels.ndim == 1 else labels[:, :n_images]
            total_loss, component_losses = compute_loss_components(
                loss_components, x_model, lbl_6d, target_grads, ctx
            )
            grad_xt = torch.autograd.grad(total_loss, xt_var)[0]
            grad_xt = grad_xt.clamp(-self.grad_clip, self.grad_clip)
            guidance = -self.gamma * (sigma_t ** 2) * grad_xt.detach()

        adjusted_mean = mu_theta.detach() + guidance
        delta_mu = guidance
        info = {name: val.item() for name, val in component_losses.items()}
        info["total_loss"] = total_loss.item()
        return adjusted_mean, delta_mu, info

    def __repr__(self) -> str:
        return (
            f"SimilarityGuidance(gamma={self.gamma}, "
            f"grad_clip={self.grad_clip}, use_x0_pred={self.use_x0_pred}, "
            f"backprop_through_unet={self.backprop_through_unet})"
        )


@register("mean.adaptive_optimization")
class AdaptiveMeanOptimization(MeanAdjustmentStrategy):
    """GradInvDiff AMO: inner Adam loop on ``mu*``, then blend with ``mu_theta``.

    Implements Eq. 6 from Wang et al. (2024)::

        mu_bar = gamma_t * mu*  +  (1 - gamma_t) * mu_theta

    where ``mu*`` is obtained by ``K`` inner Adam steps minimising the
    gradient-matching loss, and ``gamma_t`` follows a
    :class:`~leakpro.attacks.gia_attacks.modular.components.
    optimization_building_blocks.blending_schedules.BlendingSchedule`
    (linear decay by default).

    The inner optimisation is performed in DDPM ``[-1, 1]`` space starting
    from the current noisy image ``x_t``.  Using ``x_t`` (rather than the
    UNet's ``x_hat_0``) as the initialisation was found empirically to
    produce better convergence.

    Args:
        inner_steps: Number of inner Adam steps ``K`` (paper: 5).
        inner_lr: Learning rate for inner Adam (paper: 0.01).
        schedule: :class:`BlendingSchedule` for ``gamma_t``
            (default: :class:`LinearDecaySchedule` 1→0, matching paper Eq. 6).
    """

    _SCHEDULE_MAP: dict[str, type] = {
        "linear": LinearDecaySchedule,
        "cosine": CosineDecaySchedule,
        "constant": ConstantSchedule,
    }

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        inner_steps: int = 5,
        inner_lr: float = 0.01,
        schedule: BlendingSchedule | str | None = None,
    ) -> None:
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        if isinstance(schedule, str):
            cls = self._SCHEDULE_MAP.get(schedule)
            if cls is None:
                raise ValueError(f"Unknown schedule {schedule!r}. Choose from {list(self._SCHEDULE_MAP)}.")
            self.schedule = cls()
        else:
            self.schedule = schedule or LinearDecaySchedule()

    def adjust(
        self,
        mu_theta: torch.Tensor,
        sigma_t: torch.Tensor | float,
        xt: torch.Tensor,
        x0_pred: torch.Tensor,
        scheduler: Any,
        t: int,
        T: int,
        ctx: "RunContext",
        labels: torch.Tensor,
        target_grads: List[torch.Tensor],
        loss_components: List[LossComponent],
        n_images: int = 1,
        n_seeds: int = 1,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        C, H, W = xt.shape[-3], xt.shape[-2], xt.shape[-1]
        data_mean = ctx.client_observations.data_mean
        data_std = ctx.client_observations.data_std

        def _loss_fn(mu_var: torch.Tensor) -> tuple[torch.Tensor, dict]:
            mu_6d = mu_var.reshape(1, n_images, n_seeds, C, H, W)
            x_candidate = _denorm_ddpm_to_model_space(mu_6d, data_mean, data_std)
            lbl_6d = labels[:n_images].unsqueeze(0) if labels.ndim == 1 else labels[:, :n_images]
            return compute_loss_components(
                loss_components, x_candidate, lbl_6d, target_grads, ctx
            )

        best_mu, best_loss, best_component_losses = run_inner_adam_loop(
            init_tensor=mu_theta,
            loss_fn=_loss_fn,
            steps=self.inner_steps,
            lr=self.inner_lr,
        )

        # -- Blend mu* with mu_theta using schedule -------------------------
        gamma_t = self.schedule(t, T)
        mu_theta_det = mu_theta.detach()
        adjusted_mean = gamma_t * best_mu + (1.0 - gamma_t) * mu_theta_det
        delta_mu = best_mu - mu_theta_det

        return adjusted_mean, delta_mu, {
            **best_component_losses,
            "total_loss": best_loss,
            "gamma_t": gamma_t,
        }

    def __repr__(self) -> str:
        return (
            f"AdaptiveMeanOptimization(inner_steps={self.inner_steps}, "
            f"inner_lr={self.inner_lr}, schedule={self.schedule})"
        )


__all__ = [
    "MeanAdjustmentStrategy",
    "SimilarityGuidance",
    "AdaptiveMeanOptimization",
]
