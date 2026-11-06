#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Image-space regularizers and cross-run priors (TV, BN stats, multi-seed/epoch)."""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch
from torch import Tensor

from leakpro.attacks.gia_attacks.modular.core.interfaces import SPACES, LossTerm, Space
from leakpro.attacks.gia_attacks.modular.core.loss_inputs import LossInputs, LossScope
from leakpro.attacks.gia_attacks.modular.core.requirements import Requirement

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.state import Context


BNStrategy = Literal["inferred", "running", "proxy"]


def _check_space(space: str) -> Space:
    if space not in SPACES:
        raise ValueError(f"space must be 'data' or 'model_input', got {space!r}")
    return space


def _prior_tensor(inputs: LossInputs, space: Space, term: str) -> Tensor:
    """The tensor an image prior is evaluated on: the data-space ``recon`` or the model input."""
    if space == "data":
        return inputs.recon
    if inputs.model_input is None:
        raise ValueError(f"{term}(space='model_input') requires LossInputs.model_input, which the caller did not provide")
    return inputs.model_input


class TVLoss(LossTerm):
    """Total-variation regularization, on the data-space image or on the model input (``space``).

    Uses ellipsis indexing to operate strictly on the last two spatial dimensions
    (H, W), so it is correct for any number of prefix axes (batch, epoch, etc.).
    """

    def __init__(self, *, space: Space = "data") -> None:
        self.space = _check_space(space)

    def compute(self, inputs: LossInputs) -> Tensor:
        """Mean absolute difference between neighbouring pixels along width and height."""
        recon = _prior_tensor(inputs, self.space, "TVLoss")
        dx = torch.abs(recon[..., :-1] - recon[..., 1:])
        dy = torch.abs(recon[..., :-1, :] - recon[..., 1:, :])
        return dx.mean() + dy.mean()


class BNStatsLoss(LossTerm):
    """Batch-normalization statistics regularization via forward hooks.

    Supports three strategies:
      - "inferred" (default): Inverts the running-statistics (EMA) update of the client's single local step to
        recover the true batch statistics. Needs a protocol with exactly one local step (FedSGD, or FedAvg
        with one epoch on the whole batch); ``bind`` raises otherwise, since statistics accumulated over
        several steps at changing weights cannot be inverted from one batch's forward.
      - "running": Matches post-step running statistics directly (Huang et al.).
      - "proxy": Estimates layer statistics using a surrogate dataset.
    """

    def __init__(
        self,
        strategy: BNStrategy = "inferred",
        momentum: float | None = None,
        proxy_epochs: int = 1,
        proxy_lr: float = 1e-2,
    ) -> None:
        """Configure the strategy and its parameters.

        Args:
            strategy: See class docstring.
            momentum: EMA momentum used by the "inferred" strategy to invert
                each layer's running-stats update. ``None`` (default) reads
                each norm layer's own ``.momentum`` attribute off the model —
                correct even when different layers were constructed with
                different momenta (e.g. a model mixing custom-momentum and
                default-momentum BatchNorm2d blocks). Pass a float to force
                the same value for every layer instead.
            proxy_epochs: Only used by the "proxy" strategy — number of
                epochs to run over ``ctx.proxy_dataloader`` while collecting
                BN statistics.
            proxy_lr: Only used by the "proxy" strategy — learning rate for
                the meta-SGD step taken during proxy statistics collection.

        """
        if strategy not in ("inferred", "running", "proxy"):
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from: 'inferred', 'running', 'proxy'.")

        self.strategy = strategy
        self.momentum = momentum
        self.proxy_epochs = proxy_epochs
        self.proxy_lr = proxy_lr
        self.hooks: list = []

    def __repr__(self) -> str:
        """Names the strategy, which decides what the term needs from the run."""
        return f"BNStatsLoss(strategy={self.strategy!r})"

    def requirements(self) -> frozenset[Requirement]:
        """The proxy data for ``"proxy"``; the statistics before and after training and the batch size for ``"inferred"``."""
        if self.strategy == "proxy":
            return frozenset({"proxy_dataloader"})
        if self.strategy == "inferred":
            return frozenset({"bn_stats", "pre_bn_stats", "batch_size"})
        return frozenset({"bn_stats"})

    def bind(self, ctx: Context) -> BNStatsLoss:
        """A copy holding forward hooks on ``ctx.session.execution_model``; this term keeps none."""
        bound = copy.copy(self)
        bound.hooks = []
        bound._attach(ctx)
        return bound

    def _attach(self, ctx: Context) -> None:
        target_model = ctx.session.execution_model

        if self.strategy == "proxy":
            proxy_stats = self._collect_proxy_stats(target_model, ctx.proxy_dataloader, ctx.device)
            self._register_hooks(self._norm_modules(target_model), proxy_stats)
            self._freeze_bn_momentum(target_model)
            return

        observation = ctx.observation
        if observation.bn_stats is None:
            raise ValueError(f"BNStatsLoss(strategy={self.strategy!r}) requires observation.bn_stats")
        modules = self._bn_modules(target_model)
        if len(modules) != len(observation.bn_stats):
            raise ValueError(
                f"BNStatsLoss: the observation has statistics for {len(observation.bn_stats)} BatchNorm layers "
                f"but the model has {len(modules)}."
            )

        if self.strategy == "running":
            self._register_hooks(modules, list(observation.bn_stats))
            return

        pre_stats, num_images = observation.pre_bn_stats, observation.batch_size
        if pre_stats is None or num_images is None:
            raise ValueError("BNStatsLoss(strategy='inferred') requires observation.pre_bn_stats and observation.batch_size")
        if len(pre_stats) != len(modules):
            raise ValueError(
                f"BNStatsLoss: the observation has pre-training statistics for {len(pre_stats)} BatchNorm layers "
                f"but the model has {len(modules)}."
            )
        self._check_one_train_update(ctx, num_images)
        target_stats = []
        for module, (post_m, post_v), (pre_m, pre_v) in zip(modules, observation.bn_stats, pre_stats):
            m = self.momentum if self.momentum is not None else module.momentum
            if m is None:
                raise ValueError(
                    "BNStatsLoss(strategy='inferred') cannot invert a BatchNorm layer with momentum=None "
                    "(cumulative averaging); pass momentum=... explicitly."
                )
            # Reverse EMA momentum: mu_batch = (mu_post - (1-m)*mu_pre) / m. The variance recovered this way is the
            # unbiased one PyTorch feeds into running_var; the hook rescales it by (n-1)/n with the per-channel
            # element count n of the layer input (N, N*L or N*H*W), which is known at the forward.
            target_m = (post_m - (1.0 - m) * pre_m) / m
            target_v = (post_v - (1.0 - m) * pre_v) / m
            target_stats.append((target_m, target_v))

        self._register_hooks(modules, target_stats, unbiased_target_var=True)

    @staticmethod
    def _check_one_train_update(ctx: Context, num_images: int) -> None:
        """Raise unless the client updated its running statistics exactly once, which is what 'inferred' inverts."""
        steps = ctx.task.protocol.local_steps(num_images)
        if steps != 1:
            raise ValueError(
                f"BNStatsLoss(strategy='inferred') inverts one running-statistics update, but "
                f"{type(ctx.task.protocol).__name__} takes {steps if steps is not None else 'an unknown number of'} "
                f"local step(s) on {num_images} samples: the observed statistics mix batches seen at different "
                "weights and cannot be inverted. Use FedSGD or FedAvg with one local step on the whole batch "
                "(local_epochs=1, batch_size=None), or strategy='running'."
            )
        if ctx.task.model_mode != "train":
            raise ValueError(
                f"BNStatsLoss(strategy='inferred') needs a client in train mode, but model_mode={ctx.task.model_mode!r}: "
                "an eval-mode client never updates its running statistics, so there is no update to invert."
            )

    def release(self) -> None:
        """Close all registered forward hooks and clear the hook list."""
        for hook in self.hooks:
            hook.close()
        self.hooks.clear()

    def compute(self, inputs: LossInputs) -> Tensor:
        """Sum of the per-layer statistics distances collected by the hooks during the last forward pass."""
        if not self.hooks:
            raise ValueError("BNStatsLoss has no normalization layers to regularize: the model has no BatchNorm layers.")

        total = torch.tensor(0.0, device=inputs.recon.device)
        for hook in self.hooks:
            if hook.r_feature is not None:
                total = total + hook.r_feature
                hook.r_feature = None  # Reset to prevent stale values across seed slices
        return total

    @staticmethod
    def _bn_modules(model: torch.nn.Module) -> list:
        """The layers ``bn_running_stats`` reports statistics for, in the same order."""
        return [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and m.running_mean is not None]

    @staticmethod
    def _norm_modules(model: torch.nn.Module) -> list:
        """The layers the proxy strategy collects statistics for."""
        from torchvision.models.convnext import LayerNorm2d
        return [
            m for m in model.modules()
            if isinstance(
                m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.LayerNorm, LayerNorm2d, torch.nn.InstanceNorm2d)
            )
        ]

    def _register_hooks(
        self, modules: list, target_stats: list[tuple[Tensor, Tensor]], *, unbiased_target_var: bool = False,
    ) -> None:
        from torchvision.models.convnext import LayerNorm2d

        from leakpro.fl_utils.model_utils import (
            InferredBNFeatureHook,
            InferredIN2dFeatureHook,
            InferredLN2dFeatureHook,
            InferredLNFeatureHook,
        )

        for module, (target_m, target_v) in zip(modules, target_stats):
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                self.hooks.append(InferredBNFeatureHook(module, target_m, target_v, unbiased_target_var=unbiased_target_var))
            elif isinstance(module, torch.nn.LayerNorm):
                self.hooks.append(InferredLNFeatureHook(module, target_m, target_v))
            elif isinstance(module, LayerNorm2d):
                self.hooks.append(InferredLN2dFeatureHook(module, target_m, target_v))
            elif isinstance(module, torch.nn.InstanceNorm2d):
                self.hooks.append(InferredIN2dFeatureHook(module, target_m, target_v))

    def _collect_proxy_stats(self, model: torch.nn.Module, dataloader: Any, device: torch.device) -> list[tuple[Tensor, Tensor]]:  # noqa: ANN401 -- genuinely heterogeneous value
        """The normalization statistics a client would see over the proxy data while its weights move by SGD.

        A deep copy of ``model`` trains for ``proxy_epochs`` epochs of ``dataloader`` (plain SGD, cross-entropy);
        forward hooks on every layer of ``_norm_modules`` record that layer's batch statistics on each forward
        pass. The result holds one ``(mean, var)`` per layer, in module order, averaged over every recorded batch.
        The run's own model is never modified.
        """
        work = copy.deepcopy(model).to(device).train()
        layers = self._norm_modules(work)
        recorded: list[list[tuple[Tensor, Tensor]]] = [[] for _ in layers]

        def make_hook(slot: list[tuple[Tensor, Tensor]]):  # noqa: ANN202 -- returns the hook closure
            def hook_fn(mod: torch.nn.Module, inp: tuple, out: Tensor) -> None:  # noqa: ARG001 -- forward-hook signature
                x = inp[0]
                if isinstance(mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    dims = (0, 2, 3) if x.ndim == 4 else 0
                    m, v = x.mean(dim=dims), x.var(dim=dims, unbiased=False)
                else:
                    m, v = x.mean(), x.var(unbiased=False)
                slot.append((m.detach(), v.detach()))
            return hook_fn

        handles = [layer.register_forward_hook(make_hook(slot)) for layer, slot in zip(layers, recorded)]
        try:
            optimizer = torch.optim.SGD(work.parameters(), lr=self.proxy_lr)
            for _ in range(self.proxy_epochs):
                for inputs, labels in dataloader:
                    optimizer.zero_grad()
                    loss = torch.nn.functional.cross_entropy(work(inputs.to(device)), labels.to(device))
                    loss.backward()
                    optimizer.step()
        finally:
            for h in handles:
                h.remove()

        names = {id(m): n for n, m in work.named_modules()}
        stats = []
        for layer, entries in zip(layers, recorded):
            if not entries:
                raise ValueError(
                    f"BNStatsLoss(strategy='proxy'): normalization layer {names[id(layer)]} was never reached by the proxy data"
                )
            stats.append((torch.stack([m for m, _ in entries]).mean(0), torch.stack([v for _, v in entries]).mean(0)))
        return stats

    @staticmethod
    def _freeze_bn_momentum(model: torch.nn.Module) -> None:
        """Stop BatchNorm1d/2d running-stat EMA updates once proxy statistics are locked in.

        Without this, every later forward pass on the evolving reconstruction would keep nudging the
        running buffers away from the captured target statistics.
        """
        for m in model.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                m.momentum = 0


class GroupConsistencyLoss(LossTerm):
    """Pulls each seed reconstruction toward the group consensus with true cross-seed gradients.

    Unlike a per-slice approximation with a detached consensus, this loss operates on the
    FULL live reconstruction tensor in a single GROUP-scope pass.  Because the consensus
    (mean over the seed axis) is NOT detached, backpropagation flows through it into every
    seed's leaf parameters simultaneously.

    Formula (See Through Gradients, Yin et al. CVPR 2021, Eq. 11):
        L_group = (1/G) * Σ_g ||x̂_g - mean_g(x̂)||²
    which equals ``(recon - recon.mean(seed_dim, keepdim=True)).pow(2).mean()``
    because mean-of-means equals the global mean for equal-size groups.

    Returns 0 when no seed axis is registered or seed_size == 1. ``space`` selects the data-space
    image or the model input.
    """

    scope: ClassVar[LossScope] = LossScope.GROUP

    def __init__(self, *, space: Space = "data") -> None:
        self.space = _check_space(space)

    def compute(self, inputs: LossInputs) -> Tensor:
        """Mean squared deviation of each seed's reconstruction from the cross-seed mean."""
        # full live tensor [*axes, N, C, H, W] — NOT detached
        recon = _prior_tensor(inputs, self.space, "GroupConsistencyLoss")
        seed_dim = inputs.ctx.axes.get_dim("seed")
        if seed_dim is None or recon.shape[seed_dim] == 1:
            return recon.new_zeros(())
        consensus = recon.mean(dim=seed_dim, keepdim=True)  # live — gradients flow through
        return (recon - consensus).pow(2).mean()


class L2ImageNormLoss(LossTerm):
    """Spatially-normalized L2 norm of the reconstruction.

    Penalizes large pixel magnitudes independently of image resolution by dividing by the number
    of spatial pixels (H × W). ``reduction="batch"`` takes the norm of the whole tensor;
    ``"per_image_sum"`` sums the norms of the individual [C, H, W] images over every leading index
    (the See Through Gradients formulation). ``space`` selects the data-space image or the model input.
    """

    def __init__(self, *, space: Space = "data", reduction: Literal["batch", "per_image_sum"] = "batch") -> None:
        if reduction not in ("batch", "per_image_sum"):
            raise ValueError(f"reduction must be 'batch' or 'per_image_sum', got {reduction!r}")
        self.space = _check_space(space)
        self.reduction = reduction

    def compute(self, inputs: LossInputs) -> Tensor:
        """L2 norm of the reconstruction divided by the number of spatial pixels."""
        recon = _prior_tensor(inputs, self.space, "L2ImageNormLoss")
        spatial_area = recon.shape[-2] * recon.shape[-1]
        if self.reduction == "per_image_sum":
            return recon.reshape(-1, recon.shape[-3:].numel()).norm(p=2, dim=1).sum() / spatial_area
        return torch.norm(recon, p=2) / spatial_area


class EpochOrderInvariantPrior(LossTerm):
    """Encourages epoch reconstructions to be order-invariant (Dimitrov et al.).

    Aggregates each epoch's batch into a single representative via an invariant
    function (mean/sum/var), then penalizes pairwise differences between epochs.
    Larger distances push the optimizer toward epoch-consistent reconstructions.

    This loss is PER_SLICE: the driver removes the axis it slices along (``LossInputs.slice_axis``,
    usually the seed axis) before calling it, and the epoch dimension of the slice is located from
    ``ctx.axes`` for either declaration order (``(seed, epoch)`` or ``(epoch, seed)``).

    Returns 0.0 for single-epoch or no-epoch-axis configurations.

    Raises:
        ValueError: From ``compute`` when the driver slices along the epoch axis itself.

    """

    def __init__(
        self,
        order_invariant_function: str = "mean",
        distance_function: str = "l2",
        *,
        space: Space = "data",
    ) -> None:
        self.order_invariant_function = order_invariant_function
        self.distance_function = distance_function
        self.space = _check_space(space)

    def compute(self, inputs: LossInputs) -> Tensor:  # noqa: C901 -- one sequential routine
        """Mean squared (or absolute) difference between the order-invariant summaries of each epoch pair."""
        ctx = inputs.ctx
        recon = _prior_tensor(inputs, self.space, "EpochOrderInvariantPrior")
        epoch_size = ctx.axes.get_size("epoch")
        if epoch_size <= 1:
            return torch.tensor(0.0, device=recon.device)

        epoch_dim = _sliced_epoch_dim(inputs)
        if epoch_dim is None:
            return torch.tensor(0.0, device=recon.device)

        fn = self.order_invariant_function
        epoch_reprs: list[Tensor] = []
        for e in range(epoch_size):
            epoch_imgs = recon.select(epoch_dim, e)
            if fn == "mean":
                epoch_reprs.append(epoch_imgs.mean(dim=0))
            elif fn == "sum":
                epoch_reprs.append(epoch_imgs.sum(dim=0))
            elif fn == "var":
                epoch_reprs.append(epoch_imgs.var(dim=0))
            else:
                raise ValueError(f"EpochOrderInvariantPrior: unknown order_invariant_function '{fn}'")

        total = torch.tensor(0.0, device=recon.device)
        num_pairs = 0
        for i in range(len(epoch_reprs)):
            for j in range(i + 1, len(epoch_reprs)):
                diff = epoch_reprs[i] - epoch_reprs[j]
                if self.distance_function == "l2":
                    total = total + diff.pow(2).sum()
                elif self.distance_function == "l1":
                    total = total + diff.abs().sum()
                else:
                    raise ValueError(f"EpochOrderInvariantPrior: unknown distance_function '{self.distance_function}'")
                num_pairs += 1

        return total / max(num_pairs, 1)


def _sliced_epoch_dim(inputs: LossInputs) -> int | None:
    """The epoch dimension of ``inputs.recon``, or None without an epoch axis.

    A PER_SLICE term gets the reconstruction with the driver's slice axis removed, which shifts the epoch
    dimension down by one when that axis was declared before it.

    Raises:
        ValueError: If the driver slices along the epoch axis itself.

    """
    axes = inputs.ctx.axes
    epoch_dim = axes.get_dim("epoch")
    if epoch_dim is None or inputs.seed_index is None or inputs.slice_axis is None:
        return epoch_dim
    if inputs.slice_axis == "epoch":
        raise ValueError(
            "EpochOrderInvariantPrior compares epochs, so the driver must not slice along the epoch axis "
            "(set the driver's parallel_axis to another axis)."
        )
    sliced_dim = axes.get_dim(inputs.slice_axis)
    return epoch_dim - 1 if sliced_dim is not None and sliced_dim < epoch_dim else epoch_dim
