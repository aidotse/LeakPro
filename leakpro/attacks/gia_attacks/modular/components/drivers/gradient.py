#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Gradient-based driver that optimizes the representation's parameters with a torch optimizer."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from leakpro.attacks.gia_attacks.modular.components.drivers.schedules import Cosine
from leakpro.attacks.gia_attacks.modular.core.events import StepEvent
from leakpro.attacks.gia_attacks.modular.core.interfaces import (
    Callback,
    Driver,
    GradTransform,
    Representation,
    Schedule,
)
from leakpro.attacks.gia_attacks.modular.core.loss_eval import LossBreakdown, evaluate_stage_losses
from leakpro.attacks.gia_attacks.modular.core.loss_inputs import LossScope
from leakpro.attacks.gia_attacks.modular.core.result import (
    DriverResult,
    clone_artifacts,
    merge_artifact_slice,
    merge_slice,
)

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
    from leakpro.attacks.gia_attacks.modular.core.state import Context, TargetInfo


def _model_input(rep: Representation, recon: Tensor, ctx: Context, dim: int | None = None, index: int = 0) -> Tensor:
    """The model input matching ``recon`` (slice ``index`` along ``dim`` of the full iterate, if ``dim`` is set).

    A representation that overrides ``model_input`` supplies it, built afresh for each slice so every
    per-slice backward has its own graph. One that keeps the default gets it from ``recon`` here, so
    its decode is not run a second time.
    """
    if type(rep).model_input is Representation.model_input:
        return ctx.modality.to_model_input(recon)
    full = rep.model_input(ctx.modality)
    return full.select(dim, index) if dim is not None else full


@dataclass(frozen=True)
class GradientDriver(Driver):
    """Gradient-based driver covering Adam, SGD, LBFGS — all torch.optim optimizers.

    Uses a closure unconditionally so LBFGS (which calls it multiple times for line search)
    works identically to Adam/SGD (which call it once).

    Step contract
    -------------
    1. Each step runs ``zero_grad → objective → backward(s) → grad_transforms (in tuple order) →
       optimizer.step → scheduler.step → rep.project``. The transforms run inside the closure, so an
       optimizer that evaluates the closure several times sees transformed gradients every time. Each transform is bound
       once per run (``GradTransform.bind``).
       Noise then sign, ``(GaussianNoise(std), SignGradient())``, is the order of the reference implementations.
    2. The objective is ``Σ_s L_s / S + L_group``: ``L_s`` is the weighted PER_SLICE sum on slice ``s``
       of the parallel axis and ``S`` the slice count. The closure returns exactly that value.
    3. The step reports (``StepEvent.loss``/``components``) the **first** closure evaluation of the
       step: the objective at the iterate the step started from. Per-slice components are averaged
       over slices and GROUP components are added once; nothing is summed across closure calls.

    Two-pass loss evaluation
    ------------------------
    Pass 1 — PER_SLICE (default):
        Every PER_SLICE loss is evaluated once per slice. Backward is called immediately after
        each slice (on ``L_s / S``), freeing the activation graph and keeping VRAM bounded. The decode
        runs once for all slices, so every backward but the last retains the graph: the part a
        representation builds before slicing (a generator, the map from model input to data space) is
        shared by every slice, and only the slice's own activations are freed.

    Pass 2 — GROUP:
        GROUP losses (e.g. GroupConsistencyLoss) are evaluated once on the full live
        reconstruction [*axes, N, C, H, W] after the per-slice loop.  A single backward
        call lets gradients flow across slice boundaries.  GROUP losses must self-normalize
        by slice count internally (the driver does not apply the 1/S factor).

    Best-tracking uses only PER_SLICE loss values because best-per-slice is a per-slice
    concept; GROUP losses influence the gradient trajectory but not slice selection. Only the
    first closure evaluation of each step is a candidate (the iterate the step reports).

    Result
    ------
    With ``return_best=True`` each slice of the returned ``DriverResult`` (input, artifacts and,
    for a target carrying the axis, target) is taken from the evaluation that gave that slice its
    lowest PER_SLICE loss, and ``slice_losses`` holds those losses. With ``return_best=False`` the
    final iterate is returned and scored once more so ``loss``/``slice_losses`` describe it.

    Early stopping
    --------------
    With ``patience`` set, the run counts consecutive steps whose reported loss (see the step contract)
    is not strictly below the best reported so far, and stops once the count reaches ``patience``. The
    stopping step's ``StepEvent`` is still emitted and ``DriverResult.iterations`` is the number of
    steps run. ``None`` always runs ``max_iter`` steps.

    Parallel axis
    -------------
    Which axis the per-slice loop iterates over is ``parallel_axis`` (default ``"seed"``).
    When the named axis is absent from ``ctx.axes`` the loop is a single pass over the
    whole reconstruction.
    """

    lr: float = 0.1
    max_iter: int = 4000
    return_best: bool = True
    optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam
    optimizer_kwargs: Mapping[str, Any] = field(default_factory=dict)
    schedule: Schedule | None = field(default_factory=Cosine)
    grad_transforms: tuple[GradTransform, ...] = ()
    parallel_axis: str = "seed"
    patience: int | None = None

    def __post_init__(self) -> None:
        """Rejects a patience below 1."""
        if self.patience is not None and self.patience < 1:
            raise ValueError(f"patience must be at least 1, got {self.patience}")

    def tunables(self) -> dict[str, float]:
        """Learning rate and iteration budget exposed for tuning."""
        return {"lr": self.lr, "max_iter": float(self.max_iter)}

    def with_overrides(self, flat: dict[str, float]) -> GradientDriver:
        """Returns a copy with learning rate and iteration budget taken from the overrides."""
        return replace(self,
            lr=flat.get("lr", self.lr),
            max_iter=int(flat.get("max_iter", self.max_iter)),
        )

    def run(  # noqa: C901, PLR0915 -- one sequential routine
        self,
        stage: OptimStage,
        ctx: Context,
        *,
        callbacks: Sequence[Callback] = (),
        stage_idx: int = 0,
    ) -> DriverResult:
        """Optimizes the representation's parameters against the stage losses and returns the best result."""
        rep, target_source = stage.rep, stage.target_source
        all_groups = [*rep.parameters(), *target_source.parameters()]
        all_params = [p for pg in all_groups for p in pg["params"]]

        # L-BFGS does not support per-parameter groups — flatten to a single param list.
        if self.optimizer_cls is torch.optim.LBFGS:
            optimizer = self.optimizer_cls(all_params, lr=self.lr, **self.optimizer_kwargs)
        else:
            param_groups = [
                {"params": pg["params"], "lr": self.lr * pg.get("lr_scale", 1.0)}
                for pg in all_groups
            ]
            optimizer = self.optimizer_cls(param_groups, lr=self.lr, **self.optimizer_kwargs)

        scheduler = self.schedule.build(optimizer, self.max_iter) if self.schedule is not None else None

        has_group_losses = any(
            getattr(t, "scope", LossScope.PER_SLICE) is LossScope.GROUP for t, _ in stage.losses.values()
        )

        seed_dim: int | None = ctx.axes.get_dim(self.parallel_axis)
        seed_size: int = ctx.axes.get_size(self.parallel_axis) if seed_dim is not None else 1
        prefix = tuple(ctx.axes.shape)

        # Per-slice best tracking (PER_SLICE loss only). The snapshot holds the observable, the
        # artifacts and the target of the evaluation that set each slice's best loss. A target without
        # the axis prefix is shared by all slices, so it is the one of the evaluation with the lowest
        # mean slice loss.
        best_losses: list[float] = [float("inf")] * seed_size
        best_mean: float = float("inf")
        best_input: Tensor = rep.to_observable().clone()
        best_artifacts = clone_artifacts(rep.artifacts())
        best_target = _detached(target_source.current())

        def snapshot(s: int, target: TargetInfo | None) -> None:
            nonlocal best_input, best_artifacts, best_target
            observable = rep.to_observable()
            artifacts = clone_artifacts(rep.artifacts())
            if seed_dim is None:
                best_input = observable.detach().clone()
                best_artifacts = artifacts
                best_target = _detached(target)
                return
            merge_slice(best_input, observable.detach(), seed_dim, s)
            merge_artifact_slice(best_artifacts, artifacts, prefix, seed_dim, s)
            if target is not None and target.axis_ndim > 0:
                current = target.fields()
                for name, value in best_target.fields().items():
                    if isinstance(value, Tensor):
                        merge_slice(value, current[name].detach(), seed_dim, s)

        transforms = tuple(t.bind(ctx.device) for t in self.grad_transforms)

        steps_run = 0
        best_reported = float("inf")
        stagnant = 0

        for i in range(self.max_iter):
            first: dict[str, Any] | None = None

            def closure() -> Tensor:  # noqa: C901 -- one sequential routine
                nonlocal first, best_mean, best_target
                optimizer.zero_grad()
                target = target_source.current()
                objective: float = 0.0
                components: dict[str, float] = {}

                # ── Pass 1: per-slice (model-forward losses) ───────────────────────────
                # Each slice gets its own forward + backward; the activation graph is freed
                # immediately after each backward, keeping peak VRAM bounded.
                recon = rep.decode()
                slice_losses: list[float] = []
                for s, out_s in self._slice_losses(stage, ctx, recon, target, create_graph=True):
                    loss_slice = out_s.total
                    for n, value in out_s.terms.items():
                        components[n] = components.get(n, 0.0) + value.item() / seed_size
                    if loss_slice.grad_fn is not None:
                        # A decode that built a graph is shared by the slices still to come.
                        retain = recon.grad_fn is not None and s < seed_size - 1
                        (loss_slice / seed_size).backward(retain_graph=retain)
                    objective += loss_slice.item() / seed_size
                    slice_losses.append(loss_slice.item())
                    del loss_slice, out_s  # free this slice's activations before the next forward

                # Best-tracking uses per-slice (PER_SLICE-only) loss per seed, at the first
                # evaluation of the step only: that is the projected iterate the step starts
                # from, so its observable is exactly the evaluated point. Later evaluations
                # (an L-BFGS line search) sit at unprojected points whose observable differs.
                if self.return_best and first is None:
                    mean = sum(slice_losses) / seed_size
                    if seed_dim is not None and target is not None and target.axis_ndim == 0 and mean < best_mean:
                        best_target = _detached(target)
                    best_mean = min(best_mean, mean)
                    for s, value in enumerate(slice_losses):
                        if value < best_losses[s]:
                            best_losses[s] = value
                            snapshot(s, target)

                # ── Pass 2: group (full live recon, single backward) ───────────────────
                # Re-decode to get a fresh graph unaffected by the per-slice backwards above.
                if has_group_losses:
                    recon_full = rep.decode()
                    out_group = evaluate_stage_losses(
                        stage.losses, ctx, recon_full, target,
                        model_input=_model_input(rep, recon_full, ctx), scope=LossScope.GROUP, full_recon=recon_full,
                    )
                    group_loss = out_group.total
                    for n, value in out_group.terms.items():
                        components[n] = components.get(n, 0.0) + value.item()
                    if group_loss.grad_fn is not None:
                        group_loss.backward()
                    objective += group_loss.item()

                for transform in transforms:
                    transform(all_params)

                if first is None:
                    first = {"loss": objective, "components": components}
                return torch.tensor(objective, device=ctx.device)

            optimizer.step(closure)

            if scheduler is not None:
                scheduler.step()

            rep.project(ctx)

            if callbacks:
                artifacts = rep.artifacts()
                for cb in callbacks:
                    cb.on_step_end(StepEvent(stage_idx, i, first["loss"], dict(first["components"]), artifacts))

            steps_run = i + 1
            if self.patience is not None:
                if first["loss"] < best_reported:
                    best_reported, stagnant = first["loss"], 0
                else:
                    stagnant += 1
                    if stagnant >= self.patience:
                        break

        if not self.return_best:
            # Score the final iterate once so loss/slice_losses describe what is returned.
            best_losses = self._final_slice_losses(stage, ctx)
            best_input = rep.to_observable().clone()
            best_artifacts = clone_artifacts(rep.artifacts())
            best_target = _detached(target_source.current())

        return DriverResult(
            input=best_input,
            target=best_target,
            artifacts=best_artifacts,
            loss=sum(best_losses) / seed_size,
            iterations=steps_run,
            slice_losses=torch.tensor(best_losses) if seed_dim is not None else None,
        )

    def _slice_losses(
        self, stage: OptimStage, ctx: Context, recon: Tensor, target: TargetInfo | None, *, create_graph: bool,
    ) -> Iterator[tuple[int, LossBreakdown]]:
        """The PER_SLICE losses of each slice of ``recon``, evaluated lazily in slice order.

        A caller that backpropagates a slice's loss before asking for the next frees that slice's
        activation graph in time.
        """
        rep, target_source = stage.rep, stage.target_source
        seed_dim = ctx.axes.get_dim(self.parallel_axis)
        seed_size = ctx.axes.get_size(self.parallel_axis) if seed_dim is not None else 1
        for s in range(seed_size):
            recon_s = recon.select(seed_dim, s) if seed_dim is not None else recon
            target_s = target_source.slice(target, seed_dim, s) if seed_dim is not None else target
            yield s, evaluate_stage_losses(
                stage.losses, ctx, recon_s, target_s,
                model_input=_model_input(rep, recon_s, ctx, seed_dim, s), create_graph=create_graph,
                seed_index=s, full_recon=recon, slice_axis=self.parallel_axis,
            )

    def _final_slice_losses(self, stage: OptimStage, ctx: Context) -> list[float]:
        """The PER_SLICE objective of every slice at the current iterate (no second-order graph)."""
        recon = stage.rep.decode()
        target = stage.target_source.current()
        return [out.total.item() for _, out in self._slice_losses(stage, ctx, recon, target, create_graph=False)]


def _detached(target: TargetInfo | None) -> TargetInfo | None:
    return target.detach().clone() if target is not None else None
