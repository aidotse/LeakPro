#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Framework configurations equivalent to a recorded v1 configuration, using only options the framework exposes.

Where the framework has no equivalent option, the difference shows up as a parity failure and is
recorded in ``deltas.KNOWN_DELTAS``; nothing here papers over it.
"""
from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from leakpro.attacks.gia_attacks.modular.components.drivers import GradientDriver
from leakpro.attacks.gia_attacks.modular.components.drivers.grad_transforms import SignGradient
from leakpro.attacks.gia_attacks.modular.components.drivers.schedules import Cosine, MultiStep
from leakpro.attacks.gia_attacks.modular.components.initializers import FromTensor
from leakpro.attacks.gia_attacks.modular.components.losses import (
    BNStatsLoss,
    CosineGradientLoss,
    EpochOrderInvariantPrior,
    GroupConsistencyLoss,
    L2GradientLoss,
    L2ImageNormLoss,
    TVLoss,
)
from leakpro.attacks.gia_attacks.modular.components.representations.pixel import PixelRepresentation
from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget
from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
from leakpro.attacks.gia_attacks.modular.core.attack import Attack
from leakpro.attacks.gia_attacks.modular.core.axes import Axis, AxisManager
from leakpro.attacks.gia_attacks.modular.core.interfaces import LossTerm
from leakpro.attacks.gia_attacks.modular.core.loss_inputs import LossInputs, LossScope
from leakpro.attacks.gia_attacks.modular.core.protocol import FedAvg, FedSGD
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, Context, TargetInfo
from leakpro.attacks.gia_attacks.modular.core.task import Task
from leakpro.tests.gia import helpers


def task(training_simulator: dict[str, Any]) -> Task:
    """The client task of a v1 ``TrainingSimulatorConfig``."""
    ts = training_simulator
    protocol = FedSGD() if ts["compute_mode"] == "gradients" else FedAvg(
        local_epochs=ts["epochs"], batch_size=ts["batch_size"], lr=ts["meta_learning_rate"])
    return Task(CrossEntropy(), protocol, model_mode=ts["model_mode"])


def _term(spec: dict[str, Any]) -> tuple[str, LossTerm]:
    """A v1 loss ``ComponentSpec`` as (v1 component name, framework loss term).

    v1 evaluates image priors on the normalized input z, so they map to ``space="model_input"``.
    """
    kind, params = spec["type"], spec["params"]
    if kind == "loss.gradient_matching":
        loss_type = params["loss_type"]
        term = {"cosine": CosineGradientLoss, "l2": L2GradientLoss}[loss_type]()
        return f"GradientMatchingLoss({loss_type})", term
    if kind == "loss.tv":
        return "TVRegularization", TVLoss(space="model_input")
    if kind == "loss.l2":
        return "L2Regularization", L2ImageNormLoss(space="model_input", reduction="per_image_sum")
    if kind == "loss.group_consistency":
        return "GroupConsistencyRegularization", GroupConsistencyLoss(space="model_input")
    if kind == "loss.bn_stats":
        strategy = params["strategy"]
        return f"BNStatisticsRegularization({strategy}_bn_statistics)", BNStatsLoss(strategy=strategy,
                                                                                    momentum=params.get("momentum"))
    raise KeyError(f"no framework equivalent registered for v1 loss {kind!r}")


def losses(config: dict[str, Any], num_seeds: int = 1) -> dict[str, tuple[LossTerm, float]]:
    """The first stage's v1 loss list, keyed by v1 component name, with v1's weights.

    v1's L2 prior sums over all seeds while the framework averages the objective over seeds, so its
    weight is scaled by ``num_seeds`` (v1's objective is ``num_seeds`` times the seed mean of that term).
    v1 adds the epoch-order prior from ``config["fedavg"]`` (with an l2 distance) rather than from the
    stage loss list, when its weight is positive and the client trains several epochs. It is placed
    right after gradient matching: with that order the framework's step-0 gradient equals v1's bit for
    bit (autograd accumulates the terms' contributions in a different order than v1's tensor layout
    does), while v1's own list order differs from it in the last bit of a few elements, which Adam at
    lr=1 amplifies past the iterate tolerance by step 10.
    """
    out = {}
    fedavg = config["fedavg"]
    for spec in config["stages"][0]["losses"]:
        name, term = _term(spec)
        weight = spec["params"]["weight"]
        out[name] = (term, weight * num_seeds if spec["type"] == "loss.l2" else weight)
        if spec["type"] == "loss.gradient_matching" and fedavg is not None and fedavg["lambda_inv"] > 0 \
                and config["training_simulator"]["epochs"] > 1:
            fn = fedavg["order_invariant_fn"]
            out[f"EpochOrderInvariantPrior(g={fn})"] = (
                EpochOrderInvariantPrior({"variance": "var"}.get(fn, fn), "l2", space="model_input"),
                fedavg["lambda_inv"])
    return out


def stage(config: dict[str, Any]) -> OptimStage:
    """A single pixel-space stage carrying the v1 objective (driver settings are not compared here)."""
    return OptimStage(losses=losses(config), rep=PixelRepresentation(min_val=0.0, max_val=1.0),
                      driver=GradientDriver(), target_source=FixedTarget())


def observation(recorded: dict[str, Any]) -> ClientObservation:
    """A recorded v1 observation as the framework's ClientObservation."""
    return ClientObservation(gradients=[g.clone() for g in recorded["gradients"]], bn_stats=recorded["post_bn_stats"],
                             pre_bn_stats=recorded["pre_bn_stats"], batch_size=recorded["num_images"])


def context(fixture: dict[str, Any]) -> Context:
    """A run context over the fixture's model and observation with the v1 client protocol."""
    return Context(model=copy.deepcopy(fixture["model"]), observation=observation(fixture["observation"]),
                   modality=helpers.image_modality(), task=task(fixture["settings"]["config"]["training_simulator"]),
                   input_shape=tuple(fixture["images"].shape), device=helpers.DEVICE)


def target_labels(fixture: dict[str, Any]) -> torch.Tensor:
    """The labels v1 optimized against, without v1's leading epoch dimension (single-epoch cases)."""
    labels = fixture["target_labels"]
    if labels.shape[0] != 1:
        raise ValueError("multi-epoch v1 label layout; this helper covers single-epoch cases only")
    return labels[0]


# ── Trajectory cases ────────────────────────────────────────────────────────────────────────────

@dataclass
class TrajectoryCase:
    """Everything needed to replay one recorded v1 trajectory through run_attack."""
    attack: Attack
    model: torch.nn.Module
    observation: ClientObservation
    input_shape: tuple[int, ...]
    target: TargetInfo
    to_current_layout: Callable[[torch.Tensor], torch.Tensor]


def v1_step_milestones(n: int) -> tuple[int, int, int]:
    """v1's ``scheduler.step`` milestones (optimizer_factories.py): ``[n//2, int(n//1.6), int(n//1.142)]``."""
    return (n // 2, int(n // 1.6), int(n // 1.142))


def _layout(config: dict[str, Any]) -> tuple[AxisManager, Callable[[torch.Tensor], torch.Tensor]]:
    """The reference stores [E, N, G, C, H, W]; the framework stores [*axes, N, C, H, W] with axes (epoch, seed) when present."""
    epochs = config["training_simulator"]["epochs"] if config["epoch_handling"] == "multi_epoch_separate" else 1
    seeds = config["num_seeds_per_image"]
    axes = [Axis("epoch", epochs)] if epochs > 1 else []
    axes += [Axis("seed", seeds)] if seeds > 1 else []

    def convert(z: torch.Tensor) -> torch.Tensor:
        z = z.permute(0, 2, 1, 3, 4, 5)            # [E, G, N, C, H, W]
        if seeds == 1:
            z = z[:, 0]                            # [E, N, C, H, W]
        if epochs == 1:
            z = z[0]
        return z.contiguous()

    return AxisManager(*axes), convert


def trajectory_case(fixture: dict[str, Any]) -> TrajectoryCase:
    """The framework configuration equivalent to the recorded v1 run, or NotImplementedError naming the missing option."""
    axes, to_current = _layout(fixture["settings"]["config"])
    return _build(fixture, axes, to_current)


def _build(fixture: dict[str, Any], axes: AxisManager, to_current: Callable[[torch.Tensor], torch.Tensor]) -> TrajectoryCase:
    config = fixture["settings"]["config"]
    stage_cfg = config["stages"][0]
    rep = PixelRepresentation(space="model_input", init=FromTensor(to_current(fixture["z0"])),
                              clip=stage_cfg["constraint"] is not None)
    labels = fixture["target_labels"]
    if not bool((labels == labels[0]).all()):
        raise ValueError("per-epoch label assignments differ; the builder covers shared labels only")
    stage = OptimStage(losses=losses(config, num_seeds=axes.get_size("seed")), rep=rep,
                       driver=driver(stage_cfg), target_source=FixedTarget())
    attack = Attack(modality=helpers.image_modality(), task=task(config["training_simulator"]), stages=[stage],
                    axes=axes)
    return TrajectoryCase(attack=attack, model=copy.deepcopy(fixture["model"]),
                          observation=observation(fixture["observation"]),
                          input_shape=tuple(fixture["images"].shape), target=TargetInfo(value=labels[0]),
                          to_current_layout=to_current)


def driver(stage_cfg: dict[str, Any]) -> GradientDriver:
    """A v1 optimizer stage's driver: v1's optimizer, schedule and gradient sign."""
    max_iter = stage_cfg["max_iterations"]
    scheduler = stage_cfg["scheduler_type"] and stage_cfg["scheduler_type"]["type"]
    schedule = {None: None, "scheduler.cosine": Cosine(),
                "scheduler.step": MultiStep(milestones=v1_step_milestones(max_iter), gamma=0.1)}[scheduler]
    optimizer = {"optimizer.adam": torch.optim.Adam,
                 "optimizer.lbfgs": torch.optim.LBFGS}[stage_cfg["optimizer_type"]["type"]]
    kwargs = {"max_iter": 20, "history_size": 100} if optimizer is torch.optim.LBFGS else {}
    return GradientDriver(lr=stage_cfg["learning_rate"], max_iter=max_iter, return_best=True,
                          optimizer_cls=optimizer, optimizer_kwargs=kwargs, schedule=schedule,
                          grad_transforms=(SignGradient(),) if stage_cfg["use_gradient_sign"] else ())


# ── Replicas of v1 defects (evidence only, never used by the library) ──

class V1LastSeedBNStats(LossTerm):
    """v1's multi-seed BN prior: only the last seed's forward pass contributes (its hooks overwrite r_feature)."""

    def __init__(self, inner: BNStatsLoss) -> None:
        self.inner = inner

    def bind(self, ctx: Context) -> V1LastSeedBNStats:
        return V1LastSeedBNStats(self.inner.bind(ctx))

    def release(self) -> None:
        self.inner.release()

    def compute(self, inputs: LossInputs) -> torch.Tensor:
        value = self.inner.compute(inputs)
        seeds = inputs.ctx.axes.get_size("seed")
        return value * seeds if inputs.seed_index == seeds - 1 else value * 0.0


class V1BroadcastGroupConsistency(LossTerm):
    """v1's group-consistency term with its [E,N,C,H,W] vs [E,N,1,C,H,W] broadcast."""

    scope = LossScope.GROUP

    def compute(self, inputs: LossInputs) -> torch.Tensor:
        z = inputs.model_input                      # framework layout [G, N, C, H, W]
        r = z.permute(1, 0, 2, 3, 4).unsqueeze(0)   # v1 layout [1, N, G, C, H, W]
        consensus = r.mean(dim=2, keepdim=True)
        seeds = r.shape[2]
        return sum((r[:, :, g, ...] - consensus).pow(2).mean() for g in range(seeds)) / seeds


def with_v1_defects(case: TrajectoryCase) -> TrajectoryCase:
    """The same case with the STG terms replaced by the v1-defect replicas."""
    stage = case.attack.stages[0]
    losses = dict(stage.losses)
    bn_name = next(n for n in losses if n.startswith("BNStatisticsRegularization"))
    term, w = losses[bn_name]
    losses[bn_name] = (V1LastSeedBNStats(term), w)
    _, w = losses["GroupConsistencyRegularization"]
    losses["GroupConsistencyRegularization"] = (V1BroadcastGroupConsistency(), w)
    attack = dataclasses.replace(case.attack, stages=[dataclasses.replace(stage, losses=losses)])
    return dataclasses.replace(case, attack=attack)
