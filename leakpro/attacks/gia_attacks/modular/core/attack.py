#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Attack definition and the entry point that runs it."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import torch

from leakpro.attacks.gia_attacks.modular.core.axes import AxisManager
from leakpro.attacks.gia_attacks.modular.core.requirements import validate_requirements
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, Context, RunMetadata, State, TargetInfo
from leakpro.attacks.gia_attacks.modular.core.threat_model import ThreatModel

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.interfaces import (
        Callback,
        Evaluator,
        Modality,
        TargetInference,
    )
    from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
    from leakpro.attacks.gia_attacks.modular.core.task import Task


@dataclass
class Attack:
    """What a preset returns: modality + task + ordered stages + optional threat model.

    Attributes:
        modality: Maps data space to the model input.
        task: What the client trains and how it reports back.
        stages: The optimisation stages, run in order.
        threat_model: What the attacker may see. ``run_attack`` strips the observation fields it does not
            grant before any component reads them. ``None`` applies no sanitization: the attacker sees every
            field of the observation passed to ``run_attack`` (BN statistics, batch size, loss value).
            Pass ``ThreatModel()`` for an attacker who sees only the gradients.
        target_inference: Recovers the labels from the observation when ``run_attack`` gets no ``target``.
        axes: Parallel axes (seed, epoch) prefixing the shape of the optimised tensors.
        evaluator: Scores a finished reconstruction; ``None`` leaves scoring to the caller.

    """

    modality: Modality
    task: Task
    stages: list[OptimStage]
    threat_model: ThreatModel | None = None
    target_inference: TargetInference | None = None
    axes: AxisManager = field(default_factory=AxisManager)
    # How a finished reconstruction is scored.  Presets set the evaluator that
    # matches the attack's output format (pixels, one-hot rows, tree intervals);
    # None means the caller scores it themselves.
    evaluator: Evaluator | None = None

    def tunables(self) -> dict[str, float]:
        """Flat HPO surface across all stages, keyed by '<stage_idx>.<name>'."""
        flat: dict[str, float] = {}
        for i, stage in enumerate(self.stages):
            flat.update({f"{i}.{k}": v for k, v in stage.tunables().items()})
        return flat

    def with_overrides(self, flat: dict[str, float]) -> Attack:
        """Return a new Attack with per-stage overrides applied (immutable)."""
        new_stages = []
        for i, stage in enumerate(self.stages):
            prefix = f"{i}."
            stage_flat = {k[len(prefix):]: v for k, v in flat.items() if k.startswith(prefix)}
            new_stages.append(stage.with_overrides(stage_flat) if stage_flat else stage)
        return replace(self, stages=new_stages)


def run_attack(
    attack: Attack,
    model: torch.nn.Module,
    observation: ClientObservation,
    input_shape: tuple[int, ...],
    device: torch.device,
    *,
    target: TargetInfo | None = None,
    start: int = 0,
    stop: int | None = None,
    input_state: State | None = None,
    metadata: RunMetadata | None = None,
    callbacks: list[Callback] | None = None,
    proxy_dataloader: Any | None = None,  # noqa: ANN401 -- any iterable of (inputs, labels) batches
) -> tuple[State, Context]:
    """Run the attack pipeline.

    observation: intercepted ClientObservation (gradients, etc.)
    target: supervision signal (oracle labels, or None when the attack infers them)
    start/stop: run a contiguous stage range for isolated HPO
    input_state: seed the first stage from a prior run (skips seeding when start > 0)
    metadata: non-computational info (seed, logging) accessible to callbacks via state.metadata
    proxy_dataloader: auxiliary (inputs, labels) batches, for components that estimate statistics
        from proxy data (e.g. BNStatsLoss(strategy="proxy"))

    Raises:
        ValueError: Before any stage runs, if a stage in ``[start:stop]`` or the target inference
            requires an observation field the (sanitized) observation lacks, or a proxy dataloader
            that was not passed.

    """
    if metadata is None:
        metadata = RunMetadata()

    # The run works on a private copy, so nothing (target inference, BN hooks, mode switches)
    # can touch the caller's model.
    model = copy.deepcopy(model)

    if attack.threat_model is not None:
        observation = attack.threat_model.sanitize(observation)
    validate_requirements(attack, observation, proxy_dataloader=proxy_dataloader, start=start, stop=stop)

    ctx = Context(
        model=model,
        observation=observation,
        modality=attack.modality,
        task=attack.task,
        input_shape=input_shape,
        device=device,
        axes=attack.axes,
        evaluator=attack.evaluator,
        proxy_dataloader=proxy_dataloader,
    )

    if start > 0 and input_state is None:
        raise ValueError("Resuming an attack pipeline from start > 0 requires an input_state.")

    # Only seed at the very start of a full run. Resumed runs (start > 0) must not
    # re-seed — doing so would change the RNG state and break split-run parity.
    if start == 0:
        torch.manual_seed(metadata.seed)

    if start == 0 and target is None and attack.target_inference is not None:
        target = attack.target_inference.infer(ctx)

    if input_state is not None:
        # The caller's state is never modified: the run works on a copy whose tensors are remapped to
        # the execution device (the state may have been saved on a different worker/GPU).
        state = replace(
            input_state,
            input=input_state.input.to(device) if input_state.input is not None else None,
            target=input_state.target.to(device) if input_state.target is not None else None,
            artifacts={k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in input_state.artifacts.items()},
            metrics=dict(input_state.metrics),
            metadata=metadata,
        )
    else:
        state = State(target=target, metadata=metadata)

    for idx, stage_cfg in enumerate(attack.stages[start:stop], start=start):
        state = stage_cfg.run(state, ctx, stage_idx=idx, callbacks=callbacks)

    return state, ctx
