#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Mutable run state, inferred target info and the immutable run context."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from typing import TYPE_CHECKING, Any  # Any kept for State.artifacts / State.metrics

import torch
from torch import Tensor

from leakpro.attacks.gia_attacks.modular.core.axes import AxisManager
from leakpro.attacks.gia_attacks.modular.core.session import RunSession

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.interfaces import Evaluator, Modality
    from leakpro.attacks.gia_attacks.modular.core.task import Task


@dataclass
class ClientObservation:
    """What the server intercepts from the client (the FL payload).

    The ThreatModel.sanitize() call in run_attack strips (on a copy) the fields the
    attacker is not allowed to see before this reaches any component. Components
    declare the fields they read, and run_attack checks them before the first stage.

    Fields:
        gradients:   Per-parameter gradient tensors (FedSGD) or weight changes (FedAvg), one per
                     trainable parameter in ``model.parameters()`` order.
        bn_stats:    Per-BN-layer (mean, var) running statistics after local training.
                     Stripped to None by threat models without has_bn_statistics.
        pre_bn_stats: Per-BN-layer (mean, var) running statistics before local training
                     (the server's broadcast values). Stripped with bn_stats.
        batch_size:  Number of samples in the client's local batch.
                     Stripped to None by threat models without has_local_hyperparameters.
        loss_value:  Scalar training loss of the client's local batch.
                     Stripped to None by threat models without has_local_hyperparameters.
        extra:       Further named fields. No shipped component reads them, and ``ThreatModel``
                     strips only the keys it maps to a capability flag (none at present).
    """

    gradients: list[Tensor] | None = None
    bn_stats: list[tuple[Tensor, Tensor]] | None = None
    pre_bn_stats: list[tuple[Tensor, Tensor]] | None = None
    batch_size: int | None = None
    loss_value: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class TargetInfo:
    """Modality-agnostic DTO for task supervision signals.

    Dynamically stores whatever target attributes (e.g., value, soft, boxes, labels, tokens)
    a specific Task requires and a TargetSource provides.

    Axis bookkeeping
    ----------------
    ``axis_ndim`` is a reserved field recording how many parallel-axis dimensions
    (seed / epoch — the prefix contributed by ``ctx.axes``) the Tensor fields
    currently carry.  It is 0 for targets that are shared across the whole axis
    prefix (e.g. ``FixedTarget``'s oracle labels, which are the same for every
    seed) and ``len(ctx.axes.shape)`` for jointly-optimized targets allocated
    with that prefix.

    Every consumer that slices or collapses an axis goes through ``map()`` /
    ``index_axis()`` so that arbitrary fields (boxes, tokens, …) are carried
    along instead of being silently dropped by field-by-field reconstruction.
    """

    def __init__(self, **kwargs: Any) -> None:  # noqa: ANN401 -- genuinely heterogeneous value
        self.axis_ndim: int = kwargs.pop("axis_ndim", 0)
        for key, val in kwargs.items():
            setattr(self, key, val)

    def get(self, key: str, default: Any = None) -> Any:  # noqa: ANN401 -- genuinely heterogeneous value
        """Value of the field, or the default if absent."""
        return getattr(self, key, default)

    def fields(self) -> dict[str, Any]:
        """All stored fields, including the reserved ``axis_ndim``."""
        return dict(self.__dict__)

    def map(self, fn: Any, *, axis_ndim: int | None = None) -> TargetInfo:  # noqa: ANN401 -- genuinely heterogeneous value
        """Return a new TargetInfo with ``fn`` applied to every Tensor field.

        Non-Tensor fields are carried through untouched.  This is the single
        transformation primitive — ``to``/``detach``/``clone``/``index_axis``
        are all defined in terms of it, so a Task that adds a new Tensor field
        (boxes, tokens, …) needs no changes anywhere else.
        """
        new_data = {
            k: fn(v) if isinstance(v, torch.Tensor) else v
            for k, v in self.__dict__.items()
        }
        if axis_ndim is not None:
            new_data["axis_ndim"] = axis_ndim
        return TargetInfo(**new_data)

    def index_axis(self, dim: int, indices: int | torch.Tensor) -> TargetInfo:
        """Select along ``dim`` across every Tensor field.

        int    — scalar axis index; the dimension is removed (``.select``) and
                 ``axis_ndim`` drops by one.
        Tensor — 1D index tensor (minibatch selection over the sample dim); the
                 dimension is kept and ``axis_ndim`` is unchanged.
        """
        if isinstance(indices, int):
            return self.map(
                lambda t: t.select(dim, indices),
                axis_ndim=max(self.axis_ndim - 1, 0),
            )
        return self.map(lambda t: torch.index_select(t, dim, indices))

    def to(self, device: torch.device | str) -> TargetInfo:
        """Return a new TargetInfo with all internal Tensor fields moved to `device`."""
        return self.map(lambda t: t.to(device))

    def detach(self) -> TargetInfo:
        """Return a new TargetInfo with all Tensor fields detached from autograd."""
        return self.map(lambda t: t.detach())

    def clone(self) -> TargetInfo:
        """Return a deep copy with all Tensor fields cloned."""
        return self.map(lambda t: t.clone())

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 -- genuinely heterogeneous value
        """Returns a stored field, or None when absent."""
        # Dunder lookups must fail properly: copy/pickle probe the *instance* for
        # __deepcopy__/__getstate__/__reduce_ex__ and call whatever they get back,
        # so returning None here makes TargetInfo unserializable.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        # Returns None for other unset attributes instead of raising AttributeError.
        # This keeps optional attribute access (e.g. target.soft) clean and safe.
        return None

    def __repr__(self) -> str:
        """Lists the non-None fields, showing tensor shapes."""
        attrs = ", ".join(
            f"{k}={v.shape if isinstance(v, torch.Tensor) else v}"
            for k, v in self.__dict__.items()
            if v is not None
        )
        return f"TargetInfo({attrs})"


@dataclass
class RunMetadata:
    """Non-computational run info for logging/reproducibility.

    Stored on State so callbacks can read it without it affecting any math.
    """

    seed: int = 42


@dataclass(frozen=True)
class Context:
    """Immutable for the whole run — only things that actually affect computation."""

    model: torch.nn.Module
    observation: ClientObservation
    modality: Modality
    task: Task
    input_shape: tuple[int, ...]
    device: torch.device
    axes: AxisManager = field(default_factory=AxisManager)
    # Scoring is not part of the attack computation, but callers reach for it
    # through ctx right after run_attack() returns, so it rides along here.
    evaluator: Evaluator | None = None
    # Surrogate dataloader consumed by BNStatsLoss(strategy="proxy") to
    # estimate BN statistics from auxiliary data instead of the observation.
    proxy_dataloader: Any | None = None
    # The run's mutable resources (execution model, private copies of caller-owned modules).
    # The context itself stays frozen; only the session changes during a run.
    session: RunSession = field(default_factory=RunSession, repr=False, compare=False)


@dataclass
class State:
    """Mutable; flows stage → stage carrying the canonical data-space reconstruction."""

    input: Tensor | None = None
    target: TargetInfo | None = None        # seeded at run start; flows stage → stage
    artifacts: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    loss: float = inf
    iteration: int = 0
    metadata: RunMetadata = field(default_factory=RunMetadata)
