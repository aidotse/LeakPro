#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Objectives for classifiers and single-output tabular heads.

``reduction="sum"`` on the scalar heads mirrors a ``criterion(...).sum()`` training loop. It does not
change a single-step cosine gradient-matching attack (cosine is scale-invariant) but it does change
FedAvg updates, so it is exposed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

import torch
import torch.nn.functional as F  # noqa: N812 -- conventional alias
from torch import Tensor

from leakpro.attacks.gia_attacks.modular.core.interfaces import ClientProtocol, ModelFn, Objective
from leakpro.attacks.gia_attacks.modular.core.protocol import FedAvg, FedSGD

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo

Reduction = Literal["mean", "sum"]


def _flat(x: Tensor) -> Tensor:
    return x.reshape(x.shape[0], -1).squeeze(-1) if x.dim() > 1 else x


def _check_reduction(reduction: str) -> None:
    if reduction not in ("mean", "sum"):
        raise ValueError("reduction must be 'mean' or 'sum'.")


@dataclass(frozen=True)
class CrossEntropy(Objective):
    """Softmax cross-entropy of ``model_fn(x)``.

    A soft target (logits) is used as the softmax distribution and takes priority over a hard
    class index in ``value``.

    Attributes:
        reduction: ``"mean"`` or ``"sum"`` over the batch.

    """

    reduction: Reduction = "mean"
    supported_protocols: ClassVar[tuple[type[ClientProtocol], ...]] = (FedSGD, FedAvg)

    def __post_init__(self) -> None:
        """Validate the reduction."""
        _check_reduction(self.reduction)

    def loss(self, model_fn: ModelFn, model_input: Tensor, target: TargetInfo) -> Tensor:
        """Cross-entropy between the logits and the soft or hard target."""
        out = model_fn(model_input)
        if target.soft is not None:
            return F.cross_entropy(out, F.softmax(target.soft, dim=-1), reduction=self.reduction)
        if target.value is not None:
            return F.cross_entropy(out, target.value, reduction=self.reduction)
        raise ValueError("TargetInfo must provide either 'value' or 'soft'.")


@dataclass(frozen=True)
class BinaryCrossEntropy(Objective):
    """BCE-with-logits of a single-logit head.

    A soft target is a logit; its sigmoid is P(y=1).

    Attributes:
        reduction: ``"mean"`` or ``"sum"`` over the batch.
        pos_weight: Optional positive-class weight, as in ``BCEWithLogitsLoss``.

    """

    reduction: Reduction = "mean"
    pos_weight: float | None = None
    supported_protocols: ClassVar[tuple[type[ClientProtocol], ...]] = (FedSGD, FedAvg)

    def __post_init__(self) -> None:
        """Validate the reduction."""
        _check_reduction(self.reduction)

    def loss(self, model_fn: ModelFn, model_input: Tensor, target: TargetInfo) -> Tensor:
        """BCE-with-logits between the single logit and the hard or soft binary target."""
        out = model_fn(model_input)
        if target.soft is not None:
            values = torch.sigmoid(_flat(target.soft))
        elif target.value is not None:
            values = _flat(target.value).float()
        else:
            raise ValueError("TargetInfo must provide either 'value' or 'soft'.")
        pw = None if self.pos_weight is None else torch.tensor(self.pos_weight, device=out.device)
        return F.binary_cross_entropy_with_logits(_flat(out), values, pos_weight=pw, reduction=self.reduction)


@dataclass(frozen=True)
class MeanSquaredError(Objective):
    """MSE of a single-output regression head; a soft target is the target value itself.

    Attributes:
        reduction: ``"mean"`` or ``"sum"`` over the batch.

    """

    reduction: Reduction = "mean"
    supported_protocols: ClassVar[tuple[type[ClientProtocol], ...]] = (FedSGD, FedAvg)

    def __post_init__(self) -> None:
        """Validate the reduction."""
        _check_reduction(self.reduction)

    def loss(self, model_fn: ModelFn, model_input: Tensor, target: TargetInfo) -> Tensor:
        """MSE between the single output and the target value."""
        out = model_fn(model_input)
        if target.soft is not None:
            values = _flat(target.soft)
        elif target.value is not None:
            values = _flat(target.value).float()
        else:
            raise ValueError("TargetInfo must provide either 'value' or 'soft'.")
        return F.mse_loss(_flat(out), values, reduction=self.reduction)
