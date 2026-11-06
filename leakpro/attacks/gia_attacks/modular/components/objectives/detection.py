#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Objectives for object detectors.

Detectors differ in how they are called: torchvision detectors take ``(images, targets)`` and return
a loss dict in train mode, HuggingFace DETR takes ``pixel_values`` and ``labels`` and returns an output
with ``.loss``, and a YOLO-style model returns raw head outputs that a separate criterion scores.

Two-stage torchvision detectors (anything built on ``RoIAlign``) cannot be attacked by gradient
matching: ``RoIAlign``'s backward is not twice differentiable. Observing them works. DETR needs
``attn_implementation="eager"`` for the same reason (the SDPA backward is not differentiable).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from torch import Tensor, nn

from leakpro.attacks.gia_attacks.modular.components.utils.boxes import to_detr_labels, to_torchvision_targets
from leakpro.attacks.gia_attacks.modular.core.interfaces import ClientProtocol, ModelFn, ModelMode, Objective
from leakpro.attacks.gia_attacks.modular.core.protocol import FedAvg, FedSGD

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo


@dataclass(frozen=True)
class TorchvisionDetection(Objective):
    """Weighted sum of the loss dict a torchvision detector returns for ``model(images, targets)``.

    Pin the detector's ``min_size == max_size ==`` the image side so its internal transform does
    not resize. Both protocols run: FedAvg differentiates through the local update.

    Attributes:
        loss_weights: Weight per loss-dict key; missing keys weigh 1.0.

    """

    loss_weights: Mapping[str, float] | None = None
    supported_protocols: ClassVar[tuple[type[ClientProtocol], ...]] = (FedSGD, FedAvg)

    def loss(self, model_fn: ModelFn, model_input: Tensor, target: TargetInfo) -> Tensor:
        """Weighted sum of the detector's training losses on the padded box target."""
        out = model_fn(list(model_input.unbind(0)), to_torchvision_targets(target))
        if not isinstance(out, dict):
            raise ValueError(
                f"TorchvisionDetection.loss expected the loss dict torchvision "
                f"detectors return in train mode, got {type(out).__name__}. "
                f"Is the model in .eval() mode?"
            )
        weights = self.loss_weights or {}
        return sum(weights.get(name, 1.0) * value for name, value in out.items())


@dataclass(frozen=True)
class HuggingFaceDetection(Objective):
    """The Hungarian-matched loss ``DetrForObjectDetection`` computes when given ``labels``.

    Attributes:
        img_size: Side of the square image, for normalizing boxes to DETR's cxcywh format.

    """

    img_size: int
    supported_protocols: ClassVar[tuple[type[ClientProtocol], ...]] = (FedSGD, FedAvg)

    def loss(self, model_fn: ModelFn, model_input: Tensor, target: TargetInfo) -> Tensor:
        """DETR's own loss on the padded box target."""
        return model_fn(pixel_values=model_input, labels=to_detr_labels(target, self.img_size)).loss


@dataclass(frozen=True)
class CriterionDetection(Objective):
    """``criterion(model_fn(x), target_encoder(target))`` for a detector scored by a separate criterion.

    Attributes:
        criterion: Loss on the raw model output and the encoded target (e.g. a YOLO ``ComputeLoss``).
        target_encoder: Maps the ``TargetInfo`` to the criterion's target format.
        sum_output: Sum the criterion output (for a criterion returning per-component losses).
        prepare: Extra model preparation run after the train/eval mode is set.

    """

    criterion: Callable[[Any, Tensor], Tensor]
    target_encoder: Callable[[TargetInfo], Tensor]
    sum_output: bool = False
    prepare: Callable[[nn.Module, ModelMode], None] | None = None
    supported_protocols: ClassVar[tuple[type[ClientProtocol], ...]] = (FedSGD, FedAvg)

    def prepare_model(self, model: nn.Module, mode: ModelMode) -> None:
        """Set the module mode, then run ``prepare``."""
        super().prepare_model(model, mode)
        if self.prepare is not None:
            self.prepare(model, mode)

    def loss(self, model_fn: ModelFn, model_input: Tensor, target: TargetInfo) -> Tensor:
        """The criterion on the raw output and the encoded target."""
        out = model_fn(model_input)
        loss = self.criterion(out, self.target_encoder(target))
        return loss.sum() if self.sum_output else loss
