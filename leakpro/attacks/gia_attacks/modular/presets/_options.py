#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Option helpers shared by the image presets."""
from __future__ import annotations

from typing import Literal

from leakpro.attacks.gia_attacks.modular.components.drivers.schedules import MultiStep
from leakpro.attacks.gia_attacks.modular.components.initializers import Gaussian
from leakpro.attacks.gia_attacks.modular.components.losses import CosineGradientLoss, L2GradientLoss
from leakpro.attacks.gia_attacks.modular.components.representations.pixel import PixelRepresentation
from leakpro.attacks.gia_attacks.modular.components.target_inference import DimitrovInference, IDLGInference
from leakpro.attacks.gia_attacks.modular.core.interfaces import LossTerm, Space, TargetInference

LabelSource = Literal["oracle", "idlg", "dimitrov"]
GradLoss = Literal["cosine", "l2"]

# Learning rate x0.1 at n // 2, n // 1.6 and n // 1.142 steps: the schedule of the GIA Running/Estimate and Huang reference runs.
STEP_DECAY = MultiStep(divisors=(2, 1.6, 1.142))


def labels_to_inference(
    source: LabelSource, *, allow: tuple[LabelSource, ...] = ("oracle", "idlg"),
) -> TargetInference | None:
    """Map a label source to its inference step.

    ``"oracle"`` gives ``None``: the caller passes ``target=`` to ``run_attack``. ``"idlg"`` gives
    ``IDLGInference()``; ``"dimitrov"`` gives ``DimitrovInference()``, which reads epochs, batch size
    and learning rate from the ``FedAvg`` protocol.

    Raises:
        ValueError: ``source`` is not one of ``allow``.

    """
    if source not in allow:
        raise ValueError(f"labels must be one of {list(allow)}, got {source!r}")
    if source == "oracle":
        return None
    if source == "idlg":
        return IDLGInference()
    return DimitrovInference()


def pixel_representation(space: Space = "data", *, clip: bool = True) -> PixelRepresentation:
    """Pixel parameterisation in [0, 1]: data space starts from U(0, 1), model-input space from N(0, 1)."""
    if space == "model_input":
        return PixelRepresentation(0.0, 1.0, space="model_input", init=Gaussian(), clip=clip)
    return PixelRepresentation(0.0, 1.0, clip=clip)


def grad_loss(loss_type: GradLoss) -> LossTerm:
    """The gradient-matching term named by ``loss_type``."""
    if loss_type == "cosine":
        return CosineGradientLoss()
    if loss_type == "l2":
        return L2GradientLoss()
    raise ValueError(f"loss_type must be 'cosine' or 'l2', got {loss_type!r}")
