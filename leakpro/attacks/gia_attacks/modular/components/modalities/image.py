#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Mean/std normalization modality for image inputs."""

from __future__ import annotations

import torch
from torch import Tensor

from leakpro.attacks.gia_attacks.modular.core.interfaces import Modality


class ImageNormalizationModality(Modality):
    """Vision modality: normalizes [0,1] pixels for the model.

    Expects tensors in [B, C, H, W] format.  Quality scoring lives in
    ``components/evaluators/image.py::ImageEvaluator``.
    """

    def __init__(self, mean: list[float], std: list[float], device: torch.device) -> None:
        self.mean = torch.tensor(mean, device=device).view(1, -1, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, -1, 1, 1)

    def to_model_input(self, data: Tensor) -> Tensor:
        """Normalize pixels with the per-channel mean and std."""
        return (data - self.mean) / self.std

    def strip_model_input(self, model_input: Tensor) -> Tensor:
        """Undo the per-channel mean/std normalization."""
        return model_input * self.std + self.mean
