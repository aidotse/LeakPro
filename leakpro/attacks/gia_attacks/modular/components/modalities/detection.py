#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Identity modality for detectors that take [0, 1] pixels."""

from __future__ import annotations

from torch import Tensor

from leakpro.attacks.gia_attacks.modular.core.interfaces import Modality


class DetectionImageModality(Modality):
    """Identity transform for detectors that take ``[0, 1]`` pixels.

    Unlike a classifier, a detector normalizes its input itself: a torchvision
    detector's ``GeneralizedRCNNTransform`` does it inside the model's forward pass
    (see ``TorchvisionDetection``), and a Hugging Face detector such as DETR does it in
    its image processor, outside the model. Applying a second normalization here would
    double-normalize, so this modality passes ``[0, 1]`` pixels through unchanged.
    """

    def to_model_input(self, data: Tensor) -> Tensor:
        """Return the [0, 1] pixels unchanged."""
        return data

    def strip_model_input(self, model_input: Tensor) -> Tensor:
        """Return the model input unchanged."""
        return model_input
