#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Label type abstractions for different task formulations.

This module provides a small hierarchy that encapsulates all label-shape
operations so that downstream components (orchestrator, optimizer, training
simulator) can handle single-label classification and binary multi-label tasks
through a single, consistent interface.

Supported types
---------------
ClassificationLabelType
    Single-label integer class indices, shape ``[N]`` (``dtype=torch.long``).
    Typical use: CIFAR, ImageNet classification.

BinaryMultilabelType
    Binary float attribute vectors, shape ``[N, K]`` (``dtype=torch.float``).
    Typical use: CelebA 40-attribute prediction.

Adding a new type
-----------------
Subclass :class:`LabelType`, implement all abstract methods, and set the
``label_type`` field on the :class:`~leakpro.attacks.gia_attacks.modular.core.
component_base.LabelInferenceResult` returned by your inference strategy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch.nn.functional import one_hot


class LabelType(ABC):
    """Abstract base class for label representation types.

    A *LabelType* object encodes the rules for a particular label space:
    how tensors are shaped, how to add / remove an epoch prefix, and how to
    produce a confidence matrix.

    Subclass and implement all abstract methods to support a new label space
    (e.g. ordinal regression, multi-class probabilities).
    """

    # ------------------------------------------------------------------
    # Identity / validation
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier."""

    @property
    @abstractmethod
    def bare_ndim(self) -> int:
        """Number of dimensions when *no* epoch prefix is present.

        Examples:
            * ``ClassificationLabelType``:  1  (shape ``[N]``)
            * ``BinaryMultilabelType``:     2  (shape ``[N, K]``)
        """

    @abstractmethod
    def validate(self, labels: torch.Tensor) -> None:
        """Raise ``ValueError`` if *labels* is incompatible with this type."""

    # ------------------------------------------------------------------
    # Epoch-dimension helpers
    # ------------------------------------------------------------------

    @abstractmethod
    def add_epoch_dim(self, labels: torch.Tensor) -> torch.Tensor:
        """Insert a singleton epoch dimension at position 0.

        ``[N]    → [1, N]``    (classification)
        ``[N, K] → [1, N, K]`` (multi-label)
        """

    @abstractmethod
    def strip_epoch_dim(self, labels: torch.Tensor) -> torch.Tensor:
        """Remove the leading epoch dimension.

        ``[E, N]    → [N]``    (classification)
        ``[E, N, K] → [N, K]`` (multi-label)
        """

    @abstractmethod
    def expand_for_epochs(self, labels: torch.Tensor, num_epochs: int) -> torch.Tensor:
        """Tile along a new leading epoch axis.

        ``[N]    → [E, N]``    (classification)
        ``[N, K] → [E, N, K]`` (multi-label)
        """

    # ------------------------------------------------------------------
    # Confidence / probability matrix
    # ------------------------------------------------------------------

    @abstractmethod
    def to_confidence(
        self,
        labels: torch.Tensor,
        num_classes: int | None = None,
    ) -> torch.Tensor:
        """Return a ``float`` confidence / probability matrix.

        Args:
            labels:      Raw label tensor (bare, no epoch prefix).
            num_classes: Number of classes.  Required for
                         :class:`ClassificationLabelType` when the value
                         cannot be inferred from ``labels.max()``.

        Returns:
            Float tensor of shape ``[N, C]`` or ``[N, K]``.
        """

    # ------------------------------------------------------------------
    # Auto-detection
    # ------------------------------------------------------------------

    @staticmethod
    def auto_detect(labels: torch.Tensor) -> "LabelType":
        """Infer the correct :class:`LabelType` from a tensor's dtype and shape.

        Rules:
            * 2-D (or higher) **float** tensor → :class:`BinaryMultilabelType`
            * Anything else                    → :class:`ClassificationLabelType`
        """
        if labels.dtype.is_floating_point and labels.ndim >= 2:
            return BinaryMultilabelType(num_classes=labels.shape[-1])
        return ClassificationLabelType()


class ClassificationLabelType(LabelType):
    """Single-label integer class indices.

    Expected shape: ``[N]``, ``dtype=torch.long``.
    """

    @property
    def name(self) -> str:
        return "classification"

    @property
    def bare_ndim(self) -> int:
        return 1

    def validate(self, labels: torch.Tensor) -> None:
        if labels.ndim != 1:
            raise ValueError(
                f"ClassificationLabelType expects 1-D [N] labels; "
                f"got shape {tuple(labels.shape)}"
            )

    def add_epoch_dim(self, labels: torch.Tensor) -> torch.Tensor:
        return labels.unsqueeze(0)  # [N] → [1, N]

    def strip_epoch_dim(self, labels: torch.Tensor) -> torch.Tensor:
        return labels[0]  # [E, N] → [N]

    def expand_for_epochs(self, labels: torch.Tensor, num_epochs: int) -> torch.Tensor:
        return labels.unsqueeze(0).expand(num_epochs, -1).contiguous()  # [N] → [E, N]

    def to_confidence(
        self,
        labels: torch.Tensor,
        num_classes: int | None = None,
    ) -> torch.Tensor:
        labels = labels.long()
        n_cls = num_classes if num_classes is not None else int(labels.max().item()) + 1
        return one_hot(labels, n_cls).float()


class BinaryMultilabelType(LabelType):
    """Binary float attribute vectors.

    Expected shape: ``[N, K]``, ``dtype=torch.float``.

    Args:
        num_classes: Number of binary attributes *K*.  If ``None`` the value
                     is inferred from ``labels.shape[-1]`` at first use.
    """

    def __init__(self, num_classes: int | None = None) -> None:
        self._num_classes = num_classes

    @property
    def num_classes(self) -> int | None:
        return self._num_classes

    @property
    def name(self) -> str:
        k = self._num_classes if self._num_classes is not None else "?"
        return f"binary_multilabel(K={k})"

    @property
    def bare_ndim(self) -> int:
        return 2

    def validate(self, labels: torch.Tensor) -> None:
        if labels.ndim != 2:
            raise ValueError(
                f"BinaryMultilabelType expects 2-D [N, K] labels; "
                f"got shape {tuple(labels.shape)}"
            )
        if not labels.dtype.is_floating_point:
            raise ValueError(
                f"BinaryMultilabelType expects float labels; got dtype={labels.dtype}"
            )

    def add_epoch_dim(self, labels: torch.Tensor) -> torch.Tensor:
        return labels.unsqueeze(0)  # [N, K] → [1, N, K]

    def strip_epoch_dim(self, labels: torch.Tensor) -> torch.Tensor:
        return labels[0]  # [E, N, K] → [N, K]

    def expand_for_epochs(self, labels: torch.Tensor, num_epochs: int) -> torch.Tensor:
        return labels.unsqueeze(0).expand(num_epochs, -1, -1).contiguous()  # [N, K] → [E, N, K]

    def to_confidence(
        self,
        labels: torch.Tensor,
        num_classes: int | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        # The label tensor itself is already the confidence representation.
        return labels.float()


__all__ = [
    "LabelType",
    "ClassificationLabelType",
    "BinaryMultilabelType",
]
