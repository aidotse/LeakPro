#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Padding helpers for ragged per-image detection targets.

A batch of ``N`` images has a variable number of boxes per image (``k_i``).
``TargetInfo`` (``core/state.py``) only ever applies its transforms
(``map``/``index_axis``/``to``/...) to ``Tensor`` fields, and passes any
non-``Tensor`` field through untouched — so a ``list[Tensor]`` of per-image
boxes would survive ``index_axis`` on the sample dimension *unsliced*: a
minibatch selection would silently pair batch-``b`` images with the *full*
batch's boxes.

The fix is to never hand TargetInfo a ragged field in the first place. Boxes,
labels, and a validity mask are padded up front to a fixed ``k_max`` (the
attacker's assumed maximum object count — already a required assumption,
since the true count is exactly what's being reconstructed) and carried as
plain ``[N, k_max, ...]`` tensors, which the existing per-sample-dim slicing
already handles correctly.
"""
from __future__ import annotations

import torch
from torch import Tensor


def pad_boxes(
    boxes_list: list[Tensor],
    labels_list: list[Tensor],
    k_max: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pad a list of per-image ``[k_i, 4]``/``[k_i]`` tensors to ``[N, k_max, ...]``.

    Args:
        boxes_list: Per-image box tensors, each ``[k_i, 4]`` (any box format —
            padding is format-agnostic).
        labels_list: Per-image label tensors, each ``[k_i]``, same ``k_i`` as
            the matching ``boxes_list`` entry.
        k_max: Fixed capacity to pad to. Raises if any image has more than
            ``k_max`` boxes — silently truncating would misreport ground truth.

    Returns:
        ``(boxes[N, k_max, 4], labels[N, k_max], mask[N, k_max])`` — ``mask`` is
        ``True`` for real boxes and ``False`` for padding. Padded box/label
        entries are zero-filled; callers must gate on ``mask``, not on the
        padded values, since zero is a valid label/coordinate.

    """
    if len(boxes_list) != len(labels_list):
        raise ValueError(
            f"boxes_list and labels_list must have the same length, got "
            f"{len(boxes_list)} and {len(labels_list)}."
        )
    n = len(boxes_list)
    device = boxes_list[0].device if n > 0 else torch.device("cpu")
    dtype = boxes_list[0].dtype if n > 0 else torch.float32

    boxes = torch.zeros((n, k_max, 4), dtype=dtype, device=device)
    labels = torch.zeros((n, k_max), dtype=torch.long, device=device)
    mask = torch.zeros((n, k_max), dtype=torch.bool, device=device)

    for i, (b, lbl) in enumerate(zip(boxes_list, labels_list)):
        k_i = b.shape[0]
        if k_i > k_max:
            raise ValueError(f"Image {i} has {k_i} boxes, exceeding k_max={k_max}.")
        if k_i == 0:
            continue
        boxes[i, :k_i] = b
        labels[i, :k_i] = lbl
        mask[i, :k_i] = True

    return boxes, labels, mask


def unpad_boxes(
    boxes: Tensor,
    labels: Tensor,
    mask: Tensor,
) -> tuple[list[Tensor], list[Tensor]]:
    """Inverse of ``pad_boxes`` — strip padding back to a per-image ragged list.

    Args:
        boxes: ``[N, k_max, 4]``.
        labels: ``[N, k_max]``.
        mask: ``[N, k_max]`` boolean validity mask.

    Returns:
        ``(boxes_list, labels_list)`` — per-image ``[k_i, 4]``/``[k_i]`` tensors,
        in the same box format they were padded with.

    """
    boxes_list = [boxes[i][mask[i]] for i in range(boxes.shape[0])]
    labels_list = [labels[i][mask[i]] for i in range(labels.shape[0])]
    return boxes_list, labels_list


def assert_box_target_valid(target) -> None:  # noqa: ANN001 - TargetInfo, kept untyped to avoid a core import
    """Raise if ``target`` carries ``boxes`` without an accompanying ``box_mask``.

    A ``box_mask`` is what makes padding safe — every consumer (loss, evaluator,
    representation) must gate on it rather than on padded zero values, since
    zero is a legitimate label/coordinate. Call this at the boundary where a
    detection ``TargetInfo`` is constructed or received, not per-consumer.
    """
    boxes = target.get("boxes")
    if boxes is None:
        return
    mask = target.get("box_mask")
    if mask is None:
        raise ValueError(
            "TargetInfo carries 'boxes' but no 'box_mask' — padded box tensors "
            "must always travel with their validity mask."
        )
    if mask.shape != boxes.shape[:-1]:
        raise ValueError(
            f"box_mask shape {tuple(mask.shape)} does not match boxes' leading "
            f"dims {tuple(boxes.shape[:-1])}."
        )


def to_yolo_rows(target, img_size: int) -> Tensor:  # noqa: ANN001 - TargetInfo, kept untyped to avoid a core import
    """Padded ``[N, k_max, ...]`` boxes/labels/mask as YOLO's flat ``[total_real_boxes, 6]`` rows.

    Columns are ``(image_idx, class, cx, cy, w, h)`` with box coordinates normalized to ``[0, 1]``
    by ``img_size``: the target format of YOLOv5/YOLOv8-style ``ComputeLoss`` implementations.
    """
    assert_box_target_valid(target)
    boxes, labels, mask = target.boxes, target.box_labels, target.box_mask
    device = boxes.device
    rows = []
    for i in range(boxes.shape[0]):
        m = mask[i]
        b = boxes[i][m]
        if b.numel() == 0:
            continue
        lbl = labels[i][m].to(b.dtype)
        cx = (b[:, 0] + b[:, 2]) / 2.0 / img_size
        cy = (b[:, 1] + b[:, 3]) / 2.0 / img_size
        w = (b[:, 2] - b[:, 0]) / img_size
        h = (b[:, 3] - b[:, 1]) / img_size
        img_idx = torch.full((b.shape[0],), float(i), device=device, dtype=b.dtype)
        rows.append(torch.stack([img_idx, lbl, cx, cy, w, h], dim=1))
    if not rows:
        return torch.zeros((0, 6), device=device)
    return torch.cat(rows, dim=0)


def xyxy_to_cxcywh_normalized(boxes: Tensor, img_size: int) -> Tensor:
    """``[k, 4]`` absolute-pixel xyxy as ``[k, 4]`` ``img_size``-normalized cxcywh (DETR's label box format)."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2.0 / img_size
    cy = (y1 + y2) / 2.0 / img_size
    w = (x2 - x1) / img_size
    h = (y2 - y1) / img_size
    return torch.stack([cx, cy, w, h], dim=-1)


def to_torchvision_targets(target) -> list[dict[str, Tensor]]:  # noqa: ANN001 - TargetInfo, kept untyped to avoid a core import
    """Padded box target as torchvision's per-image ``[{"boxes": [k, 4] xyxy, "labels": [k]}]``."""
    assert_box_target_valid(target)
    boxes_list, labels_list = unpad_boxes(target.boxes, target.box_labels, target.box_mask)
    return [{"boxes": b, "labels": lbl.long()} for b, lbl in zip(boxes_list, labels_list)]


def to_detr_labels(target, img_size: int) -> list[dict[str, Tensor]]:  # noqa: ANN001 - TargetInfo, kept untyped to avoid a core import
    """Padded box target as HuggingFace DETR ``labels``: ``img_size``-normalized cxcywh boxes and class ids."""
    assert_box_target_valid(target)
    boxes_list, labels_list = unpad_boxes(target.boxes, target.box_labels, target.box_mask)
    return [
        {"class_labels": lbl.long(), "boxes": xyxy_to_cxcywh_normalized(b, img_size)}
        for b, lbl in zip(boxes_list, labels_list)
    ]
