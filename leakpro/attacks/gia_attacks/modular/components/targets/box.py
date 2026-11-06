#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Target source that jointly optimizes box coordinates with oracle labels."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from leakpro.attacks.gia_attacks.modular.core.interfaces import TargetSource
from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.state import Context, State


class JointBoxTarget(TargetSource):
    """Jointly optimizes box coordinates alongside the image; labels are oracle.

    Why labels are not jointly optimized
    -------------------------------------
    A detection loss consumes ``labels`` as hard integer class indices (they
    select the class entry of a classification loss or the matched queries), so
    there is no gradient path from that loss back to the label *values*: an index
    is not a continuous quantity, unlike ``JointTarget``'s soft-label trick for a
    plain classifier, which works only because ``cross_entropy(logits, softmax(soft))``
    keeps ``soft`` continuous throughout. Box coordinates, by contrast, enter the
    box-regression loss continuously and can receive a real gradient (unless the
    criterion detaches them, as YOLO's target assignment does). So this class
    reconstructs boxes only; labels must be supplied (oracle) at construction.
    Recovering labels from gradients would need an analytic method rather than
    gradient descent on a label parameter, which is out of scope here.

    ``k_max`` boxes per image are always considered real (``box_mask`` is
    constant ``True``) — the attacker assumes exactly ``k_max`` objects per
    image up front rather than inferring the count: the true count is exactly
    what a good reconstruction would reveal, so it is an attacker-chosen capacity.

    Boxes are parameterized through a sigmoid into (cx, cy, w, h) pixel-space
    before conversion to xyxy, so width/height can never go negative and
    corners can never invert — clamping raw xyxy corners directly tends to
    collapse a degenerate box to zero area instead of correcting it.

    With a seed or epoch axis every field carries the axis prefix and boxes are initialised independently per
    slice; the oracle labels are shared by all slices.
    """

    def __init__(self, oracle_labels: Tensor, k_max: int) -> None:
        self.oracle_labels = oracle_labels
        self.k_max = k_max
        self._raw_boxes: Tensor | None = None
        self._image_h: int | None = None
        self._image_w: int | None = None
        self._prefix: tuple[int, ...] = ()

    def bind(self, state: State, ctx: Context) -> JointBoxTarget:
        """Return a copy with box logits carried over from the state or freshly initialized per axis slice."""
        bound = copy.copy(self)
        n = ctx.input_shape[0]
        if tuple(self.oracle_labels.shape) != (n, self.k_max):
            raise ValueError(f"oracle_labels must have shape (N={n}, k_max={self.k_max}), got {tuple(self.oracle_labels.shape)}.")
        prefix = tuple(ctx.axes.shape)
        bound._prefix = prefix
        bound._image_h = ctx.input_shape[-2]
        bound._image_w = ctx.input_shape[-1]

        if state.target is not None and state.target.get("raw_boxes") is not None:
            raw = state.target.raw_boxes.detach().clone().to(ctx.device)
            if tuple(raw.shape[-3:]) != (n, self.k_max, 4):
                raise ValueError(f"carried-over raw_boxes must end in (N={n}, k_max={self.k_max}, 4), got {tuple(raw.shape)}.")
            carried = tuple(raw.shape[:-3])
            if carried != prefix:
                if carried:  # an axis prefix we cannot map onto the current one: refuse rather than mis-shape
                    raise ValueError(
                        f"carried-over raw_boxes have axis prefix {carried} but this stage has {prefix}; "
                        "aggregate the axes (or keep the same prefix) between stages."
                    )
                # a previous stage collapsed its axes: every slice restarts from the aggregate
                raw = raw.expand(*prefix, *raw.shape).clone()
            bound._raw_boxes = raw.requires_grad_(True)
            return bound

        raw = torch.randn((*prefix, n, bound.k_max, 4), device=ctx.device)
        # Bias the width/height logits toward a larger starting box.  A detector's
        # loss only gives box coordinates a gradient through anchors matched as
        # foreground (IoU above the matcher's threshold) — an N(0,1) init maps
        # through sigmoid to width/height around half the image, but combined
        # with a random center it can still land a box that overlaps no anchor
        # above threshold, leaving it a dead parameter (zero gradient) for the
        # entire run. +1.0 pushes sigmoid(w/h logit) to ~0.73 of the image
        # dimension, large enough to reliably clear the matcher's threshold.
        raw[..., 2:] += 1.0
        bound._raw_boxes = raw.requires_grad_(True)
        return bound

    def _decode_boxes(self, raw: Tensor | None = None) -> Tensor:
        raw = self._raw_boxes if raw is None else raw
        cxcy = torch.sigmoid(raw[..., :2])
        # clamp_min avoids a zero-area box the optimizer can never escape
        # (gradient of a zero-width box w.r.t. its own width is zero at 0).
        wh = torch.sigmoid(raw[..., 2:]).clamp_min(1e-3)
        cx = cxcy[..., 0] * self._image_w
        cy = cxcy[..., 1] * self._image_h
        w = wh[..., 0] * self._image_w
        h = wh[..., 1] * self._image_h
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        # A centered box can still extend past the canvas even with cx/cy
        # in-bounds; clamp corners to the image so reconstructed boxes stay
        # meaningful to score. clamp_min(1e-3) on w/h above keeps x2 > x1
        # except in the boundary case where both corners clamp to the same edge.
        x1 = x1.clamp(0.0, self._image_w)
        x2 = x2.clamp(0.0, self._image_w)
        y1 = y1.clamp(0.0, self._image_h)
        y2 = y2.clamp(0.0, self._image_h)
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def current(self) -> TargetInfo:
        """Return the current boxes, the oracle labels and an all-true box mask."""
        boxes = self._decode_boxes()
        labels = self.oracle_labels.to(boxes.device).expand(*self._prefix, *self.oracle_labels.shape)
        mask = torch.ones(boxes.shape[:-1], dtype=torch.bool, device=boxes.device)
        return TargetInfo(
            boxes=boxes,
            box_labels=labels,
            box_mask=mask,
            raw_boxes=self._raw_boxes,
            axis_ndim=len(self._prefix),
        )

    def reduce(self, target: TargetInfo, dim: int, method: str) -> TargetInfo:
        """Collapse a parallel axis: ``"mean"`` averages the float fields (boxes), ``"match_epoch_0"`` keeps slice 0.

        Integer and boolean fields (oracle labels, mask) are identical across the axis, so slice 0 stands for all.
        """
        if method == "mean":
            return target.map(
                lambda t: t.mean(dim=dim) if t.is_floating_point() else t.select(dim, 0),
                axis_ndim=max(target.axis_ndim - 1, 0),
            )
        if method == "match_epoch_0":
            return target.index_axis(dim, 0)
        raise ValueError(f"JointBoxTarget.reduce: unsupported method {method!r}.")

    def slice(self, target: TargetInfo, dim: int, indices: int | Tensor) -> TargetInfo:
        """Select indices along an axis from every field of the target, re-decoding boxes from the sliced raw logits."""
        sliced = target.index_axis(dim, indices)
        if not isinstance(indices, int) or self._image_h is None or sliced.get("raw_boxes") is None:
            return sliced
        # Decode the slice's boxes from its own raw parameters: slices of one shared decode would each
        # backpropagate through the same sigmoid graph, which the first backward frees.
        return TargetInfo(**{**sliced.__dict__, "boxes": self._decode_boxes(sliced.raw_boxes)})

    def parameters(self) -> list[dict]:
        """Return the box logits as the single optimizable parameter group."""
        return [{"name": "box_coords", "params": [self._raw_boxes], "lr_scale": 1.0}]
