#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Box/label reconstruction quality via per-image Hungarian IoU matching."""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from leakpro.attacks.gia_attacks.modular.components.utils.matching import hungarian_match
from leakpro.attacks.gia_attacks.modular.core.evaluation import EvalResult
from leakpro.attacks.gia_attacks.modular.core.interfaces import Evaluator
from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo
from leakpro.utils.logger import logger

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.state import Context, State


class BoxEvaluator(Evaluator):
    """Scores reconstructed boxes/labels against ground truth.

    Ground truth annotations are not part of the attack's own state: they are
    only needed for scoring, the same role ``ImageEvaluator``'s ``target``
    argument plays for pixels. Pass them as ``ground_truth`` (a ``TargetInfo`` or
    a mapping with ``boxes``, ``labels`` and an optional ``mask``) or set
    ``state.artifacts[GT_KEY]`` before calling ``evaluate()``. Reconstructed
    boxes/labels come from ``state.target`` (the attack's final ``TargetInfo``).

    Missing ground truth degrades to an empty ``EvalResult`` plus a warning
    rather than raising: this evaluator composes inside a ``CompositeEvaluator``
    alongside ``ImageEvaluator``, and absent box ground truth should not block
    the pixel score.

    A box only counts as recovered if its IoU with some ground-truth box (under
    the assignment that maximizes total IoU) clears ``iou_threshold`` — matching
    on proximity alone, with no threshold, would credit a barely-overlapping box.
    """

    GT_KEY = "detection_ground_truth"

    def __init__(self, iou_threshold: float = 0.5, ground_truth: Mapping[str, Tensor] | TargetInfo | None = None) -> None:
        self.iou_threshold = iou_threshold
        self.ground_truth = ground_truth

    def _resolve_ground_truth(self, state: State | None) -> Mapping[str, Tensor] | None:
        gt = self.ground_truth
        if gt is None and state is not None:
            gt = (state.artifacts or {}).get(self.GT_KEY)
        if isinstance(gt, TargetInfo):
            gt = {"boxes": gt.boxes, "labels": gt.box_labels, "mask": gt.box_mask}
        return gt

    def evaluate(
        self,
        recon: Tensor,
        target: Tensor,  # noqa: ARG002 -- interface signature
        *,
        ctx: Context | None = None,  # noqa: ARG002 -- interface signature
        state: State | None = None,
    ) -> EvalResult:
        """Matches reconstructed boxes to ground-truth boxes and reports box metrics."""
        gt = self._resolve_ground_truth(state)
        if gt is None:
            logger.warning("BoxEvaluator: no ground truth (pass ground_truth=... or set state.artifacts[GT_KEY]); skipping.")
            return EvalResult()
        if state is None or state.target is None:
            return EvalResult()

        if state.target.boxes.ndim != 3:
            raise ValueError(
                f"BoxEvaluator expects per-image boxes of shape (N, k_max, 4) but got {tuple(state.target.boxes.shape)}: "
                "the target still carries an axis prefix (seed/epoch); aggregate it first."
            )

        from torchvision.ops import box_iou  # noqa: PLC0415 -- optional torchvision dependency, imported on use

        device = state.target.boxes.device
        recon_boxes, recon_labels, recon_mask = state.target.boxes, state.target.box_labels, state.target.box_mask
        gt_boxes, gt_labels = gt["boxes"].to(device), gt["labels"].to(device)
        gt_mask = gt.get("mask")
        gt_mask = gt_mask.to(device) if gt_mask is not None else torch.ones(gt_boxes.shape[:2], dtype=torch.bool, device=device)
        if recon_mask is None:
            recon_mask = torch.ones(recon_boxes.shape[:2], dtype=torch.bool, device=device)
        whole_image_box = torch.tensor([[0.0, 0.0, float(recon.shape[-1]), float(recon.shape[-2])]], device=device)
        n = recon_boxes.shape[0]

        total_tp = total_fp = total_fn = 0
        matched_ious: list[float] = []
        label_hits: list[bool] = []
        baseline_ious: list[float] = []
        n_true = n_recon = 0

        for i in range(n):
            gt_b, gt_l = gt_boxes[i][gt_mask[i]], gt_labels[i][gt_mask[i]]
            pred_b, pred_l = recon_boxes[i][recon_mask[i]].detach(), recon_labels[i][recon_mask[i]]

            n_true += gt_b.shape[0]
            n_recon += pred_b.shape[0]

            if gt_b.numel() > 0:
                baseline_ious.extend(box_iou(whole_image_box, gt_b.detach())[0].tolist())

            if gt_b.numel() == 0 and pred_b.numel() == 0:
                continue
            if gt_b.numel() == 0:
                total_fp += pred_b.shape[0]
                continue
            if pred_b.numel() == 0:
                total_fn += gt_b.shape[0]
                continue

            iou = box_iou(pred_b, gt_b.detach())
            row_ind, col_ind = hungarian_match(1.0 - iou)
            matched_iou = iou[row_ind, col_ind]
            hits = matched_iou >= self.iou_threshold
            n_hits = int(hits.sum().item())

            total_tp += n_hits
            total_fp += pred_b.shape[0] - n_hits
            total_fn += gt_b.shape[0] - n_hits
            matched_ious.extend(matched_iou[hits].tolist())
            label_hits.extend((pred_l[row_ind][hits] == gt_l[col_ind][hits]).tolist())

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return EvalResult(
            metrics={
                "precision_at_iou": precision,
                "recall_at_iou": recall,
                "f1_at_iou": f1,
                "mean_matched_iou": sum(matched_ious) / len(matched_ious) if matched_ious else 0.0,
                "label_accuracy_on_matches": sum(label_hits) / len(label_hits) if label_hits else 0.0,
                "baseline_whole_image_mean_iou": sum(baseline_ious) / len(baseline_ious) if baseline_ious else 0.0,
                "n_true_boxes": n_true,
                "n_recon_boxes": n_recon,
            }
        )
