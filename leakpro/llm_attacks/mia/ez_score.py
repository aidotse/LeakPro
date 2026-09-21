#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import torch


def _apply_mask_1d(values: torch.Tensor, mask: torch.Tensor, ignore_bos: bool) -> tuple[torch.Tensor, torch.Tensor]:
	values_flat = values
	mask_flat = mask
	if values_flat.ndim >= 2:
		if values_flat.ndim == 2:
			values_flat = values_flat.reshape(-1)
		elif values_flat.ndim >= 3:
			batch_size, seq_len = values_flat.shape[0], values_flat.shape[1]
			values_flat = values_flat.reshape(batch_size * seq_len, *values_flat.shape[2:])
	if mask_flat.ndim == 2:
		mask_flat = mask_flat.reshape(-1)
	mask_flat = mask_flat.to(torch.float32)
	if ignore_bos and mask_flat.numel() > 0:
		mask_flat = mask_flat.clone(); mask_flat[0] = 0.0
	return values_flat, mask_flat


def error_zone_pos_neg_sum_ratio(
	tgt_correct_log_probs: torch.Tensor,
	ref_correct_log_probs: torch.Tensor,
	top1_indices_tgt: torch.Tensor,
	target_ids: torch.Tensor,
	mask: torch.Tensor,
	*,
	ignore_bos: bool = True,
	min_tokens: int = 2,
) -> float:
	"""Positive vs negative sum ratio restricted to error-zone tokens (target wrong).

	Normalizes global improvement dominance specifically where target struggles.
	"""
	target_correct_lp, mask_flat = _apply_mask_1d(tgt_correct_log_probs, mask, ignore_bos)
	ref_correct_lp, _ = _apply_mask_1d(ref_correct_log_probs, mask, ignore_bos)
	top1_indices_flat = top1_indices_tgt.reshape(-1)
	targets_flat = target_ids.reshape(-1)
	base_mask = (mask_flat > 0.5)
	error_mask = ((top1_indices_flat != targets_flat) & base_mask)
	if error_mask.sum().item() < min_tokens:
		return 0.0
	deltas = (target_correct_lp - ref_correct_lp)[error_mask]
	positive_sum = deltas[deltas > 0].sum()
	negative_sum = deltas[deltas < 0].abs().sum()
	if negative_sum.item() == 0:
		return 0.0
	value = float((positive_sum / (negative_sum + 1e-12)).item())
	if not (value == value) or value < 0:
		return 0.0
	return value
