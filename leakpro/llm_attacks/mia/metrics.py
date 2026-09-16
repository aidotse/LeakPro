#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
	fpr_values, tpr_values, _ = roc_curve(labels.astype(int), scores.astype(float), pos_label=1)
	if fpr_values.size == 0:
		return float("nan")
	target_fpr = float(np.clip(target_fpr, 0.0, 1.0))
	return float(np.interp(target_fpr, np.clip(fpr_values, 0.0, 1.0), np.clip(tpr_values, 0.0, 1.0)))
