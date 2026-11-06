#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Shared column-layout metadata for mixed continuous/categorical tabular attacks.

``TabularRepresentation`` (which optimizes the columns), ``TabularModality``
(which transforms them) and the evaluators (which score them) all need the same
continuous/categorical layout; ``TabularSchema`` collects it into one value
object so presets build it once instead of repeating the same arguments.

Two encodings, one canonical view
---------------------------------
Tabular attacks in this framework produce rows in one of two encodings:

* **one-hot** — each categorical feature occupies a contiguous block of logits
  (``categorical_col_groups``).  Produced by ``TabularRepresentation``, because
  gradient descent needs a differentiable relaxation of a discrete choice.
* **scalar-coded** — each categorical feature is a single integer-valued column
  (``categorical_scalar_cols``), for attacks that work directly in the
  original column space.

Scoring is identical in both cases once you collapse each one-hot block to its
argmax, so ``to_scalar_codes()`` defines that collapse and every evaluator works
in the resulting canonical scalar space.  Column order in that space is:
continuous columns first (in ``continuous_cols`` order), then one column per
categorical feature (in ``categorical_col_groups`` / ``categorical_scalar_cols``
order).  ``scalar_feature_names`` reports that order.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass(frozen=True)
class TabularSchema:
    """Column layout for a mixed continuous/categorical feature vector.

    Exactly one of ``categorical_col_groups`` / ``categorical_scalar_cols`` may
    be non-empty — they are two encodings of the same thing, and a schema that
    declared both would have an ambiguous ``num_features``.

    Args:
        num_features: Total number of columns F in the feature vector
            (including all one-hot expanded categorical blocks).
        continuous_cols: Indices of continuous feature columns in the F-dim vector.
        categorical_col_groups: One-hot encoding. One entry per categorical
            feature as ``(start_index, num_categories)``; the block
            ``[start_index : start_index + num_categories]`` holds the logits /
            one-hot values for that feature.
        categorical_scalar_cols: Scalar encoding. Indices of columns that hold an
            integer category code rather than a continuous value.
        col_means: Mean for each continuous column, keyed by column index.
        col_stds: Std for each continuous column, keyed by column index.
        feature_names: Optional display names, one per column of the F-dim
            vector.  Used for per-column metric reporting; when omitted,
            evaluators fall back to positional names.

    """

    num_features: int
    continuous_cols: list[int] = field(default_factory=list)
    categorical_col_groups: list[tuple[int, int]] = field(default_factory=list)
    categorical_scalar_cols: list[int] = field(default_factory=list)
    col_means: dict[int, float] = field(default_factory=dict)
    col_stds: dict[int, float] = field(default_factory=dict)
    feature_names: list[str] | None = None

    def __post_init__(self) -> None:  # noqa: C901 -- one sequential routine
        """Validate the layout at construction.

        Column indices here are load-bearing in three separate modules, and a
        mismatch (an off-by-one block start, an overlapping range, a stale
        ``num_features``) shows up much later as a quietly wrong metric rather
        than an exception.  Checking once at construction is the only place the
        whole layout is visible at the same time.
        """
        if self.categorical_col_groups and self.categorical_scalar_cols:
            raise ValueError(
                "TabularSchema accepts either categorical_col_groups (one-hot) or "
                "categorical_scalar_cols (scalar codes), not both — they are two "
                "encodings of the same features and imply different num_features."
            )

        seen: dict[int, str] = {}

        def claim(idx: int, owner: str) -> None:
            if not 0 <= idx < self.num_features:
                raise ValueError(
                    f"TabularSchema: {owner} references column {idx}, which is "
                    f"outside [0, num_features={self.num_features})."
                )
            if idx in seen:
                raise ValueError(
                    f"TabularSchema: column {idx} is claimed by both {seen[idx]} "
                    f"and {owner}; each column belongs to exactly one feature."
                )
            seen[idx] = owner

        for i in self.continuous_cols:
            claim(i, "continuous_cols")
        for g, (start, n) in enumerate(self.categorical_col_groups):
            if n < 1:
                raise ValueError(
                    f"TabularSchema: categorical_col_groups[{g}] has "
                    f"num_categories={n}; must be >= 1."
                )
            for i in range(start, start + n):
                claim(i, f"categorical_col_groups[{g}]")
        for i in self.categorical_scalar_cols:
            claim(i, "categorical_scalar_cols")

        if self.feature_names is not None and len(self.feature_names) != self.num_features:
            raise ValueError(
                f"TabularSchema: feature_names has {len(self.feature_names)} entries "
                f"but num_features is {self.num_features}."
            )

    # ── Canonical scalar-code view ────────────────────────────────────────────

    @property
    def is_one_hot(self) -> bool:
        """Whether any categorical column is encoded as a one-hot block."""
        return bool(self.categorical_col_groups)

    @property
    def num_scalar_features(self) -> int:
        """Width of the canonical scalar-code space."""
        return len(self.continuous_cols) + len(self.categorical_col_groups) + len(
            self.categorical_scalar_cols
        )

    @property
    def scalar_categorical_idx(self) -> list[int]:
        """Indices, in canonical scalar space, of the categorical columns."""
        n_cont = len(self.continuous_cols)
        n_cat = len(self.categorical_col_groups) + len(self.categorical_scalar_cols)
        return list(range(n_cont, n_cont + n_cat))

    @property
    def scalar_feature_names(self) -> list[str]:
        """Display names in canonical scalar-space order."""
        names = self.feature_names
        if names is None:
            return (
                [f"cont{i}" for i in self.continuous_cols]
                + [f"cat{g}" for g in range(len(self.categorical_col_groups))]
                + [f"cat{i}" for i in self.categorical_scalar_cols]
            )
        return (
            [names[i] for i in self.continuous_cols]
            + [names[start] for start, _ in self.categorical_col_groups]
            + [names[i] for i in self.categorical_scalar_cols]
        )

    def to_scalar_codes(self, x: Tensor) -> Tensor:
        """Collapse a ``[..., num_features]`` tensor to ``[..., num_scalar_features]``.

        Continuous columns are copied through; each one-hot block becomes its
        argmax index.  Scalar-coded schemas are already canonical apart from
        column reordering.
        """
        if x.shape[-1] != self.num_features:
            raise ValueError(
                f"Expected trailing dim {self.num_features} to match the schema, "
                f"got tensor with shape {tuple(x.shape)}."
            )

        parts: list[Tensor] = []
        if self.continuous_cols:
            parts.append(x[..., self.continuous_cols].float())
        for start, n in self.categorical_col_groups:
            parts.append(x[..., start : start + n].argmax(dim=-1, keepdim=True).float())
        if self.categorical_scalar_cols:
            parts.append(x[..., self.categorical_scalar_cols].float())

        if not parts:
            return x.new_zeros((*x.shape[:-1], 0), dtype=torch.float32)
        return torch.cat(parts, dim=-1)

    def scalar_col_stds(self, reference: Tensor) -> Tensor:
        """Per-column std in canonical scalar space, measured on ``reference``.

        Continuous tolerances are expressed as a multiple of the feature's own
        spread, so the spread has to come from real data rather than from
        ``col_stds`` (which is keyed by *raw* column index and is often absent
        for schemas built from normalized features).
        """
        std = reference.float().std(dim=0, unbiased=False)
        # A constant column has zero spread; any tolerance band around it is
        # then exact-match, which is the correct degenerate behaviour.
        return torch.nan_to_num(std, nan=0.0)
