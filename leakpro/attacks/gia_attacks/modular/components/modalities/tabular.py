#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Tabular modalities: data space <-> model input space.

Scoring lives in ``components/evaluators/tabular.py``.  The two modalities here
differ only in how categorical columns are encoded, which is why they need
different transforms but share a single evaluator.
"""
from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from leakpro.attacks.gia_attacks.modular.components.tabular.schema import TabularSchema
from leakpro.attacks.gia_attacks.modular.core.interfaces import Modality

CategoricalRelaxation = Literal["softmax", "raw"]


class TabularModality(Modality):
    """Mixed continuous/categorical data with one-hot categorical blocks.

    The client holds rows whose categorical blocks are one-hot; ``client_to_model_input`` feeds them to the
    model unchanged. The attacker's reconstruction holds a continuous relaxation of each block (logits), which
    ``to_model_input`` maps onto the probability simplex with a softmax, so a reconstruction can approach the
    client's one-hot rows.
    """

    def __init__(self, schema: TabularSchema, categorical_relaxation: CategoricalRelaxation = "softmax") -> None:
        """Store the schema and the categorical relaxation.

        ``categorical_relaxation``: ``"softmax"`` (default) feeds each one-hot block through
        a softmax; ``"raw"`` feeds the optimised values to the model unchanged (used by ``tabular_raw_onehot``).
        """
        if categorical_relaxation not in ("softmax", "raw"):
            raise ValueError("categorical_relaxation must be 'softmax' or 'raw'.")
        self.schema = schema
        self.categorical_relaxation = categorical_relaxation
        self.continuous_cols = schema.continuous_cols
        self.categorical_col_groups = schema.categorical_col_groups

    def to_model_input(self, data: Tensor) -> Tensor:
        """Softmax each one-hot categorical block, or pass the data through when the relaxation is raw."""
        if self.categorical_relaxation == "raw":
            return data
        out = data.clone()
        for start, n in self.categorical_col_groups:
            out[..., start : start + n] = torch.softmax(data[..., start : start + n], dim=-1)
        return out

    def client_to_model_input(self, data: Tensor) -> Tensor:
        """The client's real rows, unchanged: their one-hot blocks already are what the model expects."""
        return data

    def strip_model_input(self, model_input: Tensor) -> Tensor:
        """Return the model input unchanged.

        With the softmax relaxation this is not the inverse of ``to_model_input``: the softmax is not invertible,
        so the result holds category probabilities rather than the logits the representation optimises. Scoring
        is unaffected, because evaluators take the argmax of each block.
        """
        return model_input


class RawTabularModality(Modality):
    """Scalar-coded mixed tabular data with no transform in either direction.

    Any attack that already works in the victim's original, unnormalized
    column space uses this identity modality.
    """

    def to_model_input(self, data: Tensor) -> Tensor:
        """Return the data unchanged."""
        return data

    def strip_model_input(self, model_input: Tensor) -> Tensor:
        """Return the model input unchanged."""
        return model_input
