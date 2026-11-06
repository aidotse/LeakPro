#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Tabular representation: direct optimization of a [*axes, N, F] parameter tensor."""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor

from leakpro.attacks.gia_attacks.modular.components.tabular.schema import TabularSchema
from leakpro.attacks.gia_attacks.modular.core.interfaces import Representation

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.state import Context, State


CatInit = Literal["small", "randn"]


class TabularRepresentation(Representation):
    """Optimizes a flat [*axes, N, F] tensor for tabular gradient inversion.

    Continuous columns are initialized from N(0, 1) (normalized space).
    Categorical logit blocks are initialized from N(0, 0.1) so that softmax
    starts near-uniform, avoiding biased initial reconstructions.

    Args:
        schema: Column layout shared with ``TabularModality`` — see
            ``components/tabular/schema.py::TabularSchema``.

    """

    def __init__(self, schema: TabularSchema, cat_init: CatInit = "small") -> None:
        """Store the column layout; ``cat_init``: ``"small"`` (logits from N(0, 0.1)) or ``"randn"`` (N(0, 1))."""
        if cat_init not in ("small", "randn"):
            raise ValueError("cat_init must be 'small' or 'randn'.")
        self.cat_init_scale = 0.1 if cat_init == "small" else 1.0
        self.num_features = schema.num_features
        self.continuous_cols: list[int] = schema.continuous_cols
        self.categorical_col_groups: list[tuple[int, int]] = schema.categorical_col_groups
        self._param: Tensor | None = None

    def bind(self, state: State, ctx: Context) -> TabularRepresentation:
        """Copy resuming from state.input or the "tabular" artifact, else initialized from per-column Gaussian noise."""
        bound = copy.copy(self)
        if state.input is not None:
            bound._param = state.input.detach().clone().requires_grad_(True)
            return bound
        if "tabular" in state.artifacts:
            bound._param = state.artifacts["tabular"].detach().clone().requires_grad_(True)
            return bound

        # Fresh initialization: [*axes, N, F]
        param = torch.zeros(*ctx.axes.shape, *ctx.input_shape, device=ctx.device)

        # Continuous columns: N(0, 1) — works well in normalized feature space.
        if bound.continuous_cols:
            param[..., bound.continuous_cols] = torch.randn_like(
                param[..., bound.continuous_cols]
            )

        # Categorical logit blocks: N(0, 0.1) — near-uniform after softmax.
        for start, n in bound.categorical_col_groups:
            param[..., start : start + n] = (
                torch.randn_like(param[..., start : start + n]) * bound.cat_init_scale
            )

        # Fall back to N(0, 1) for any columns not explicitly categorized.
        categorized = set(bound.continuous_cols)
        for start, n in bound.categorical_col_groups:
            categorized.update(range(start, start + n))
        uncategorized = [i for i in range(bound.num_features) if i not in categorized]
        if uncategorized:
            param[..., uncategorized] = torch.randn_like(param[..., uncategorized])

        bound._param = param.requires_grad_(True)
        return bound

    def parameters(self) -> list[dict]:
        """The single "tabular" parameter group."""
        return [{"name": "tabular", "params": [self._param], "lr_scale": 1.0}]

    def decode(self) -> Tensor:
        """The raw feature tensor."""
        return self._param

    def encode(self, data: Tensor) -> Tensor:
        """Identity; data space is returned unchanged."""
        return data

    @torch.no_grad()
    def project(self, ctx: Context | None = None) -> None:
        # No projection needed: continuous columns are unconstrained in normalized
        # space; categorical logits are unconstrained (softmax applied in modality).
        """No-op; columns and categorical logits are unconstrained."""
        pass

    def to_observable(self) -> Tensor:
        """The feature tensor, detached."""
        return self._param.detach()

    def artifacts(self) -> dict[str, Any]:
        """The feature tensor under "tabular", or nothing before binding."""
        return {"tabular": self._param.detach()} if self._param is not None else {}
