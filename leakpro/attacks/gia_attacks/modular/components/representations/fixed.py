#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""A representation that holds a fixed data tensor with no learnable parameters."""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from torch import Tensor

from leakpro.attacks.gia_attacks.modular.core.interfaces import Representation

if TYPE_CHECKING:
    from leakpro.attacks.gia_attacks.modular.core.state import Context, State


class FixedRepresentation(Representation):
    """Non-optimized representation that decodes to a fixed data tensor.

    The fixed value comes from ``state.input`` at bind time — the same slot
    ``PixelRepresentation`` reads for a resume/transition init, except here it
    is never wrapped in a parameter and never updated. Useful for ablations
    that hold one part of a reconstruction at ground truth while another
    representation/target is optimized (e.g. isolating whether box
    optimization works when the image is already known, or vice versa) —
    without needing a second driver/loss path that special-cases "this input
    doesn't move".

    Raises at bind time if ``state.input`` is absent — a fixed representation
    with nothing to be fixed at is a configuration error, not a valid "start
    from scratch" case the way ``PixelRepresentation`` treats it.
    """

    def __init__(self, min_val: float = 0.0, max_val: float = 1.0) -> None:
        self.min_val = min_val
        self.max_val = max_val
        self._data: Tensor | None = None

    def bind(self, state: State, ctx: Context) -> FixedRepresentation:
        """Copy holding a detached copy of state.input on the run device; raises if state.input is missing."""
        bound = copy.copy(self)
        if state.input is None:
            raise ValueError(
                "FixedRepresentation requires state.input to be set (the ground-truth "
                "data to hold fixed) — pass it via run_attack(input_state=State(input=...))."
            )
        bound._data = state.input.detach().clone().to(ctx.device)
        return bound

    def parameters(self) -> list[dict]:
        """No parameter groups; the data is fixed."""
        return []

    def decode(self) -> Tensor:
        """The fixed data tensor."""
        return self._data

    def encode(self, data: Tensor) -> Tensor:
        """Clamp data to [min_val, max_val]."""
        return data.clamp(self.min_val, self.max_val)

    def project(self, ctx: Context | None = None) -> None:
        """No-op; there are no parameters to project."""
        pass  # no parameters to project

    def to_observable(self) -> Tensor:
        """The fixed data, detached and clamped to [min_val, max_val]."""
        return self._data.detach().clamp(self.min_val, self.max_val)

    def artifacts(self) -> dict[str, Any]:
        """The fixed data under "pixels", or nothing before binding."""
        return {"pixels": self._data.detach()} if self._data is not None else {}
