#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Result DTO for the evaluation pass.

Mirrors the ``loss_inputs.py`` / ``interfaces.py::LossTerm`` split: the plain
data container lives here, the ABC that produces it lives in ``interfaces.py``.

Why this is not just ``dict[str, float]``
-----------------------------------------
Evaluators produce two genuinely different kinds of output:

* **metrics** — scalars you can log, average across runs, and put in a table.
* **extras** — everything else the caller needs but cannot average: the
  Hungarian permutation used to align rows, the matched reconstruction, the
  per-interval bounds.  Keeping them out of ``metrics`` means ``metrics`` is always
  ``dict[str, float]``.

Keeping them apart means ``result.metrics`` can be handed straight to a logger
or a dataframe without filtering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalResult:
    """Outcome of one evaluation pass.

    Attributes:
        metrics: Scalar quality measures, safe to log/average/tabulate.
        extras:  Non-scalar by-products (permutations, matched tensors, bounds).

    """

    metrics: dict[str, float] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401 -- genuinely heterogeneous value
        """Read a metric, falling back to extras.

        Lets callers write ``result["psnr"]`` / ``result["permutation"]`` without knowing which dict holds the key.
        """
        if key in self.metrics:
            return self.metrics[key]
        if key in self.extras:
            return self.extras[key]
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        """Whether the key is a metric or an extra."""
        return key in self.metrics or key in self.extras

    def get(self, key: str, default: Any = None) -> Any:  # noqa: ANN401 -- genuinely heterogeneous value
        """Value of the metric or extra, or the default if absent."""
        try:
            return self[key]
        except KeyError:
            return default
