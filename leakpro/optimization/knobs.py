#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Search-space definition for PET optimization campaigns.

A ``KnobSpace`` holds the tunable hyperparameters (knobs) of a PET configuration.
Users may fix any knob to a constant or narrow its bounds; the default DP-SGD
space searches {noise multiplier, clipping norm, learning rate, batch size}
jointly — never utility knobs first, privacy knobs after.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True)
class Knob:
    """One tunable hyperparameter.

    Args:
        name: Config key the sampled value is emitted under.
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).
        log_scale: Sample uniformly in log10 space (bounds must be > 0).
        integer: Round the sampled value to the nearest integer.

    """

    name: str
    low: float
    high: float
    log_scale: bool = False
    integer: bool = False

    def __post_init__(self) -> None:
        """Validate bounds."""
        if self.low >= self.high:
            raise ValueError(f"Knob '{self.name}': low ({self.low}) must be < high ({self.high}).")
        if self.log_scale and self.low <= 0:
            raise ValueError(f"Knob '{self.name}': log_scale requires positive bounds, got low={self.low}.")

    def from_unit(self, u: float) -> float:
        """Map a value in [0, 1] to the knob's native range."""
        if self.log_scale:
            value = 10 ** (np.log10(self.low) + u * (np.log10(self.high) - np.log10(self.low)))
        else:
            value = self.low + u * (self.high - self.low)
        if self.integer:
            return int(np.clip(round(value), self.low, self.high))
        return float(value)


class KnobSpace:
    """A set of knobs with optional user-fixed values.

    Fixed knobs are emitted in every sampled configuration but consume no
    search dimension.
    """

    def __init__(self, knobs: list[Knob], fixed: dict[str, float] | None = None) -> None:
        names = [k.name for k in knobs]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate knob names in {names}.")
        self.fixed = dict(fixed or {})
        overlap = set(names) & set(self.fixed)
        if overlap:
            raise ValueError(f"Knobs {sorted(overlap)} are both searchable and fixed; pick one.")
        self.knobs = list(knobs)

    @property
    def dim(self) -> int:
        """Number of searched dimensions."""
        return len(self.knobs)

    def narrow(self, name: str, low: float, high: float) -> "KnobSpace":
        """Return a copy of the space with one knob's bounds narrowed."""
        knobs = [replace(k, low=low, high=high) if k.name == name else k for k in self.knobs]
        if not any(k.name == name for k in self.knobs):
            raise KeyError(f"No searchable knob named '{name}'.")
        return KnobSpace(knobs, fixed=self.fixed)

    def fix(self, name: str, value: float) -> "KnobSpace":
        """Return a copy of the space with one knob fixed to a constant."""
        if not any(k.name == name for k in self.knobs):
            raise KeyError(f"No searchable knob named '{name}'.")
        knobs = [k for k in self.knobs if k.name != name]
        return KnobSpace(knobs, fixed={**self.fixed, name: value})

    def sample_sobol(self, n: int, seed: int = 0) -> list[dict[str, float]]:
        """Draw ``n`` configurations from a scrambled Sobol sequence (deterministic per seed)."""
        if self.dim == 0:
            return [dict(self.fixed) for _ in range(n)]
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=seed)
        unit = sampler.random(n)
        configs = []
        for row in unit:
            config = {knob.name: knob.from_unit(u) for knob, u in zip(self.knobs, row)}
            config.update(self.fixed)
            configs.append(config)
        return configs


def default_dpsgd_space() -> KnobSpace:
    """The plan's default joint DP-SGD search space.

    Bounds follow published DP-SGD tuning practice: wide, log-scaled, and
    searched jointly because batch size enters the privacy accounting.
    """
    return KnobSpace([
        Knob("noise_multiplier", 0.4, 8.0, log_scale=True),
        Knob("max_grad_norm", 0.1, 10.0, log_scale=True),
        Knob("learning_rate", 1e-5, 1e-1, log_scale=True),
        Knob("batch_size", 32, 1024, log_scale=True, integer=True),
    ])
