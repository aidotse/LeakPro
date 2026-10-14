#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Search-space definition for privacy-utility optimization runs.

A ``KnobSpace`` holds the tunable hyperparameters (knobs) of a PET configuration
and knows how to offer them to an Optuna trial. The optimizer (see
``leakpro.optimization.search``) proposes each next configuration from the
results observed so far — the knob space only describes *what* may be searched,
never *which* point comes next.

Users may fix any knob to a constant or narrow its bounds; the default DP-SGD
space searches {noise multiplier, clipping norm, learning rate, batch size}
jointly — never utility knobs first, privacy knobs after.
"""

from dataclasses import dataclass, replace

from optuna.trial import Trial

from leakpro.schemas import KnobConfig


@dataclass(frozen=True)
class Knob:
    """One tunable hyperparameter.

    Args:
        name: Config key the suggested value is emitted under.
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).
        log_scale: Search on a log scale (bounds must be > 0).
        integer: Suggest an integer value.

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

    def suggest(self, trial: Trial) -> float:
        """Ask an Optuna trial for a value of this knob within its range."""
        if self.integer:
            return trial.suggest_int(self.name, int(self.low), int(self.high), log=self.log_scale)
        return trial.suggest_float(self.name, self.low, self.high, log=self.log_scale)


class KnobSpace:
    """A set of knobs with optional user-fixed values.

    Fixed knobs are emitted in every configuration but consume no search
    dimension.
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

    @classmethod
    def from_config(cls, knobs: list[KnobConfig], fixed: dict[str, float] | None = None) -> "KnobSpace":
        """Build a knob space from validated config entries (see PrivacyUtilityConfig)."""
        return cls([Knob(**k.model_dump()) for k in knobs], fixed=fixed)

    @property
    def dim(self) -> int:
        """Number of searched dimensions."""
        return len(self.knobs)

    def narrow(self, name: str, low: float, high: float) -> "KnobSpace":
        """Return a copy of the space with one knob's bounds narrowed."""
        if not any(k.name == name for k in self.knobs):
            raise KeyError(f"No searchable knob named '{name}'.")
        knobs = [replace(k, low=low, high=high) if k.name == name else k for k in self.knobs]
        return KnobSpace(knobs, fixed=self.fixed)

    def fix(self, name: str, value: float) -> "KnobSpace":
        """Return a copy of the space with one knob fixed to a constant."""
        if not any(k.name == name for k in self.knobs):
            raise KeyError(f"No searchable knob named '{name}'.")
        knobs = [k for k in self.knobs if k.name != name]
        return KnobSpace(knobs, fixed={**self.fixed, name: value})

    def suggest(self, trial: Trial) -> dict[str, float]:
        """Build one configuration by asking ``trial`` for every searchable knob.

        Fixed knobs are added verbatim so a configuration always carries the full
        set of hyperparameters, whether searched or pinned.
        """
        config = {knob.name: knob.suggest(trial) for knob in self.knobs}
        config.update(self.fixed)
        return config

    def to_dict(self) -> dict:
        """JSON-serializable description of the space, stored on the study."""
        return {
            "knobs": [
                {"name": k.name, "low": k.low, "high": k.high, "log_scale": k.log_scale, "integer": k.integer}
                for k in self.knobs
            ],
            "fixed": dict(self.fixed),
        }


def default_dpsgd_space() -> KnobSpace:
    """The default joint DP-SGD search space.

    Bounds follow published DP-SGD tuning practice: wide, log-scaled, and
    searched jointly because batch size enters the privacy accounting.
    """
    return KnobSpace([
        Knob("noise_multiplier", 0.4, 8.0, log_scale=True),
        Knob("max_grad_norm", 0.1, 10.0, log_scale=True),
        Knob("learning_rate", 1e-5, 1e-1, log_scale=True),
        Knob("batch_size", 32, 1024, log_scale=True, integer=True),
    ])
