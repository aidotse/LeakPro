#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Progress lines for the long stages of an extraction run (generation, scoring, verification, training).

A stage on an 8B model can run for hours; :class:`Progress` logs where it is, how fast it goes and how
long it still needs, at most once every ``every`` seconds plus once at the end::

    initial-generate-baseline: 70/625 batches (11%) · 31.4 s/batch · ~4h 50m left
"""

import time
from typing import Callable, Iterable, Iterator, Optional, TypeVar

from leakpro.utils.logger import logger

T = TypeVar("T")


def format_duration(seconds: float) -> str:
    """``3h 05m``, ``12m 40s`` or ``45s``."""
    seconds = max(0, int(round(seconds)))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


class Progress:
    """Count finished units of a stage and log progress with speed and remaining time.

    Args:
    ----
        label: Stage name at the start of every line.
        total: Units in the whole stage.
        unit: Name of one unit, singular (``batch``, ``candidate``, ``step``).
        done: Units already finished before this run (resumed from the cache); they count towards
            the total but not towards the speed.
        every: Minimum seconds between two lines.
        clock: Time source (tests pass a fake one).

    """

    def __init__(  # noqa: PLR0913
        self,
        label: str,
        total: int,
        unit: str = "batch",
        done: int = 0,
        every: float = 60.0,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.label, self.total, self.unit, self.every = label, total, unit, every
        self.units = unit + ("es" if unit.endswith(("ch", "s", "x")) else "s")
        self.done = self._resumed = done
        self._clock = clock or time.monotonic
        self._start = self._last = self._clock()

    def update(self, n: int = 1) -> None:
        """Mark ``n`` more units as finished; logs when ``every`` seconds have passed or the stage is done."""
        self.done += n
        now = self._clock()
        if self.done >= self.total or now - self._last >= self.every:
            self._last = now
            logger.info(self.message(now))

    def message(self, now: Optional[float] = None) -> str:
        """The progress line for the current state."""
        now = self._clock() if now is None else now
        pct = 100 * self.done / self.total if self.total else 100.0
        line = f"{self.label}: {self.done}/{self.total} {self.units} ({pct:.0f}%)"
        new, elapsed = self.done - self._resumed, now - self._start
        if new > 0 and elapsed > 0:
            per_unit = elapsed / new
            speed = f"{per_unit:.1f} s/{self.unit}" if per_unit >= 1 else f"{1 / per_unit:.1f} {self.units}/s"
            if self.done >= self.total:
                line += f" · {speed} · done in {format_duration(elapsed)}"
            else:
                line += f" · {speed} · ~{format_duration(per_unit * (self.total - self.done))} left"
        return line

    def wrap(self, items: Iterable[T]) -> Iterator[T]:
        """Yield ``items``, counting one unit after each."""
        for item in items:
            yield item
            self.update()
