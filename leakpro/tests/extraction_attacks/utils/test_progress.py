"""Progress lines of long extraction stages."""

import logging

import pytest

from leakpro.attacks.utils.extraction.progress import Progress, format_duration


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_format_duration() -> None:
    assert format_duration(45) == "45s"
    assert format_duration(760) == "12m 40s"
    assert format_duration(3 * 3600 + 5 * 60 + 9) == "3h 05m"


def test_logs_at_most_every_interval_with_speed_and_eta(caplog: pytest.LogCaptureFixture) -> None:
    clock = FakeClock()
    progress = Progress("initial-generate-baseline", total=625, unit="batch", done=69, every=60, clock=clock)
    with caplog.at_level(logging.INFO, logger="leakpro"):
        clock.now = 31.0
        progress.update()          # 31 s since start: no line yet
        clock.now = 62.0
        progress.update()          # 62 s: logs
    lines = [r.getMessage() for r in caplog.records]
    assert lines == ["initial-generate-baseline: 71/625 batches (11%) · 31.0 s/batch · ~4h 46m left"]


def test_fast_batches_are_pluralized(caplog: pytest.LogCaptureFixture) -> None:
    clock = FakeClock()
    progress = Progress("score", total=2, unit="batch", clock=clock)
    with caplog.at_level(logging.INFO, logger="leakpro"):
        clock.now = 0.5
        progress.update(2)
    assert caplog.records[-1].getMessage() == "score: 2/2 batches (100%) · 4.0 batches/s · done in 0s"


def test_final_line_and_fast_units(caplog: pytest.LogCaptureFixture) -> None:
    clock = FakeClock()
    progress = Progress("initial-verify", total=4, unit="candidate", every=60, clock=clock)
    with caplog.at_level(logging.INFO, logger="leakpro"):
        for _ in progress.wrap(range(4)):
            clock.now += 0.25
    assert [r.getMessage() for r in caplog.records] == ["initial-verify: 4/4 candidates (100%) · 4.0 candidates/s · done in 1s"]
