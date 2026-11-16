"""Run every extraction test on the CPU, or on the device in LEAKPRO_TEST_DEVICE.

The tests use tiny models, so the CPU is the default. ``LEAKPRO_TEST_DEVICE=cuda`` (or ``hpu``)
runs the same tests on an accelerator, e.g. to check the GPU path of all four phases.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def _device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEAKPRO_DEVICE", os.environ.get("LEAKPRO_TEST_DEVICE", "cpu"))
