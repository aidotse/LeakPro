#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""The tabular examples' full-size runs in run_all."""
from __future__ import annotations

from pathlib import Path


def test_run_all_runs_the_tabular_entries_on_cpu_deterministically(tmp_path: Path):
    from examples.gia.common import run_all

    assert not run_all.needs_gpu(run_all.FULL["architecture"][1])  # CPU-only entries skip the GPU wait
    assert "--deterministic" in run_all.command("architecture", tmp_path)
