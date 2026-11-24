#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""The README's compose-your-own-attack snippet runs as written."""
from __future__ import annotations

import re
from pathlib import Path

README = Path(__file__).resolve().parents[3] / "attacks" / "gia_attacks" / "modular" / "README.md"


def _snippet() -> str:
    text = README.read_text()
    section = text.split("## Compose your own attack", 1)[1]
    return re.search(r"```python\n(.*?)```", section, re.S).group(1)


def test_compose_your_own_attack_snippet_runs():
    namespace: dict = {}
    exec(compile(_snippet(), str(README), "exec"), namespace)  # noqa: S102 -- executing documentation is the test
    state = namespace["state"]
    assert tuple(state.input.shape) == tuple(namespace["images"].shape)
