#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Render the result tables in a README from the ``results.json`` files the examples wrote."""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

from examples.gia.common import cli, results

_BLOCK = re.compile(r"<!-- results:(\w[\w-]*) -->.*?<!-- /results -->", re.S)


def _table(data: dict) -> str:
    metrics = list(dict.fromkeys(m for row in data["rows"].values() for m in row))
    lines = ["| Attack | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)]
    for name, row in data["rows"].items():
        cells = [f"{row[m]['mean']:.2f} ± {row[m]['std']:.2f} (n={len(row[m]['values'])})" if m in row else "" for m in metrics]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(_provenance(data))
    return "\n".join(lines)


def _relative(command: str) -> str:
    """The recorded command with every absolute path made relative to the examples folder's repository."""
    def fix(token: str) -> str:
        if not token.startswith("/"):
            return token
        try:
            return Path(token).relative_to(cli.REPO).as_posix()
        except ValueError:
            return Path(token).name

    return " ".join(fix(t) for t in command.split(" "))


def _provenance(data: dict) -> str:
    """Date, torch version, device, determinism and settings of the run; deliberately no git commit."""
    env, config = data["env"], data["config"]
    settings = ", ".join(f"{k}={v}" for k, v in config.items() if k != "command" and isinstance(v, (str, int, float, bool)))
    parts = [f"Command: `{_relative(config.get('command', ''))}`"]
    if env.get("date"):
        parts.append(f"run on {env['date']}")
    parts += [f"torch `{env['torch']}`", f"{env['device']}", "deterministic" if env.get("deterministic") else "not deterministic"]
    if settings:
        parts.append(f"settings: {settings}")
    return "; ".join(parts) + "."


def render(readme: Path, out_root: Path) -> None:
    """Replace every ``<!-- results:NAME -->`` block in ``readme`` by the table of ``out_root/NAME/results.json``.

    A block whose ``results.json`` does not exist yet is left unchanged, with a warning.
    """
    readme = Path(readme)

    def fill(match: re.Match) -> str:
        name = match.group(1)
        out = Path(out_root) / name
        if not (out / "results.json").exists():
            warnings.warn(f"{readme}: no results for block {name!r} ({out / 'results.json'} is missing); left unchanged",
                          stacklevel=2)
            return match.group(0)
        return f"<!-- results:{name} -->\n{_table(results.load(out))}\n<!-- /results -->"

    readme.write_text(_BLOCK.sub(fill, readme.read_text()))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("readme", type=Path)
    p.add_argument("--out-root", type=Path, default=Path("examples/gia/outputs"))
    a = p.parse_args()
    render(a.readme, a.out_root)
