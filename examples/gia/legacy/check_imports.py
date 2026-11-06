#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Check that every legacy example still imports the LeakPro names it uses."""
import ast
import importlib
import sys
from pathlib import Path


def broken(path: Path) -> list[str]:
    bad = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("leakpro"):
            try:
                mod = importlib.import_module(node.module)
            except Exception as e:  # noqa: BLE001
                bad.append(f"{node.module}: {type(e).__name__}")
                continue
            for alias in node.names:
                if not hasattr(mod, alias.name):
                    try:
                        importlib.import_module(f"{node.module}.{alias.name}")
                    except Exception:  # noqa: BLE001
                        bad.append(f"{node.module}.{alias.name}")
    return bad


if __name__ == "__main__":
    failures = {str(p): b for p in sorted(Path(__file__).parent.rglob("*.py")) if "data_" not in p.parts and (b := broken(p))}
    for p, b in failures.items():
        print(p, b)
    sys.exit(1 if failures else 0)
