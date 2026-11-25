#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""On-disk cache for extraction stages, so an interrupted or re-configured run redoes only what changed.

Every stage writes its candidates under ``<root>/<stage name>-<key[:16]>/``. The key hashes the
stage's own settings *and* the key of the stage it consumed, so changing anything upstream gives
every downstream stage a new key, while an unchanged stage is read back instead of recomputed.

Inside a stage directory:

* ``batch_<i>.jsonl``: one file per generation batch, written atomically. A resumed run skips the
  batches that already exist.
* ``candidates.jsonl`` + ``done.json``: the stage's complete output. ``done.json`` is written last,
  so a stage without it is incomplete.

Only the latest copy of each stage is kept: when a stage completes, the directories of the same
stage under other keys (earlier settings, or an unfinished run of them) are deleted. A stage that is
still running is never deleted, so an interrupted run resumes; the copy it replaces stays until the
new one is complete. Going back to an earlier setting therefore recomputes the stages it affects.
"""

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

from leakpro.attacks.utils.extraction.records import Candidate
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


def stage_key(*parts: Any) -> str:  # noqa: ANN401
    """SHA-256 of the canonical JSON of ``parts``."""
    blob = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _dump(candidates: Sequence[Candidate]) -> str:
    return "".join(json.dumps(c.to_dict(), ensure_ascii=False) + "\n" for c in candidates)


def _load(path: Path) -> List[Candidate]:
    with path.open(encoding="utf-8") as f:
        return [Candidate.from_dict(json.loads(line)) for line in f if line.strip()]


class Stage:
    """One cached stage directory."""

    def __init__(self: Self, root: Path, name: str, key: str) -> None:
        self.name = name
        self.key = key
        self.path = root / f"{name}-{key[:16]}"

    def is_done(self: Self) -> bool:
        """Whether the stage completed in an earlier run."""
        return (self.path / "done.json").exists()

    def read(self: Self) -> List[Candidate]:
        """The stage's complete output."""
        if not self.is_done():
            raise RuntimeError(f"Stage {self.path} is not complete")
        return _load(self.path / "candidates.jsonl")

    def write(self: Self, candidates: Sequence[Candidate], info: Dict[str, Any] = None) -> None:
        """Store the stage's complete output and mark it done."""
        self.path.mkdir(parents=True, exist_ok=True)
        _write_atomic(self.path / "candidates.jsonl", _dump(candidates))
        _write_atomic(self.path / "done.json", json.dumps({"key": self.key, "count": len(candidates), **(info or {})}))
        self.remove_other_copies()

    def remove_other_copies(self: Self) -> None:
        """Delete the directories of this stage under other keys (``<name>-<16 hex>`` only, never ``<name>-x-...``)."""
        pattern = re.compile(rf"{re.escape(self.name)}-[0-9a-f]{{16}}")
        for other in self.path.parent.iterdir():
            if other != self.path and other.is_dir() and pattern.fullmatch(other.name):
                shutil.rmtree(other)
                logger.info(f"Removed the older copy {other.name} of stage {self.name}")

    def completed_batches(self: Self) -> Set[int]:
        """Indices of generation batches already on disk."""
        if not self.path.exists():
            return set()
        return {int(p.stem.split("_")[1]) for p in self.path.glob("batch_*.jsonl")}

    def write_batch(self: Self, index: int, candidates: Sequence[Candidate]) -> None:
        """Store one generation batch."""
        self.path.mkdir(parents=True, exist_ok=True)
        _write_atomic(self.path / f"batch_{index:06d}.jsonl", _dump(candidates))

    def read_batches(self: Self) -> List[Candidate]:
        """Every stored batch, in batch order."""
        out: List[Candidate] = []
        for path in sorted(self.path.glob("batch_*.jsonl")):
            out.extend(_load(path))
        return out


class ArtifactStore:
    """Stage directories under one root.

    Args:
    ----
        root: Directory holding every stage, e.g. ``<output_dir>/artifacts``.

    """

    def __init__(self: Self, root: str) -> None:
        self.root = Path(root)

    def stage(self: Self, name: str, key: str) -> Stage:
        """The stage directory for ``name`` under ``key``."""
        return Stage(self.root, name, key)
