#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Result of a memorization extraction attack.

Extraction has no membership labels and no ROC curve. The result holds counts per phase and
prompting strategy, and the extracted (verified) sequences.

Saved files:

* ``<output_dir>/data_objects/<attack_id>.json``: config and metrics (same place as MIA results).
* ``<output_dir>/results/<attack_id>/result.json``: config and metrics.
* ``<output_dir>/results/<attack_id>/extracted.jsonl``: one record per extracted sequence, with its
  text and matched windows.

Only the latest result of an attack is kept: saving removes the files of earlier results with the
same attack name (other attack ids, i.e. earlier configs) from both places.
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from leakpro.reporting.audit_result import AuditResult, latex_escape
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger

_PHASE_TITLES = {
    "initial": "Phase 1: initial extraction",
    "expansion": "Phase 2: expansion",
    "finetuned": "Phase 4: re-extraction after fine-tuning",
}
# (metric key, column header); a column appears only when some row of the phase has a value for it.
_COLUMNS = [
    ("anchors", "Anchors"), ("generated", "Generated"), ("overlap_removed", "Overlap removed"), ("kept", "Kept"),
    ("selected", "Selected"), ("verified", "Verified"), ("unique", "Unique"), ("extraction_rate", "Rate"),
    ("novel", "Novel"), ("novel_unique", "Novel unique"),
]


def _cell(key: str, value: Any) -> str:  # noqa: ANN401
    if value is None:
        return "--"
    if key.endswith("rate"):
        return f"{100 * value:.2f}\\%"
    return latex_escape(value)


def _result_name(result_json: Path) -> Optional[str]:
    try:
        return json.loads(result_json.read_text(encoding="utf-8")).get("name")
    except (OSError, ValueError, AttributeError):
        return None


class ExtractionResult(AuditResult):
    """Counts and extracted sequences of one extraction attack.

    Args:
    ----
        name: Attack name.
        attack_id: Attack hash (also the result's id).
        config: The attack config as a dict.
        metrics: ``{phase: {strategy | "total": {count name: value}}}``.
        extracted: Records of the sequences the attacker verifier confirmed.

    """

    def __init__(
        self: Self,
        name: str,
        attack_id: str,
        config: Dict[str, Any],
        metrics: Dict[str, Dict[str, Dict[str, Optional[float]]]],
        extracted: List[Dict[str, Any]],
    ) -> None:
        self.name = name
        self.result_name = name
        self.id = attack_id
        self.config = config
        self.metrics = metrics
        self.extracted = extracted

    report_title = "Training-data extraction"

    def to_dict(self: Self) -> Dict[str, Any]:
        """Config and metrics (no extracted text)."""
        return {"name": self.name, "id": self.id, "config": self.config, "metrics": self.metrics}

    def save(self: Self, attack_obj: Any = None, output_dir: str = ".") -> None:  # noqa: ANN401, ARG002
        """Write the result files (see module docstring)."""
        data_dir = Path(output_dir) / "data_objects"
        result_dir = Path(output_dir) / "results" / self.id
        data_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        summary = json.dumps(self.to_dict(), indent=2, default=str)
        (data_dir / f"{self.id}.json").write_text(summary, encoding="utf-8")
        (result_dir / "result.json").write_text(summary, encoding="utf-8")
        with (result_dir / "extracted.jsonl").open("w", encoding="utf-8") as f:
            for record in self.extracted:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._remove_older_results(Path(output_dir))

    def _remove_older_results(self: Self, output_dir: Path) -> None:
        """Delete earlier results of this attack (same name, other id); other attacks' files are left alone."""
        for result_json in (output_dir / "results").glob("*/result.json"):
            other = result_json.parent
            if other.name == self.id or _result_name(result_json) != self.name:
                continue
            shutil.rmtree(other)
            data_object = output_dir / "data_objects" / f"{other.name}.json"
            if data_object.exists():
                data_object.unlink()
            logger.info(f"Removed the earlier {self.name} result {other.name[:12]}")

    @classmethod
    def load(cls, result_dir: str) -> "ExtractionResult":  # noqa: ANN102
        """Read a result written by :meth:`save` from ``<output_dir>/results/<attack_id>``."""
        path = Path(result_dir)
        data = json.loads((path / "result.json").read_text(encoding="utf-8"))
        extracted_path = path / "extracted.jsonl"
        extracted = []
        if extracted_path.exists():
            with extracted_path.open(encoding="utf-8") as f:
                extracted = [json.loads(line) for line in f if line.strip()]
        return cls(data["name"], data["id"], data["config"], data["metrics"], extracted)

    def summary(self: Self) -> str:
        """Human-readable table of the counts."""
        lines = []
        for phase, by_strategy in self.metrics.items():
            lines.append(f"{phase}:")
            for strategy, m in by_strategy.items():
                verified, unique = m.get("verified"), m.get("unique")
                rate = m.get("extraction_rate")
                extras = "".join(f"  {k} {m[k]}" for k in ("overlap_removed", "anchors", "novel", "novel_unique",
                                                           "finetune_corpus") if k in m)
                lines.append(
                    f"  {strategy:<12} generated {m['generated']:>6}  kept {m['kept']:>6}  selected {m['selected']:>5}  "
                    f"verified {'-' if verified is None else verified:>5}  unique {'-' if unique is None else unique:>5}  "
                    f"rate {'-' if rate is None else f'{100 * rate:.2f}%'}{extras}"
                )
        return "\n".join(lines)

    @classmethod
    def create_results(cls, results: List["ExtractionResult"], save_dir: str = "./") -> str:  # noqa: ANN102, ARG003
        """LaTeX report section: one table of counts per phase for each result (no extracted text)."""
        latex = (
            "\nA sequence counts as extracted when one of its word windows occurs verbatim in the attacker's "
            "reference corpus (\\emph{Verified}). "
            "\\emph{Unique} counts distinct passages: verified sequences whose matches share corpus words or a "
            "window-length text count once. "
            "Rates are verified sequences over generated sequences.\n"
        )
        seen = set()
        for res in results:
            if res.id in seen:
                continue
            seen.add(res.id)
            latex += f"\n\\subsection{{{latex_escape(res.name)} ({latex_escape(res.id[:12])})}}\n"
            for phase, rows in res.metrics.items():
                columns = [(k, h) for k, h in _COLUMNS if any(r.get(k) is not None for r in rows.values())]
                spec = "l" + "r" * len(columns)
                header = " & ".join(["Strategy"] + [h for _, h in columns])
                body = " \\\\\n".join(
                    " & ".join([latex_escape(name)] + [_cell(k, row.get(k)) for k, _ in columns])
                    for name, row in rows.items()
                )
                title = _PHASE_TITLES.get(phase, phase)
                extra = rows.get("total", {}).get("finetune_corpus")
                note = f" Fine-tuning corpus: {extra} sequences." if extra is not None else ""
                latex += (
                    f"\n\\paragraph{{{latex_escape(title)}}}{note}\n"
                    "\\begin{table}[ht]\n\\centering\n\\small\n"
                    f"\\begin{{tabular}}{{{spec}}}\n\\hline\n{header} \\\\\n\\hline\n{body} \\\\\n"
                    "\\hline\n\\end{tabular}\n\\end{table}\n"
                )
            latex += (
                f"\nExtracted sequences: {len(res.extracted)}, listed in "
                f"\\texttt{{results/{latex_escape(res.id)}/extracted.jsonl}}.\n"
            )
        return latex
