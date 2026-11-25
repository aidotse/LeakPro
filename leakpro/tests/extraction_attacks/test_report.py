"""PDF report support for extraction results (AuditResult dispatch in ReportHandler)."""

import json
import shutil
import logging
from pathlib import Path

import numpy as np
import pytest
import yaml

from leakpro import LeakPro
from leakpro.reporting.audit_result import AuditResult, latex_escape
from leakpro.reporting.extraction_result import ExtractionResult
from leakpro.reporting.mia_result import MIAResult
from leakpro.reporting.report_handler import ReportHandler
from leakpro.tests.extraction_attacks import fixtures
from leakpro.tests.extraction_attacks.fixtures import SECRET, ExtractionInputHandler

ROW = {"generated": 10, "kept": 9, "selected": 2, "verified": 1, "unique": None, "extraction_rate": 0.1}


def _result(rid: str = "abc_123") -> ExtractionResult:
    metrics = {
        "initial": {"baseline": dict(ROW), "total": dict(ROW)},
        "expansion": {"expansion": {**ROW, "anchors": 3, "novel": 1}, "total": {**ROW, "anchors": 3, "novel": 1}},
    }
    return ExtractionResult("memorization_extraction", rid, {}, metrics, [{"id": "x"}])


def test_latex_escape() -> None:
    assert latex_escape("a_b & 50% #1 {x} $y ~ ^ \\") == \
        r"a\_b \& 50\% \#1 \{x\} \$y \textasciitilde{} \textasciicircum{} \textbackslash{}"


def test_extraction_section_lists_every_phase_without_text() -> None:
    latex = ExtractionResult.create_results([_result(), _result()])  # duplicate ids render once
    assert latex.count("\\subsection{") == 1
    assert "Phase 1: initial extraction" in latex and "Phase 2: expansion" in latex
    assert "Anchors" in latex and "Novel" in latex and "& Unique" not in latex  # empty columns dropped
    assert "10.00\\%" in latex and "abc\\_123" in latex


def test_report_handler_renders_extraction_next_to_mia(tmp_path: Path) -> None:
    mia = MIAResult.from_full_scores(true_membership=np.array([False, True, True, False]),
                                     signal_values=np.array([0.1, 0.7, 0.4, 0.3]), result_name="dummy")
    handler = ReportHandler(results=[mia, _result()], report_dir=str(tmp_path))
    assert handler.audit_result_types == {"ExtractionResult": ExtractionResult}
    handler.create_report()  # compiling needs pdflatex; the .tex is written either way
    tex = (tmp_path / "LeakPro_output.tex").read_text()
    assert "\\section{MIAResult}" in tex
    assert "\\section{Training-data extraction}" in tex and "Phase 2: expansion" in tex


def test_unknown_result_types_are_skipped_not_fatal(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    class SomethingElse:
        pass

    with caplog.at_level(logging.WARNING, logger="leakpro"):
        handler = ReportHandler(results=[SomethingElse(), _result()], report_dir=str(tmp_path))
    assert len(handler.results) == 1
    assert "SomethingElse cannot be rendered" in caplog.text


def test_custom_audit_result_gets_its_own_section(tmp_path: Path) -> None:
    class ToyResult(AuditResult):
        report_title = "Toy attack"

        @classmethod
        def create_results(cls, results: list, save_dir: str = "./") -> str:
            return f"toy results: {len(results)}"

    handler = ReportHandler(results=[ToyResult(), ToyResult()], report_dir=str(tmp_path))
    handler.create_report()
    tex = (tmp_path / "LeakPro_output.tex").read_text()
    assert "\\section{Toy attack}toy results: 2" in tex


def test_run_audit_with_pdf(tmp_path: Path) -> None:
    corpus = tmp_path / "aux.jsonl"
    corpus.write_text(json.dumps({"doc_id": "a", "text": SECRET}) + "\n")
    prefixes = tmp_path / "prefixes.jsonl"
    prefixes.write_text(json.dumps({"prefix": SECRET[:4]}) + "\n")
    config = {
        "audit": {"random_seed": 1, "attack_type": "extraction", "data_modality": "text",
                  "output_dir": str(tmp_path / "out"),
                  "attack_list": [{"attack": "memorization_extraction", "generation_backend": "torch",
                                   "initial": [{"type": "baseline", "n": 2, "batch_size": 2, "max_tokens": 300}],
                                   "attacker_verifier": {"path": str(corpus)}}]},
        "target": {"module_path": fixtures.__file__, "model_class": "ParrotLM",
                   "target_folder": str(tmp_path / "target"), "data_path": str(prefixes)},
    }
    path = tmp_path / "audit.yaml"
    path.write_text(yaml.safe_dump(config))
    LeakPro(ExtractionInputHandler, str(path)).run_audit(create_pdf=True)
    tex = (tmp_path / "out" / "results" / "LeakPro_output.tex").read_text()
    assert "\\section{Training-data extraction}" in tex and "baseline & 2 & 1 & 1 & 1 & 1 & 50.00\\%" in tex


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="needs pdflatex (installed in CI)")
def test_extraction_section_compiles(tmp_path: Path) -> None:
    ReportHandler(results=[_result()], report_dir=str(tmp_path)).create_report()
    assert (tmp_path / "LeakPro_output.pdf").is_file()
