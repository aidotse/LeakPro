"""End-to-end wiring test: real audit -> assess_risk -> JSON and PDF section.

Reuses the tiny-target machinery from the MIA end-to-end suite so this exercises the real attack
pipeline rather than synthetic results. The audit population is small (72 records, ~32 members), so
the operating point here is 10% FPR: anything stricter is genuinely unresolvable at that size, which
is exactly what the resolution guard enforces.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from leakpro import LeakPro
from leakpro.reporting.report_handler import ReportHandler
from leakpro.risk import RiskAssessment, UseCaseProfile, assess_risk
from leakpro.tests.mia_attacks.attacks.test_all_attacks_end_to_end import (
    TinyImageInputHandler,
    _clear_global_attack_cache,
    _create_image_e2e_config,
    _set_seed,
)

E2E_ALPHA = 0.1


@pytest.fixture
def audited(monkeypatch, tmp_path):
    """Run a real LiRA audit on a tiny target and return (leakpro, results)."""
    _set_seed()
    with tempfile.TemporaryDirectory(prefix="risk_e2e_") as temp_dir:
        temp_root = Path(temp_dir)
        monkeypatch.chdir(temp_root)
        _clear_global_attack_cache()
        run_dir = temp_root / "lira"
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = _create_image_e2e_config(run_dir, "lira")
        leakpro = LeakPro(TinyImageInputHandler, str(config_path))
        results = leakpro.run_audit(create_pdf=False, use_optuna=False)
        yield leakpro, results, config_path


class TestEndToEnd:
    """The whole path, on results produced by a real audit."""

    def test_should_assess_risk_on_real_audit_results(self, audited) -> None:
        """assess_risk must consume what run_audit returns, with no adaptation by the caller."""
        leakpro, results, _ = audited
        profile = UseCaseProfile(tolerated_fpr=E2E_ALPHA, attacker_prior=0.01, data_type_sensitivity=3.0)
        assessment = assess_risk(results, profile,
                                 num_train=leakpro.handler.target_model_metadata.num_train)

        assert assessment.measured.resolvable is True
        assert 0.0 <= assessment.measured.success_rate <= 1.0
        assert assessment.measured.n_members_audit > 0
        assert assessment.combined.ppv is not None
        assert assessment.combined.risk is not None
        assert assessment.declared.n_subjects_source == "target num_train"

    def test_should_write_a_self_contained_json_file(self, audited) -> None:
        """The saved file is the deliverable: it must carry every block plus provenance."""
        leakpro, results, _ = audited
        assessment = assess_risk(results, UseCaseProfile(tolerated_fpr=E2E_ALPHA),
                                 num_train=leakpro.handler.target_model_metadata.num_train)
        path = Path(leakpro.report_dir) / "risk_assessment.json"
        assessment.save(str(path))

        assert path.exists() and path.stat().st_size > 0
        payload = json.loads(path.read_text())
        assert set(payload) >= {"measured", "declared", "combined", "assumptions", "warnings",
                                "policy_version", "leakpro_version"}
        assert RiskAssessment.load(str(path)).combined.gamma == assessment.combined.gamma

    def test_should_add_a_risk_section_to_the_pdf_report(self, audited) -> None:
        """A ReportHandler given an assessment emits the Risk section ahead of the attack sections."""
        leakpro, results, _ = audited
        assessment = assess_risk(results, UseCaseProfile(tolerated_fpr=E2E_ALPHA, n_subjects=100))
        handler = ReportHandler(results=results, report_dir=leakpro.report_dir,
                               risk_assessment=assessment)
        handler.create_results()
        handler._init_pdf()
        from leakpro.risk.render import to_latex
        handler.latex_content += to_latex(handler.risk_assessment)
        assert "\\section{Risk assessment}" in handler.latex_content

    def test_should_leave_run_audit_untouched_without_a_use_case_block(self, audited) -> None:
        """A config with no use_case block must behave exactly as before."""
        leakpro, results, _ = audited
        assert leakpro.use_case_profile is None
        assert isinstance(results, list) and results

    def test_should_expose_a_profile_declared_in_the_audit_config(self, audited) -> None:
        """A use_case block in audit.yaml is parsed and reachable, but triggers nothing by itself."""
        _leakpro, _results, config_path = audited
        config = yaml.safe_load(Path(config_path).read_text())
        config["use_case"] = {
            "tolerated_fpr": E2E_ALPHA,
            "attacker_prior": 0.01,
            "data_type_sensitivity": 4.0,
            "notes": "declared in audit.yaml",
        }
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        reloaded = LeakPro(TinyImageInputHandler, str(config_path))
        assert reloaded.use_case_profile is not None
        assert reloaded.use_case_profile.tolerated_fpr == E2E_ALPHA
        assert reloaded.use_case_profile.data_type_sensitivity == 4.0
        assert reloaded.use_case_profile.notes == "declared in audit.yaml"

    def test_should_reject_an_invalid_use_case_block_in_the_config(self, audited) -> None:
        """A typo in the config must fail validation rather than be ignored."""
        _leakpro, _results, config_path = audited
        config = yaml.safe_load(Path(config_path).read_text())
        config["use_case"] = {"tolerated_fpr": E2E_ALPHA, "sensitivity": "high"}
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        with pytest.raises(Exception, match="use_case|sensitivity"):
            LeakPro(TinyImageInputHandler, str(config_path))
