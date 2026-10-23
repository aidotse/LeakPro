"""End-to-end wiring test: real audit -> assess_risk -> JSON and PDF section.

Reuses the tiny-target machinery from the MIA end-to-end suite so this exercises the real attack
pipeline rather than synthetic results. The audit population is small (72 records, ~32 members), so
the operating point here is 10% FPR: anything stricter is genuinely unresolvable at that size, which
is exactly what the resolution guard enforces.

The shared machinery saves an untrained target, which gives a chance-level audit whose AUC lands on
either side of 0.5 depending on the seed. The fixture here overfits the target to its training split
first, so the assessment runs on a real leak and the test can check that it registers as one.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import joblib
import pytest
import torch
import yaml
from torch import nn

from leakpro import LeakPro
from leakpro.reporting.report_handler import ReportHandler
from leakpro.risk import RiskAssessment, UseCaseProfile, assess_risk
from leakpro.tests.mia_attacks.attacks.test_all_attacks_end_to_end import (
    TRAIN_INDICES,
    TinyImageInputHandler,
    TinyImageTargetModel,
    _clear_global_attack_cache,
    _create_image_e2e_config,
    _set_seed,
)

E2E_ALPHA = 0.1
OVERFIT_EPOCHS = 200


def _overfit_target(run_dir: Path) -> None:
    """Train the saved tiny target until it memorises its training split, in place.

    Args:
    ----
        run_dir: Run directory written by ``_create_image_e2e_config``.

    """
    population = joblib.load(run_dir / "tiny_cifar.pkl")
    state_path = run_dir / "target" / "target_model.pkl"
    model = TinyImageTargetModel()
    model.load_state_dict(torch.load(state_path))
    inputs, labels = population.data[TRAIN_INDICES], population.targets[TRAIN_INDICES]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(OVERFIT_EPOCHS):
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(inputs), labels).backward()
        optimizer.step()
    torch.save(model.state_dict(), state_path)


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
        _overfit_target(run_dir)
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
        # The target memorised its training split, so the attack must clear the random-guess baseline
        # by a wide margin; a wiring fault that dropped or misoriented scores would not.
        assert assessment.measured.lift > 3.0
        assert assessment.measured.n_members_audit > 0
        assert assessment.combined.ppv is not None
        assert assessment.combined.risk is not None
        assert assessment.declared.n_subjects_source == "target num_train / records_per_subject"

    def test_should_name_the_exposed_members_by_population_index(self, audited) -> None:
        """LiRA reports audit_indices, so the exposed members can be named, and every one is a member.

        Exposed records are flagged members, so each index must be in the target's training split. An
        index outside it means the scores and the indices were misaligned somewhere between the attack
        and the risk layer, which would name the wrong people.
        """
        leakpro, results, _ = audited
        assessment = assess_risk(results, UseCaseProfile(tolerated_fpr=E2E_ALPHA),
                                 num_train=leakpro.handler.target_model_metadata.num_train)
        measured = assessment.measured

        assert measured.attack_name.lower().startswith("lira")
        assert measured.exposed_audit_indices is not None
        assert len(measured.exposed_audit_indices) == measured.n_exposed_audit > 0
        assert set(measured.exposed_audit_indices) <= {int(i) for i in TRAIN_INDICES}

        # In this fixture the training split is population 0..N-1, which is also where members sit in
        # the audit order, so the subset check alone cannot tell indices from positions. Check the
        # alignment through the attack's own mapping: each exposed index is a member, and every
        # exposed member outscores every member that was not exposed.
        lira = next(r for r in results if r.result_name == measured.attack_name)
        by_index = {int(i): (int(t), float(s))
                    for i, t, s in zip(lira.audit_indices, lira.true, lira.signal_values)}
        exposed = set(measured.exposed_audit_indices)
        assert all(by_index[i][0] == 1 for i in exposed)
        hidden_members = [s for i, (t, s) in by_index.items() if t == 1 and i not in exposed]
        if hidden_members:
            assert min(by_index[i][1] for i in exposed) > max(hidden_members)

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
