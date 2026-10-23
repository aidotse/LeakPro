#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the webapp's risk endpoint: the request is the library profile, scoring is the library's."""

from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="webapp extras (fastapi) not installed")

from leakpro.tests.risk.conftest import make_result  # noqa: E402
from leakpro.utils.import_helper import Self  # noqa: E402


def _stored_job(*results: object, num_train: int = 1000) -> dict:
    """A finished job as the worker stores it, with results serialised by the worker itself."""
    from leakpro.webapp.backend.models import JobStatus
    from leakpro.webapp.backend.worker import _serialise_result

    model = {"model_name": "target", "source": "upload", "dpsgd": False, "num_train": num_train,
             "train_accuracy": 0.99, "test_accuracy": 0.70,
             "attacks": [_serialise_result(r) for r in results]}
    return {"status": JobStatus.done, "results": [model]}


class _WithClient:
    """A test client against an isolated job store."""

    @pytest.fixture(autouse=True)
    def _isolated_jobs(self: Self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from fastapi.testclient import TestClient

        from leakpro.webapp.backend import main as backend_main
        from leakpro.webapp.backend import security

        monkeypatch.setattr(backend_main, "JOBS_ROOT", tmp_path)
        monkeypatch.setattr(backend_main, "_jobs", {})
        self.jobs = backend_main._jobs
        self.client = TestClient(backend_main.app)
        self.auth = {"Authorization": f"Bearer {security.API_TOKEN}"}


class TestRiskRequest(_WithClient):
    """RiskRequest is a UseCaseProfile, so the two cannot drift."""

    def test_should_inherit_every_profile_field_and_default(self: Self) -> None:
        """No field is re-declared: the request's fields are the profile's plus model_name."""
        from leakpro.risk.schemas import UseCaseProfile
        from leakpro.webapp.backend.models import RiskRequest

        assert issubclass(RiskRequest, UseCaseProfile)
        assert set(RiskRequest.model_fields) == set(UseCaseProfile.model_fields) | {"model_name"}
        for name, field in UseCaseProfile.model_fields.items():
            assert RiskRequest.model_fields[name].default == field.default

    @pytest.mark.parametrize("body", [
        {},                                              # tolerated_fpr is required
        {"tolerated_fpr": 1.5},                          # outside (0, 1)
        {"tolerated_fpr": 0.01, "attacker_prior": 0.0},  # outside (0, 1)
        {"tolerated_fpr": 0.01, "unknown_factor": 2.0},  # extra="forbid" is inherited
    ])
    def test_should_reject_an_invalid_profile_with_422_before_the_endpoint_runs(self: Self, body: dict) -> None:
        """Validation is the library's, applied by FastAPI, so the job lookup is never reached."""
        res = self.client.post("/jobs/no-such-job/risk", json=body, headers=self.auth)
        assert res.status_code == 422

    def test_should_assess_a_stored_job_with_the_library_profile(self: Self) -> None:
        """A valid request reaches assess_risk and the declared block echoes it without model_name."""
        self.jobs["job"] = _stored_job(make_result(separation=2.0))
        res = self.client.post("/jobs/job/risk", json={"tolerated_fpr": 0.01, "attacker_prior": 0.01,
                                                         "model_name": "target"}, headers=self.auth)
        assert res.status_code == 200
        declared = res.json()["assessments"]["target"]["declared"]
        assert declared["attacker_prior"] == 0.01
        assert "model_name" not in declared


class TestResultsMeasuredBlock(_WithClient):
    """/results serves the measured block from the library, so the UI never selects or thresholds."""

    def _measured(self: Self, job: dict) -> dict:
        self.jobs["job"] = job
        res = self.client.get("/jobs/job/results", headers=self.auth)
        assert res.status_code == 200
        return res.json()["results"][0]["measured"]

    def test_should_offer_every_operating_point_the_ui_shows(self: Self) -> None:
        """One block per alpha in the UI's selector, keyed the way the frontend looks them up."""
        measured = self._measured(_stored_job(make_result()))
        assert set(measured) == {"0.01", "0.001", "0.0001"}

    def test_should_agree_with_the_assessment_at_the_same_operating_point(self: Self) -> None:
        """Pre-assessment and assessed views report the same attack and the same figures."""
        from leakpro.risk import UseCaseProfile, assess_risk

        results = (make_result(separation=0.5, name="weak", seed=1), make_result(separation=2.0, name="strong", seed=2))
        served = self._measured(_stored_job(*results))["0.01"]
        assessed = assess_risk(list(results), UseCaseProfile(tolerated_fpr=0.01)).measured
        assert served["measured"]["attack_name"] == assessed.attack_name == "strong"
        for field in ("success_rate", "lift", "advantage", "n_exposed_audit"):
            assert served["measured"][field] == pytest.approx(getattr(assessed, field))

    def test_should_report_skipped_inverted_attacks_alongside_the_measurement(self: Self) -> None:
        """An inverted attack is excluded by the library and named, so the UI can show it as a caveat."""
        served = self._measured(_stored_job(make_result(name="ok"),
                                            make_result(separation=2.0, name="flipped", inverted=True)))["0.01"]
        assert served["measured"]["attack_name"] == "ok"
        assert any("'flipped' skipped" in s and "inverted" in s for s in served["skipped"])

    def test_should_return_an_error_rather_than_fail_when_every_attack_is_inverted(self: Self) -> None:
        """No usable attack is a per-alpha message, not a 500 that loses the rest of the results."""
        served = self._measured(_stored_job(make_result(separation=2.0, name="flipped", inverted=True)))["0.01"]
        assert "measured" not in served
        assert "No attack result can support" in served["error"]

    def test_should_mark_an_operating_point_the_audit_set_cannot_resolve(self: Self) -> None:
        """400 non-members cannot express 0.01%: the block says unresolvable, so a TPR of 0 is not shown as safe."""
        measured = self._measured(_stored_job(make_result()))
        assert measured["0.01"]["measured"]["resolvable"] is True
        assert measured["0.0001"]["measured"]["resolvable"] is False

    def test_should_not_persist_the_derived_block_into_the_stored_job(self: Self) -> None:
        """The block is derived on request; the job keeps only what the worker stored."""
        job = _stored_job(make_result())
        self._measured(job)
        assert "measured" not in job["results"][0]
