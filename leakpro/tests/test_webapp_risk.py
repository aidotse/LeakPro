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
