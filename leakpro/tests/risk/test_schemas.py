"""Tests for the risk schemas, including the deliberate absence of defaults."""

import json

import pytest
from pydantic import ValidationError

from leakpro.risk.assessment import assess_risk
from leakpro.risk.schemas import RiskAssessment, UseCaseProfile


class TestUseCaseProfile:
    """Validation of the declared harm parameters."""

    def test_should_require_the_operating_point(self) -> None:
        """alpha has no default: the choice changes the headline number and belongs to the user."""
        with pytest.raises(ValidationError, match="tolerated_fpr"):
            UseCaseProfile()

    def test_should_reject_unknown_fields(self) -> None:
        """A typo in a profile must fail rather than be silently ignored."""
        with pytest.raises(ValidationError):
            UseCaseProfile(tolerated_fpr=0.01, sensitivity="high")

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
    def test_should_reject_operating_points_outside_the_open_unit_interval(self, alpha: float) -> None:
        """An FPR of 0 or 1 is not an operating point an attacker can occupy."""
        with pytest.raises(ValidationError):
            UseCaseProfile(tolerated_fpr=alpha)

    @pytest.mark.parametrize("prior", [0.0, 1.0, -0.5])
    def test_should_reject_priors_outside_the_open_unit_interval(self, prior: float) -> None:
        """A prior of 0 or 1 makes precision undefined or trivial."""
        with pytest.raises(ValidationError):
            UseCaseProfile(tolerated_fpr=0.01, attacker_prior=prior)

    @pytest.mark.parametrize("field", ["records_per_subject", "data_type_sensitivity", "subject_type_weight"])
    def test_should_reject_non_positive_loss_magnitude_factors(self, field: str) -> None:
        """A zero factor would silently zero out the whole loss magnitude product."""
        with pytest.raises(ValidationError):
            UseCaseProfile(**{"tolerated_fpr": 0.01, field: 0.0})

    def test_should_default_to_neutral_factors_and_a_balanced_prior(self) -> None:
        """Defaults are neutral, so an unconfigured profile weights nothing invisibly."""
        profile = UseCaseProfile(tolerated_fpr=0.01)
        assert (profile.data_type_sensitivity, profile.subject_type_weight, profile.records_per_subject) == (1.0, 1.0, 1.0)
        assert profile.attacker_prior == 0.5
        assert profile.cost_per_exposed_subject is None
        assert profile.extrapolate_to_population is False

    def test_should_reject_a_negative_cost(self) -> None:
        """A negative cost per exposed subject is not a meaningful input."""
        with pytest.raises(ValidationError):
            UseCaseProfile(tolerated_fpr=0.01, cost_per_exposed_subject=-1.0)


class TestRiskAssessmentSerialisation:
    """Round-tripping an assessment through disk."""

    def test_should_round_trip_through_json(self, result, tmp_path) -> None:
        """A saved assessment must reload into an equal object, for later re-reporting."""
        assessment = assess_risk([result], UseCaseProfile(tolerated_fpr=0.01, n_subjects=100))
        path = tmp_path / "risk_assessment.json"
        assessment.save(str(path))
        assert RiskAssessment.load(str(path)).model_dump_json() == assessment.model_dump_json()

    def test_should_write_all_three_blocks_plus_provenance(self, result, tmp_path) -> None:
        """The file must be self-contained: measured, declared, combined, assumptions, versions."""
        assessment = assess_risk([result], UseCaseProfile(tolerated_fpr=0.01))
        path = tmp_path / "risk_assessment.json"
        assessment.save(str(path))
        with open(path) as f:
            payload = json.load(f)
        assert set(payload) >= {"measured", "declared", "combined", "assumptions", "warnings",
                                "policy_version", "leakpro_version"}
        assert payload["assumptions"], "an assessment without stated assumptions is not auditable"
