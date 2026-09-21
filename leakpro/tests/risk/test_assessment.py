"""Tests for the combination of measured vulnerability with declared harm parameters."""

import pytest

from leakpro.risk.assessment import assess_risk, loss_magnitude, positive_predictive_value
from leakpro.risk.schemas import UseCaseProfile
from leakpro.tests.risk.conftest import make_result


class TestPositivePredictiveValue:
    """PPV per Jayaraman et al. (PoPETs 2021) Theorem 4.2."""

    @pytest.mark.parametrize(("prior", "expected"), [
        # PPV = TPR / (TPR + gamma*alpha), gamma = (1-p)/p, with TPR = 0.10 and alpha = 0.01.
        (0.5, 0.10 / (0.10 + 1.0 * 0.01)),          # gamma = 1     -> 0.9091
        (0.01, 0.10 / (0.10 + 99.0 * 0.01)),        # gamma = 99    -> 0.0917
        (0.001, 0.10 / (0.10 + 999.0 * 0.01)),      # gamma = 999   -> 0.0099
    ])
    def test_should_match_hand_computed_precision_at_each_prior(self, prior: float, expected: float) -> None:
        """Precision collapses as the prior skews, which is the entire point of reporting it."""
        assert positive_predictive_value(0.10, 0.01, prior) == pytest.approx(expected)

    def test_should_equal_the_equivalent_prior_weighted_form(self) -> None:
        """The gamma form and the prior-weighted form are algebraically identical."""
        tpr, alpha, prior = 0.25, 0.001, 0.02
        weighted = prior * tpr / (prior * tpr + (1 - prior) * alpha)
        assert positive_predictive_value(tpr, alpha, prior) == pytest.approx(weighted)

    def test_should_be_zero_when_the_attack_flags_nothing(self) -> None:
        """A zero success rate yields zero precision rather than a division error."""
        assert positive_predictive_value(0.0, 0.01, 0.5) == 0.0


class TestLossMagnitude:
    """Sion et al. LM = DTS x NR x DST x NDS."""

    def test_should_multiply_all_four_declared_factors(self) -> None:
        """The product is exactly the published decomposition, with no extra scaling."""
        profile = UseCaseProfile(tolerated_fpr=0.01, records_per_subject=2.0,
                                 data_type_sensitivity=3.0, subject_type_weight=4.0)
        assert loss_magnitude(profile, 1000) == pytest.approx(3.0 * 2.0 * 4.0 * 1000)

    def test_should_reduce_to_the_subject_count_with_neutral_factors(self) -> None:
        """Neutral factors leave the interpretable special case: one unit of harm per subject."""
        assert loss_magnitude(UseCaseProfile(tolerated_fpr=0.01), 500) == pytest.approx(500.0)

    def test_should_be_unavailable_without_a_subject_count(self) -> None:
        """No population size means no loss magnitude, rather than a guessed one."""
        assert loss_magnitude(UseCaseProfile(tolerated_fpr=0.01), None) is None


class TestAssessRisk:
    """End-to-end combination on real results."""

    def test_should_derive_risk_as_loss_magnitude_times_success_rate(self, result) -> None:
        """Risk is Sion's LM x LEF, with LEF the measured success rate."""
        profile = UseCaseProfile(tolerated_fpr=0.01, n_subjects=10_000, data_type_sensitivity=3.0)
        assessment = assess_risk([result], profile)
        expected = 3.0 * 1.0 * 1.0 * 10_000 * assessment.measured.success_rate
        assert assessment.combined.risk == pytest.approx(expected)
        assert assessment.combined.loss_event_frequency == pytest.approx(assessment.measured.success_rate)

    def test_should_report_expected_exposed_subjects_unweighted(self, result) -> None:
        """The interpretable count ignores the sensitivity weights on purpose."""
        profile = UseCaseProfile(tolerated_fpr=0.01, n_subjects=10_000, data_type_sensitivity=3.0)
        assessment = assess_risk([result], profile)
        assert assessment.combined.expected_exposed_subjects == pytest.approx(
            assessment.measured.success_rate * 10_000)

    def test_should_report_both_declared_and_balanced_precision(self, result) -> None:
        """The balanced value is the audit protocol's own assumption and travels alongside."""
        assessment = assess_risk([result], UseCaseProfile(tolerated_fpr=0.01, attacker_prior=0.001))
        assert assessment.combined.ppv < assessment.combined.ppv_balanced
        assert assessment.combined.gamma == pytest.approx(999.0)

    def test_should_omit_cost_entirely_when_none_was_declared(self, result) -> None:
        """No declared cost means no monetary figure, not a zero that reads as "no impact"."""
        assessment = assess_risk([result], UseCaseProfile(tolerated_fpr=0.01, n_subjects=100))
        assert assessment.combined.expected_cost is None
        assert any("no monetary figure" in a for a in assessment.assumptions)

    def test_should_compute_cost_when_one_was_declared(self, result) -> None:
        """Cost scales the expected exposure by the declared per-subject cost."""
        profile = UseCaseProfile(tolerated_fpr=0.01, n_subjects=100, cost_per_exposed_subject=250.0)
        assessment = assess_risk([result], profile)
        assert assessment.combined.expected_cost == pytest.approx(
            assessment.combined.expected_exposed_subjects * 250.0)

    def test_should_take_the_subject_count_from_num_train_when_not_declared(self, result) -> None:
        """The target's training-set size is the natural default, and its use is recorded."""
        assessment = assess_risk([result], UseCaseProfile(tolerated_fpr=0.01), num_train=1234)
        assert assessment.declared.n_subjects == 1234
        assert assessment.declared.n_subjects_source == "target num_train"

    def test_should_report_audit_set_figures_only_without_a_population_size(self, result) -> None:
        """Absent num_train and a declared count, population figures are withheld, not invented."""
        assessment = assess_risk([result], UseCaseProfile(tolerated_fpr=0.01))
        assert assessment.declared.n_subjects_source == "unavailable"
        assert assessment.combined.expected_exposed_subjects is None
        assert assessment.combined.risk is None
        assert assessment.combined.ppv is not None, "measured-side figures must survive"

    def test_should_prefer_a_declared_subject_count_over_num_train(self, result) -> None:
        """An explicit declaration wins: subjects and training records are not the same thing."""
        profile = UseCaseProfile(tolerated_fpr=0.01, n_subjects=50)
        assessment = assess_risk([result], profile, num_train=99_999)
        assert (assessment.declared.n_subjects, assessment.declared.n_subjects_source) == (50, "declared")

    def test_should_extrapolate_only_when_requested_and_flag_the_assumption(self, result) -> None:
        """Extrapolation is opt-in and always carries its representativeness caveat."""
        off = assess_risk([result], UseCaseProfile(tolerated_fpr=0.01), num_train=4000)
        assert off.combined.extrapolated_exposed_records is None
        assert not any("Extrapolation" in a for a in off.assumptions)

        on = assess_risk([result], UseCaseProfile(tolerated_fpr=0.01, extrapolate_to_population=True),
                         num_train=4000)
        expected = on.measured.n_exposed_audit * (4000 / on.measured.n_members_audit)
        assert on.combined.extrapolated_exposed_records == pytest.approx(expected)
        assert any("privacy onion" in a for a in on.assumptions)

    def test_should_warn_when_the_prior_is_the_balanced_default(self, result) -> None:
        """Quoting balanced-prior precision as real-world risk is the mistake to flag."""
        assessment = assess_risk([result], UseCaseProfile(tolerated_fpr=0.01))
        assert any("balanced-audit assumption" in w for w in assessment.warnings)

    def test_should_record_the_single_attempt_and_independence_assumptions(self, result) -> None:
        """Both simplifications inherited from the published model must be stated, not hidden."""
        assumptions = " ".join(assess_risk([result], UseCaseProfile(tolerated_fpr=0.01)).assumptions)
        assert "single attack attempt" in assumptions
        assert "independent" in assumptions

    def test_should_withhold_derived_figures_when_the_operating_point_is_unresolvable(self) -> None:
        """A non-strict assessment degrades explicitly instead of reporting zero risk."""
        small = make_result(n_members=200, n_non_members=200)
        assessment = assess_risk([small], UseCaseProfile(tolerated_fpr=0.001, n_subjects=100),
                                 strict=False)
        assert assessment.measured.resolvable is False
        assert (assessment.combined.ppv, assessment.combined.risk) == (None, None)
        assert assessment.combined.vulnerability_band is None
        assert any("not measurable" in w for w in assessment.warnings)

    def test_should_accept_a_single_result_as_well_as_a_list(self, result) -> None:
        """Callers should not have to wrap one result in a list."""
        assert assess_risk(result, UseCaseProfile(tolerated_fpr=0.01)).measured.attack_name == "lira"

    def test_should_be_deterministic_for_identical_inputs(self, result) -> None:
        """The scored output carries no timestamps or randomness, so it must be byte-identical."""
        profile = UseCaseProfile(tolerated_fpr=0.01, n_subjects=1000, notes="run twice")
        first = assess_risk([result], profile, num_train=1000)
        second = assess_risk([result], profile, num_train=1000)
        assert first.model_dump_json() == second.model_dump_json()

    def test_should_carry_policy_and_version_provenance(self, result) -> None:
        """Any band that is emitted must be traceable to a policy version."""
        assessment = assess_risk([result], UseCaseProfile(tolerated_fpr=0.01))
        assert assessment.policy_version
        assert assessment.leakpro_version
