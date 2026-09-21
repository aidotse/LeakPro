#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Combination of measured vulnerability with declared harm parameters.

The structure follows Sion et al. (*Privacy Risk Assessment for Data Subject-aware Threat Modeling*,
IWPE 2019, doi:10.1109/SPW.2019.00023):

.. code-block:: text

    Risk = LM x LEF
    LM   = DTS x NR x DST x NDS          all four declared by the data controller
    LEF  = V x (RP x TEF)                V is the probability of a successful attack

LeakPro's contribution is that ``V`` is *measured* rather than elicited from experts: it is the true
positive rate of the strongest attack at the operating point the user chose. ``RP x TEF`` (retention
period times threat event frequency) is assumed to be 1, i.e. a single attack attempt against a
retained model, because LeakPro cannot observe adversary behaviour. That assumption is recorded in
the output rather than hidden.

Precision is reported per Jayaraman et al. (*Revisiting Membership Inference Under Realistic
Assumptions*, PoPETs 2021, Theorem 4.2): ``PPV = TPR / (TPR + gamma * FPR)`` with
``gamma = (1 - pi) / pi``. Reporting it is not optional decoration. A balanced audit implies
``pi = 0.5``, and at realistic priors the same measurement can mean a very different threat.

Nothing in this module imports from ``leakpro.attacks``; it consumes the family-agnostic measured
block so that other attack families can feed the same combination later.
"""

from leakpro.risk.policy import POLICY_VERSION, resolve_vulnerability_band
from leakpro.risk.schemas import (
    CombinedRisk,
    DeclaredInputs,
    MiaVulnerability,
    RiskAssessment,
    UseCaseProfile,
)
from leakpro.risk.vulnerability import measure_mia, strongest_for_risk
from leakpro.utils.import_helper import Any, List, Optional, Tuple

LEAKPRO_VERSION = "0.1.0"

_BALANCED_PRIOR = 0.5


def positive_predictive_value(success_rate: float, alpha: float, prior: float) -> float:
    """Return the attacker's precision at a given membership prior.

    Implements ``PPV = TPR / (TPR + gamma * alpha)`` with ``gamma = (1 - prior) / prior``
    (Jayaraman et al., PoPETs 2021, Theorem 4.2). Algebraically identical to
    ``prior * TPR / (prior * TPR + (1 - prior) * alpha)``.

    Args:
    ----
        success_rate: True positive rate at the operating point, as a fraction.
        alpha: Operating point (false-positive rate), as a fraction.
        prior: Probability that a candidate record is a member.

    Returns:
    -------
        Precision in [0, 1]. Zero when the attack flags nothing.

    """
    gamma = (1.0 - prior) / prior
    denominator = success_rate + gamma * alpha
    if denominator <= 0.0:
        return 0.0
    return success_rate / denominator


def loss_magnitude(profile: UseCaseProfile, n_subjects: Optional[int]) -> Optional[float]:
    """Return Sion's Loss Magnitude, ``DTS x NR x DST x NDS``.

    Args:
    ----
        profile: The declared use-case profile.
        n_subjects: Resolved number of data subjects (NDS), or None when unknown.

    Returns:
    -------
        The product, or None when the number of subjects is unknown.

    """
    if n_subjects is None:
        return None
    return (profile.data_type_sensitivity
            * profile.records_per_subject
            * profile.subject_type_weight
            * float(n_subjects))


def _resolve_n_subjects(profile: UseCaseProfile, num_train: Optional[int]) -> Tuple[Optional[int], str]:
    """Decide how many data subjects the assessment scales to, and say where the number came from.

    Args:
    ----
        profile: The declared use-case profile.
        num_train: Training-set size of the target model, when the caller supplied it.

    Returns:
    -------
        ``(n_subjects, source)``.

    """
    if profile.n_subjects is not None:
        return profile.n_subjects, "declared"
    if num_train is not None:
        return int(num_train), "target num_train"
    return None, "unavailable"


def _build_assumptions(profile: UseCaseProfile,
                       measured: MiaVulnerability,
                       n_subjects_source: str,
                       extrapolated: Optional[float]) -> List[str]:
    """Collect one assumption string per derived figure or degraded path.

    Args:
    ----
        profile: The declared use-case profile.
        measured: The measured block.
        n_subjects_source: Where the subject count came from.
        extrapolated: The extrapolated exposed-record count, when one was produced.

    Returns:
    -------
        The assumption strings, in reporting order.

    """
    assumptions = [
        f"Vulnerability is measured: TPR={measured.success_rate:.4f} at FPR={measured.operating_point} from attack "
        f"'{measured.attack_name}'.",
        "Loss Event Frequency equals the measured success rate: Sion's retention period times threat event frequency "
        "(RP x TEF) is assumed to be 1, i.e. a single attack attempt against a retained model. LeakPro cannot observe "
        "adversary behaviour, so raise this factor yourself if repeated attempts are plausible.",
        "Risk = LM x LEF treats the four Loss Magnitude factors as independent. Sion et al. section VI-B flags this as "
        "a simplification of reality.",
        f"Precision (PPV) is reported at the declared attacker prior pi={profile.attacker_prior}. The audit protocol "
        f"itself is balanced, which corresponds to pi=0.5 and is reported separately as ppv_balanced.",
        f"Number of data subjects: {n_subjects_source}.",
    ]
    if profile.cost_per_exposed_subject is None:
        assumptions.append("No cost per exposed subject was declared, so no monetary figure is reported. This is "
                           "deliberate: a default cost would be fabricated.")
    if measured.n_exposed_audit is None:
        assumptions.append("The result carries no per-record signal values, so the exposed-member count could not be "
                           "computed. Rates are reported without absolute counts.")
    if extrapolated is not None:
        assumptions.append(
            f"Extrapolation from {measured.n_members_audit} audited members to the training population assumes the "
            "audit sample is representative. Per-record vulnerability is strongly non-uniform (the privacy onion "
            "effect), so treat the extrapolated count as an estimate, not a measurement."
        )
    return assumptions


def _build_warnings(profile: UseCaseProfile, measured: MiaVulnerability, skipped: List[str]) -> List[str]:
    """Collect conditions that make the assessment unsafe to quote without qualification.

    Args:
    ----
        profile: The declared use-case profile.
        measured: The measured block.
        skipped: Reasons emitted while selecting the strongest attack.

    Returns:
    -------
        The warning strings.

    """
    warnings = list(skipped)
    if not measured.resolvable:
        warnings.append(
            f"alpha={measured.operating_point} is finer than the audit set can resolve (minimum "
            f"{measured.min_resolvable_fpr:.2e} with {measured.n_non_members_audit} non-members). Derived figures are "
            "withheld: a TPR of 0 here means 'not measurable', not 'no leakage'."
        )
    if profile.attacker_prior == _BALANCED_PRIOR:
        warnings.append(
            "The attacker prior is 0.5, the balanced-audit assumption. Real candidate pools are usually far more "
            "skewed, and precision falls steeply with the prior. Declare a realistic pi before quoting these figures."
        )
    if measured.n_exposed_audit is not None and 0 < measured.n_exposed_audit <= 5:
        warnings.append(
            f"Only {measured.n_exposed_audit} members were flagged at this operating point. High precision over a "
            "handful of records is a weak basis for a risk statement."
        )
    return warnings


def assess_risk(results: Any,
                profile: UseCaseProfile,
                *,
                num_train: Optional[int] = None,
                train_test_gap: Optional[float] = None,
                dp_epsilon: Optional[float] = None,
                bands: Optional[List[Tuple[float, str]]] = None,
                strict: bool = True) -> RiskAssessment:
    """Assess use-case risk for a set of MIA results.

    Called explicitly after ``LeakPro.run_audit`` rather than from inside it, so that one expensive
    audit can be re-assessed cheaply under different operating points and priors::

        results = leakpro.run_audit()
        assessment = assess_risk(results, profile, num_train=leakpro.handler.target_model_metadata.num_train)
        assessment.save("risk_assessment.json")

    Args:
    ----
        results: A ``MIAResult`` or a list of them.
        profile: Declared harm parameters.
        num_train: Training-set size of the target model. Without it the assessment reports audit-set
            figures only rather than guessing a population size.
        train_test_gap: Optional context: target train accuracy minus test accuracy.
        dp_epsilon: Optional context: DP-SGD epsilon of the target model.
        bands: Optional override for the advisory vulnerability bands.
        strict: When True, an operating point the audit set cannot resolve raises. When False, the
            assessment is returned with derived figures withheld and a warning attached.

    Returns:
    -------
        The full assessment.

    """
    candidates = list(results) if isinstance(results, (list, tuple)) else [results]
    best, skipped = strongest_for_risk(candidates, profile.tolerated_fpr)
    measured = measure_mia(best,
                           alpha=profile.tolerated_fpr,
                           strict=strict,
                           train_test_gap=train_test_gap,
                           dp_epsilon=dp_epsilon)

    n_subjects, n_subjects_source = _resolve_n_subjects(profile, num_train)
    magnitude = loss_magnitude(profile, n_subjects)

    # Derived figures are withheld entirely when the operating point was not measurable, because a
    # success rate of 0.0 there is an absence of measurement rather than an absence of leakage.
    if measured.resolvable:
        ppv = positive_predictive_value(measured.success_rate, measured.operating_point, profile.attacker_prior)
        ppv_balanced = positive_predictive_value(measured.success_rate, measured.operating_point, _BALANCED_PRIOR)
        risk = magnitude * measured.success_rate if magnitude is not None else None
        expected_exposed = measured.success_rate * n_subjects if n_subjects is not None else None
        band = resolve_vulnerability_band(measured.lift, bands)
    else:
        ppv, ppv_balanced, risk, expected_exposed, band = None, None, None, None, None

    expected_cost = None
    if expected_exposed is not None and profile.cost_per_exposed_subject is not None:
        expected_cost = expected_exposed * profile.cost_per_exposed_subject

    extrapolated = None
    if (profile.extrapolate_to_population
            and measured.resolvable
            and measured.n_exposed_audit is not None
            and measured.n_members_audit > 0
            and num_train is not None):
        extrapolated = measured.n_exposed_audit * (float(num_train) / measured.n_members_audit)

    combined = CombinedRisk(
        ppv=ppv,
        ppv_balanced=ppv_balanced,
        gamma=(1.0 - profile.attacker_prior) / profile.attacker_prior,
        loss_magnitude=magnitude,
        loss_event_frequency=measured.success_rate,
        risk=risk,
        expected_exposed_subjects=expected_exposed,
        expected_cost=expected_cost,
        extrapolated_exposed_records=extrapolated,
        vulnerability_band=band,
    )

    declared = DeclaredInputs(
        tolerated_fpr=profile.tolerated_fpr,
        attacker_prior=profile.attacker_prior,
        n_subjects=n_subjects,
        n_subjects_source=n_subjects_source,
        records_per_subject=profile.records_per_subject,
        data_type_sensitivity=profile.data_type_sensitivity,
        subject_type_weight=profile.subject_type_weight,
        cost_per_exposed_subject=profile.cost_per_exposed_subject,
        extrapolate_to_population=profile.extrapolate_to_population,
        notes=profile.notes,
    )

    return RiskAssessment(
        measured=measured,
        declared=declared,
        combined=combined,
        assumptions=_build_assumptions(profile, measured, n_subjects_source, extrapolated),
        warnings=_build_warnings(profile, measured, skipped),
        policy_version=POLICY_VERSION,
        leakpro_version=LEAKPRO_VERSION,
    )
