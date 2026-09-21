#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Leakage risk assessment: from measured attack success to a use-case risk statement.

The audit measures how well an attack succeeds. Turning that into risk needs one thing LeakPro cannot
observe: how bad a successful attack would be for the people in the training data. This package keeps
those two halves visibly separate and combines them without inventing weights.

Usage::

    from leakpro import LeakPro
    from leakpro.risk import UseCaseProfile, assess_risk

    leakpro = LeakPro(MyHandler, "audit.yaml")
    results = leakpro.run_audit()

    profile = UseCaseProfile(tolerated_fpr=0.01, attacker_prior=0.01, data_type_sensitivity=3.0)
    assessment = assess_risk(results, profile, num_train=leakpro.handler.target_model_metadata.num_train)
    assessment.save(f"{leakpro.report_dir}/risk_assessment.json")

A profile declared in ``audit.yaml`` under a top-level ``use_case:`` block is available as
``leakpro.use_case_profile``. Assessment is always an explicit call: ``run_audit`` neither computes
nor writes it.

Version 1 covers membership inference only. Model inversion, gradient inversion and synthetic data
each need their own definition of attack success before they can produce a
:class:`~leakpro.risk.schemas.VulnerabilityMeasurement`.
"""

from leakpro.risk.assessment import assess_risk, loss_magnitude, positive_predictive_value
from leakpro.risk.policy import (
    POLICY_VERSION,
    SUGGESTED_SENSITIVITY_SCALE,
    VULNERABILITY_BANDS,
    resolve_vulnerability_band,
)
from leakpro.risk.render import to_latex, to_markdown
from leakpro.risk.schemas import (
    CombinedRisk,
    DeclaredInputs,
    MiaVulnerability,
    RiskAssessment,
    UseCaseProfile,
    VulnerabilityMeasurement,
)
from leakpro.risk.vulnerability import (
    InvertedResultError,
    NoUsableResultError,
    UnresolvableOperatingPointError,
    measure_mia,
    result_from_mapping,
    strongest_for_risk,
)

__all__ = [
    "POLICY_VERSION",
    "SUGGESTED_SENSITIVITY_SCALE",
    "VULNERABILITY_BANDS",
    "CombinedRisk",
    "DeclaredInputs",
    "InvertedResultError",
    "MiaVulnerability",
    "NoUsableResultError",
    "RiskAssessment",
    "UnresolvableOperatingPointError",
    "UseCaseProfile",
    "VulnerabilityMeasurement",
    "assess_risk",
    "loss_magnitude",
    "measure_mia",
    "positive_predictive_value",
    "resolve_vulnerability_band",
    "result_from_mapping",
    "strongest_for_risk",
    "to_latex",
    "to_markdown",
]
