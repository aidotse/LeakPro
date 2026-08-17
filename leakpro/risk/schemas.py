#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Schemas for the leakage risk assessment layer.

The assessment is reported as three blocks that are never collapsed into a single opaque score:

* :class:`VulnerabilityMeasurement` (and its MIA subclass) holds what the audit *measured*. Every
  field is reproducible by re-running the audit.
* :class:`DeclaredInputs` echoes what the data controller *declared*. None of it is measurable by
  LeakPro, and it is reproduced verbatim in the output so a reader can see what was assumed.
* :class:`CombinedRisk` holds figures derived from the two blocks above. Every one of them is
  traceable to measured or declared inputs, and :attr:`RiskAssessment.assumptions` names the
  assumption behind each derived quantity.

See ``leakpro/risk/policy.py`` for the provenance of the advisory bands and the suggested
sensitivity scale.
"""

import json

from pydantic import BaseModel, ConfigDict, Field

from leakpro.utils.import_helper import List, Optional, Self


class UseCaseProfile(BaseModel):
    """Harm parameters declared by the data controller.

    These cannot be measured by LeakPro: they describe the deployment context, not the model. The
    factor names follow the Loss Magnitude decomposition of Sion et al. (IWPE 2019,
    doi:10.1109/SPW.2019.00023), ``LM = DTS x NR x DST x NDS``, so that a reader can map each field
    onto the published model. That paper deliberately supplies no numeric values for the factors;
    see :data:`leakpro.risk.policy.SUGGESTED_SENSITIVITY_SCALE` for a cited starting point.
    """

    tolerated_fpr: float = Field(..., gt=0.0, lt=1.0,
                                 description="Operating point alpha: the false-positive rate the attacker is assumed to "
                                             "tolerate. Required on purpose - it changes the headline number and the "
                                             "choice belongs to you. Use 0.01 unless you have a reason to go lower; "
                                             "smaller values are unresolvable on small audit sets.")
    attacker_prior: float = Field(default=0.5, gt=0.0, lt=1.0,
                                  description="pi: probability that a record in the attacker's candidate pool is "
                                              "actually a member. The default 0.5 is the balanced-audit assumption "
                                              "baked into every TPR and AUC LeakPro reports, and it overstates "
                                              "precision for realistic pools (Jayaraman et al., PoPETs 2021).")
    n_subjects: Optional[int] = Field(default=None, ge=1,
                                      description="NDS: number of data subjects in the training population. Defaults "
                                                  "to the target model's num_train when available.")
    records_per_subject: float = Field(default=1.0, gt=0.0,
                                       description="NR: records of this data type per data subject. Fractions are "
                                                   "allowed when only some subjects contribute the data type.")
    data_type_sensitivity: float = Field(default=1.0, gt=0.0,
                                         description="DTS: sensitivity of the leaked data type on a numeric scale you "
                                                     "choose. 1.0 is neutral. See policy.SUGGESTED_SENSITIVITY_SCALE.")
    subject_type_weight: float = Field(default=1.0, gt=0.0,
                                       description="DST: weight for the data subject type, capturing vulnerable "
                                                   "subjects such as minors or patients. 1.0 is neutral.")
    cost_per_exposed_subject: Optional[float] = Field(default=None, ge=0.0,
                                                      description="Cost to the organization per exposed subject, in any "
                                                                  "unit you choose. Left unset, no monetary figure is "
                                                                  "reported at all rather than a fabricated one.")
    extrapolate_to_population: bool = Field(default=False,
                                            description="Scale the exposed-record count from the audit set up to the "
                                                        "training population. Off by default: per-record vulnerability "
                                                        "is strongly non-uniform, so the extrapolation is an estimate "
                                                        "under an assumption, not a measurement.")
    notes: str = Field(default="", description="Free text kept in the output for the audit trail.")

    model_config = ConfigDict(extra="forbid")  # Prevent extra fields


class VulnerabilityMeasurement(BaseModel):
    """Family-agnostic measured block.

    Any attack family that wants to feed the risk layer produces one of these. Only MIA is
    implemented in v1 (:class:`MiaVulnerability`); model inversion, gradient inversion and synthetic
    data each need their own success definition first. Nothing in ``assessment.py`` may depend on
    fields below this class.
    """

    attack_name: str = Field(..., description="Attack the measurement came from")
    operating_point: float = Field(..., gt=0.0, lt=1.0, description="alpha for MIA; family-specific elsewhere")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="TPR at the operating point for MIA")
    baseline_rate: float = Field(..., gt=0.0, lt=1.0, description="Random-guess rate at the operating point (alpha)")
    n_at_risk: int = Field(..., ge=0, description="Records the measurement covers (audit-set members for MIA)")
    resolvable: bool = Field(..., description="False when the audit set cannot resolve the operating point")
    provenance: dict = Field(default_factory=dict, description="Attack config and result id")

    model_config = ConfigDict(extra="forbid")


class MiaVulnerability(VulnerabilityMeasurement):
    """Measured block for membership inference."""

    roc_auc: Optional[float] = Field(default=None, description="Context only. Not used in any risk figure")
    advantage: float = Field(..., description="TPR(alpha) - alpha: absolute gain over random guessing")
    lift: float = Field(..., ge=0.0, description="TPR(alpha) / alpha: multiplicative gain over random guessing")
    n_members_audit: int = Field(..., ge=0, description="Members in the audit set")
    n_non_members_audit: int = Field(..., ge=0, description="Non-members in the audit set")
    n_exposed_audit: Optional[int] = Field(default=None, ge=0,
                                           description="Members flagged at the threshold yielding FPR <= alpha. None "
                                                       "when the result carries no signal values")
    exposed_audit_indices: Optional[List[int]] = Field(default=None,
                                                       description="Dataset indices of the exposed members. Usually None: "
                                                                   "the live MIAResult does not carry audit indices, so "
                                                                   "exposed records can be counted but not named")
    min_resolvable_fpr: float = Field(..., gt=0.0, description="1 / n_non_members_audit: the finest FPR this audit "
                                                              "set can express")
    train_test_gap: Optional[float] = Field(default=None, description="Context only: train accuracy minus test accuracy")
    dp_epsilon: Optional[float] = Field(default=None, description="Context only: DP-SGD epsilon of the target")

    model_config = ConfigDict(extra="forbid")


class DeclaredInputs(BaseModel):
    """Verbatim echo of the use-case profile, plus where n_subjects came from."""

    tolerated_fpr: float
    attacker_prior: float
    n_subjects: Optional[int] = None
    n_subjects_source: str = Field(..., description="'declared', 'target num_train', or 'unavailable'")
    records_per_subject: float
    data_type_sensitivity: float
    subject_type_weight: float
    cost_per_exposed_subject: Optional[float] = None
    extrapolate_to_population: bool
    notes: str = ""

    model_config = ConfigDict(extra="forbid")


class CombinedRisk(BaseModel):
    """Figures derived from the measured and declared blocks."""

    ppv: Optional[float] = Field(default=None, ge=0.0, le=1.0,
                                 description="Precision of the attacker at the declared prior: "
                                             "TPR / (TPR + gamma*alpha) with gamma = (1-pi)/pi. Jayaraman et al. Thm 4.2")
    ppv_balanced: Optional[float] = Field(default=None, ge=0.0, le=1.0,
                                          description="Same at pi = 0.5, the assumption built into the audit protocol")
    gamma: float = Field(..., gt=0.0, description="(1 - attacker_prior) / attacker_prior")
    loss_magnitude: Optional[float] = Field(default=None, ge=0.0,
                                            description="Sion LM = DTS x NR x DST x NDS. None without n_subjects")
    loss_event_frequency: float = Field(..., ge=0.0,
                                        description="Sion LEF. Equals the measured success rate under the "
                                                    "single-attempt assumption")
    risk: Optional[float] = Field(default=None, ge=0.0,
                                  description="Sion Risk = LM x LEF, in sensitivity-weighted expected exposed records. "
                                              "With neutral factors this equals expected_exposed_subjects")
    expected_exposed_subjects: Optional[float] = Field(default=None, ge=0.0,
                                                       description="success_rate x n_subjects: unweighted and directly "
                                                                   "interpretable")
    expected_cost: Optional[float] = Field(default=None, ge=0.0,
                                           description="Omitted unless a cost per exposed subject was declared")
    extrapolated_exposed_records: Optional[float] = Field(default=None, ge=0.0,
                                                          description="Audit-set exposed count scaled to the training "
                                                                      "population. Only when explicitly requested")
    vulnerability_band: Optional[str] = Field(default=None,
                                              description="Advisory label over the measured lift only. Never a "
                                                          "combined risk label - see policy.py")

    model_config = ConfigDict(extra="forbid")


class RiskAssessment(BaseModel):
    """Complete assessment: measured, declared, combined, plus provenance."""

    measured: MiaVulnerability
    declared: DeclaredInputs
    combined: CombinedRisk
    assumptions: List[str] = Field(default_factory=list,
                                   description="One entry per derived figure or degraded path, naming the assumption")
    warnings: List[str] = Field(default_factory=list, description="Conditions that make the assessment unsafe to quote")
    policy_version: str = Field(..., description="Version of the band policy used")
    leakpro_version: str = Field(..., description="LeakPro version that produced the assessment")

    model_config = ConfigDict(extra="forbid")

    def save(self: Self, path: str) -> None:
        """Write the assessment to ``path`` as JSON.

        Args:
        ----
            path: Destination file path.

        """
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    @staticmethod
    def load(path: str) -> "RiskAssessment":
        """Read an assessment back from a JSON file written by :meth:`save`.

        Args:
        ----
            path: Path to the JSON file.

        Returns:
        -------
            The deserialized assessment.

        """
        with open(path) as f:
            return RiskAssessment(**json.load(f))
