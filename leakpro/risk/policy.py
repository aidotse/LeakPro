#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Advisory policy data for the risk layer, with the provenance of every value.

Two things live here, and neither is a measurement.

**Suggested sensitivity scale.** Sion et al. (*Privacy Risk Assessment for Data Subject-aware Threat
Modeling*, IWPE 2019, doi:10.1109/SPW.2019.00023) define Loss Magnitude as a product of numeric
factors but deliberately supply no values for them: section III-G states every input is an analyst
estimate, and section VI-A points at "GDPR sensitivity interpretations" and "the severity scale from
the CNIL PIA knowledge bases" as sources for reusable sensitivity values. :data:`SUGGESTED_SENSITIVITY_SCALE`
reproduces the CNIL PIA severity levels (PIA-3, *Knowledge Bases*, February 2018 edition: negligible,
limited, significant, maximum) as a starting point. It is a suggestion, not a default: the profile
field defaults to a neutral 1.0 and users are expected to substitute their own scale.

**Vulnerability bands.** :data:`VULNERABILITY_BANDS` label the measured lift (TPR at alpha divided by
alpha). They are ``heuristic``: no published source defines thresholds for this quantity, and we do
not pretend otherwise. They exist so a UI can colour a card, and they deliberately describe *only*
the measured side.

There is no combined risk band. A label over the combination of a measurement and a value judgement
would require weights nobody has published, which is exactly the failure mode this module replaces.
Report :attr:`leakpro.risk.schemas.CombinedRisk.risk` and the expected exposure count as numbers,
next to the inputs that produced them.
"""

from leakpro.utils.import_helper import Dict, List, Optional, Tuple

POLICY_VERSION = "2026-08-17.1"

# CNIL PIA-3 "Knowledge Bases" (Feb 2018) severity levels, offered as a starting scale for the
# Data Type Sensitivity (DTS) and Data Subject Type (DST) factors. Sourced, not invented, but still
# a policy choice: substitute your own scale when you have one.
SUGGESTED_SENSITIVITY_SCALE: Dict[str, float] = {
    "negligible": 1.0,
    "limited": 2.0,
    "significant": 3.0,
    "maximum": 4.0,
}
SUGGESTED_SENSITIVITY_SOURCE = (
    "CNIL, Privacy Impact Assessment (PIA-3): Knowledge Bases, February 2018 - severity scale "
    "(negligible / limited / significant / maximum), as recommended by Sion et al. IWPE 2019 SVI-A."
)

# (lower bound on lift, inclusive) -> label. Heuristic: see the module docstring.
VULNERABILITY_BANDS: List[Tuple[float, str]] = [
    (100.0, "SEVERE"),
    (10.0, "HIGH"),
    (3.0, "MODERATE"),
    (1.0, "LOW"),
    (0.0, "NONE"),
]
VULNERABILITY_BANDS_SOURCE = (
    "heuristic - no published thresholds exist for TPR/alpha lift. Round numbers chosen so that "
    "'NONE' means no better than random at the operating point and 'SEVERE' means two orders of "
    "magnitude better. Override with your own bands rather than treating these as calibrated."
)


def resolve_vulnerability_band(lift: float, bands: Optional[List[Tuple[float, str]]] = None) -> str:
    """Return the advisory band label for a measured lift.

    Args:
    ----
        lift: Measured TPR at alpha divided by alpha.
        bands: Optional override, as ``(lower_bound, label)`` pairs sorted from highest bound down.

    Returns:
    -------
        The label of the first band whose lower bound the lift meets.

    """
    table = bands if bands is not None else VULNERABILITY_BANDS
    for lower_bound, label in table:
        if lift >= lower_bound:
            return label
    return table[-1][1]
