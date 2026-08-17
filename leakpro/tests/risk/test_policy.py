"""Tests for the advisory policy data and the renderers."""

import pytest

from leakpro.risk.assessment import assess_risk
from leakpro.risk.policy import (
    POLICY_VERSION,
    SUGGESTED_SENSITIVITY_SCALE,
    VULNERABILITY_BANDS,
    VULNERABILITY_BANDS_SOURCE,
    resolve_vulnerability_band,
)
from leakpro.risk.render import to_latex, to_markdown
from leakpro.risk.schemas import UseCaseProfile


class TestVulnerabilityBands:
    """Bands label the measured lift only, and are explicitly advisory."""

    @pytest.mark.parametrize(("lift", "label"), [
        (0.5, "NONE"), (1.0, "LOW"), (2.9, "LOW"), (3.0, "MODERATE"),
        (9.9, "MODERATE"), (10.0, "HIGH"), (99.9, "HIGH"), (100.0, "SEVERE"), (5000.0, "SEVERE"),
    ])
    def test_should_map_lift_to_the_documented_band(self, lift: float, label: str) -> None:
        """Boundaries are inclusive at the lower bound, as documented."""
        assert resolve_vulnerability_band(lift) == label

    def test_should_accept_a_caller_supplied_band_table(self) -> None:
        """Users must be able to substitute their own policy, since ours is not calibrated."""
        custom = [(50.0, "UNACCEPTABLE"), (0.0, "ACCEPTABLE")]
        assert resolve_vulnerability_band(60.0, custom) == "UNACCEPTABLE"
        assert resolve_vulnerability_band(10.0, custom) == "ACCEPTABLE"

    def test_should_declare_the_bands_as_heuristic(self) -> None:
        """The provenance string must not claim a source the bands do not have."""
        assert "heuristic" in VULNERABILITY_BANDS_SOURCE

    def test_should_keep_bands_sorted_from_highest_bound_down(self) -> None:
        """Resolution walks the table in order, so ordering is a correctness property."""
        bounds = [bound for bound, _ in VULNERABILITY_BANDS]
        assert bounds == sorted(bounds, reverse=True)

    def test_should_expose_a_policy_version(self) -> None:
        """Any emitted band must be traceable to a version of this table."""
        assert POLICY_VERSION


class TestSuggestedSensitivityScale:
    """The suggested scale is sourced, unlike the bands."""

    def test_should_offer_the_four_cnil_severity_levels(self) -> None:
        """CNIL PIA-3 defines negligible, limited, significant and maximum."""
        assert set(SUGGESTED_SENSITIVITY_SCALE) == {"negligible", "limited", "significant", "maximum"}

    def test_should_increase_monotonically_with_severity(self) -> None:
        """A scale that is not ordered would silently invert the loss magnitude."""
        values = [SUGGESTED_SENSITIVITY_SCALE[k] for k in ("negligible", "limited", "significant", "maximum")]
        assert values == sorted(values)


class TestRendering:
    """Both renderers must show the inputs next to any derived figure."""

    def test_markdown_should_show_measured_declared_and_combined_sections(self, result) -> None:
        """A reader must be able to see which half of the assessment each number came from."""
        assessment = assess_risk([result], UseCaseProfile(tolerated_fpr=0.01, n_subjects=1000))
        markdown = to_markdown(assessment)
        assert "## Measured" in markdown
        assert "## Declared" in markdown
        assert "## Combined" in markdown
        assert "## Assumptions" in markdown

    def test_markdown_should_print_band_provenance_next_to_the_band(self, result) -> None:
        """A bare band label with no policy is the failure mode this layer replaces."""
        markdown = to_markdown(assess_risk([result], UseCaseProfile(tolerated_fpr=0.01)))
        assert "heuristic" in markdown

    def test_markdown_should_label_absent_figures_rather_than_printing_zero(self, result) -> None:
        """A withheld figure must read as withheld, never as zero risk."""
        markdown = to_markdown(assess_risk([result], UseCaseProfile(tolerated_fpr=0.01)))
        assert "not reported" in markdown

    def test_latex_should_produce_a_risk_section_with_assumptions(self, result) -> None:
        """The PDF section carries the same assumption list as the JSON."""
        latex = to_latex(assess_risk([result], UseCaseProfile(tolerated_fpr=0.01, n_subjects=10)))
        assert "\\section{Risk assessment}" in latex
        assert "\\subsection*{Assumptions}" in latex
        assert "\\begin{tabularx}" in latex

    def test_latex_should_escape_special_characters_from_free_text(self, result) -> None:
        """User notes containing LaTeX metacharacters must not break compilation."""
        profile = UseCaseProfile(tolerated_fpr=0.01, notes="100% of records & 50_000 subjects")
        latex = to_latex(assess_risk([result], profile))
        assert "100\\%" in latex or "100\\% of records" in latex
        assert "50\\_000" in latex
