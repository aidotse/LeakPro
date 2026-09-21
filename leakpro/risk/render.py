#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Rendering of a risk assessment to markdown and LaTeX.

Both renderers follow the same rule: a band or a derived number never appears without the inputs that
produced it. The assumption list and the policy version are part of the output, not an appendix.
"""

from leakpro.risk.policy import VULNERABILITY_BANDS_SOURCE
from leakpro.risk.schemas import RiskAssessment
from leakpro.utils.import_helper import Optional


def _fmt(value: Optional[float], digits: int = 4) -> str:
    """Format an optional float for display.

    Args:
    ----
        value: The value, possibly None.
        digits: Decimal places.

    Returns:
    -------
        The formatted value, or an em dash placeholder when absent.

    """
    if value is None:
        return "not reported"
    return f"{value:.{digits}f}"


def _fmt_count(value: Optional[float]) -> str:
    """Format an optional count for display.

    Args:
    ----
        value: The count, possibly None or fractional.

    Returns:
    -------
        The formatted count, or a placeholder when absent.

    """
    if value is None:
        return "not reported"
    return f"{value:,.1f}" if value % 1 else f"{int(value):,}"


def to_markdown(assessment: RiskAssessment) -> str:
    """Render an assessment as markdown.

    Args:
    ----
        assessment: The assessment to render.

    Returns:
    -------
        A markdown document.

    """
    m, d, c = assessment.measured, assessment.declared, assessment.combined
    lines = [
        "# Leakage risk assessment",
        "",
        f"Attack: **{m.attack_name}** · operating point alpha = **{d.tolerated_fpr}** · "
        f"policy {assessment.policy_version} · LeakPro {assessment.leakpro_version}",
        "",
        "## Measured (reproducible from the audit)",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| TPR at alpha | {_fmt(m.success_rate)} |",
        f"| Advantage (TPR - alpha) | {_fmt(m.advantage)} |",
        f"| Lift (TPR / alpha) | {_fmt(m.lift, 1)}x |",
        f"| Members flagged at alpha | {_fmt_count(m.n_exposed_audit)} of {m.n_members_audit} audited members |",
        f"| Audit set | {m.n_members_audit} members, {m.n_non_members_audit} non-members |",
        f"| Finest resolvable FPR | {m.min_resolvable_fpr:.2e} |",
        f"| ROC AUC (context only) | {_fmt(m.roc_auc)} |",
        f"| Train-test gap (context only) | {_fmt(m.train_test_gap)} |",
        f"| DP-SGD epsilon (context only) | {_fmt(m.dp_epsilon)} |",
        "",
        "## Declared (your inputs, not measured)",
        "",
        "| Factor | Value |",
        "|---|---|",
        f"| Attacker prior pi | {d.attacker_prior} (gamma = {_fmt(c.gamma, 2)}) |",
        f"| Data subjects (NDS) | {_fmt_count(d.n_subjects)} ({d.n_subjects_source}) |",
        f"| Records per subject (NR) | {d.records_per_subject} |",
        f"| Data type sensitivity (DTS) | {d.data_type_sensitivity} |",
        f"| Subject type weight (DST) | {d.subject_type_weight} |",
        f"| Cost per exposed subject | {_fmt(d.cost_per_exposed_subject, 2)} |",
        "",
        "## Combined",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| Attacker precision at your pi | {_fmt(c.ppv)} |",
        f"| Attacker precision at pi = 0.5 | {_fmt(c.ppv_balanced)} (the audit protocol's own assumption) |",
        f"| Loss Magnitude (DTS x NR x DST x NDS) | {_fmt_count(c.loss_magnitude)} |",
        f"| Loss Event Frequency | {_fmt(c.loss_event_frequency)} |",
        f"| Risk (LM x LEF) | {_fmt_count(c.risk)} sensitivity-weighted expected exposed records |",
        f"| Expected exposed subjects | {_fmt_count(c.expected_exposed_subjects)} |",
        f"| Expected cost | {_fmt(c.expected_cost, 2)} |",
        f"| Extrapolated exposed records | {_fmt_count(c.extrapolated_exposed_records)} |",
        f"| Vulnerability band (advisory) | {c.vulnerability_band or 'not reported'} |",
        "",
        f"Band provenance: {VULNERABILITY_BANDS_SOURCE}",
        "",
        "## Assumptions",
        "",
    ]
    lines += [f"{i}. {a}" for i, a in enumerate(assessment.assumptions, start=1)]
    if assessment.warnings:
        lines += ["", "## Warnings", ""]
        lines += [f"- {w}" for w in assessment.warnings]
    if d.notes:
        lines += ["", "## Notes", "", d.notes]
    return "\n".join(lines) + "\n"


def _tex_escape(text: str) -> str:
    """Escape LaTeX special characters in free text.

    Args:
    ----
        text: Raw text.

    Returns:
    -------
        Text safe to inline in a LaTeX document.

    """
    for char, replacement in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                              ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                              ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        text = text.replace(char, replacement)
    return text


def to_latex(assessment: RiskAssessment) -> str:
    """Render an assessment as a LaTeX section for the PDF report.

    Args:
    ----
        assessment: The assessment to render.

    Returns:
    -------
        LaTeX markup, suitable for appending to the report document body.

    """
    m, d, c = assessment.measured, assessment.declared, assessment.combined
    rows = [
        ("TPR at alpha", _fmt(m.success_rate)),
        ("Advantage", _fmt(m.advantage)),
        ("Lift", f"{_fmt(m.lift, 1)}x"),
        ("Members flagged", f"{_fmt_count(m.n_exposed_audit)} of {m.n_members_audit}"),
        ("Attacker precision at declared prior", _fmt(c.ppv)),
        ("Attacker precision at prior 0.5", _fmt(c.ppv_balanced)),
        ("Loss Magnitude", _fmt_count(c.loss_magnitude)),
        ("Risk (LM x LEF)", _fmt_count(c.risk)),
        ("Expected exposed subjects", _fmt_count(c.expected_exposed_subjects)),
        ("Expected cost", _fmt(c.expected_cost, 2)),
        ("Vulnerability band (advisory)", c.vulnerability_band or "not reported"),
    ]
    body = "\n".join(f"        {_tex_escape(label)} & {_tex_escape(value)} \\\\" for label, value in rows)
    assumptions = "\n".join(f"        \\item {_tex_escape(a)}" for a in assessment.assumptions)
    latex = f"""
        \\section{{Risk assessment}}
        Attack {_tex_escape(m.attack_name)}, operating point $\\alpha = {d.tolerated_fpr}$,
        declared attacker prior $\\pi = {d.attacker_prior}$, policy {_tex_escape(assessment.policy_version)}.

        \\begin{{tabularx}}{{\\textwidth}}{{lX}}
        \\hline
{body}
        \\hline
        \\end{{tabularx}}

        \\subsection*{{Assumptions}}
        \\begin{{enumerate}}
{assumptions}
        \\end{{enumerate}}
        """
    if assessment.warnings:
        warnings = "\n".join(f"        \\item {_tex_escape(w)}" for w in assessment.warnings)
        latex += f"""
        \\subsection*{{Warnings}}
        \\begin{{itemize}}
{warnings}
        \\end{{itemize}}
        """
    if d.notes:
        latex += f"""
        \\subsection*{{Notes}}
        {_tex_escape(d.notes)}
        """
    return latex
