#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""LaTeX helpers shared by the report sections."""

_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def latex_escape(text: object) -> str:
    """Escape LaTeX special characters in ``str(text)``, so it can be inlined in a LaTeX document."""
    return "".join(_LATEX_SPECIAL.get(ch, ch) for ch in str(text))
