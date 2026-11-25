#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Base class for results that render their own section of the PDF report.

:class:`~leakpro.reporting.report_handler.ReportHandler` knows a fixed set of result types
(``MIAResult``, ``GIAResults``, the synthetic-data results). Any other result that subclasses
:class:`AuditResult` is grouped by class and rendered with its own :meth:`AuditResult.create_results`,
under the heading ``report_title``. New attack families can therefore add a report section without
changing the report handler.
"""

from abc import ABC, abstractmethod
from typing import List

from leakpro.utils.latex import latex_escape  # noqa: F401  (re-exported for the report sections)


class AuditResult(ABC):
    """A result that renders its own report section."""

    #: Section heading in the report.
    report_title: str = "Results"

    @classmethod
    @abstractmethod
    def create_results(cls, results: List["AuditResult"], save_dir: str = "./") -> str:  # noqa: ANN102
        """Return the LaTeX for ``results`` (all of this class); figures may be written to ``save_dir``."""
