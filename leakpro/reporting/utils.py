"""Utility functions for generating privacy risk report."""

from leakpro.metrics.attack_result import CombinedMetricResult
from leakpro.reporting.audit_report import (
    ROCCurveReport,
    SignalHistogramReport,
)


def prepare_privacy_risk_report(
    audit_results: CombinedMetricResult,
    configs: dict,
    save_path: str = None,
) -> None:
    """Generate privacy risk report based on the auditing report.

    Args:
    ----
        audit_results(List): Privacy meter results.
        configs (dict): Auditing configuration.
        save_path (str, optional): Report path. Defaults to None.

    Raises:
    ------
        NotImplementedError: Check if the report for the privacy game is implemented.

    """
    if save_path is None:
        raise ValueError("Please provide a save path for the report")

    if audit_results is None:
        raise ValueError("Please provide the audit results")

    # Generate privacy risk report for auditing the model
    ROCCurveReport.generate_report(
        metric_result=audit_results,
        save=True,
        filename=f"{save_path}/ROC.png",
        configs=configs,
    )
    SignalHistogramReport.generate_report(
        metric_result=audit_results,
        save=True,
        filename=f"{save_path}/Histogram.png",
    )