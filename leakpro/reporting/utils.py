"""Utility functions for generating privacy risk report."""

from leakpro.import_helper import List
from leakpro.reporting import audit_report
from leakpro.reporting.audit_report import (
    ROCCurveReport,
    SignalHistogramReport,
)


def prepare_priavcy_risk_report(
    log_dir: str,
    audit_results: List,
    configs: dict,
    save_path: str = None,
) -> None:
    """Generate privacy risk report based on the auditing report.

    Args:
    ----
        log_dir(str): Log directory that saved all the information, including the models.
        audit_results(List): Privacy meter results.
        configs (dict): Auditing configuration.
        save_path (str, optional): Report path. Defaults to None.

    Raises:
    ------
        NotImplementedError: Check if the report for the privacy game is implemented.

    """
    audit_report.REPORT_FILES_DIR = "privacy_meter/report_files"
    if save_path is None:
        save_path = log_dir

    if configs["privacy_game"] in [
        "privacy_loss_model",
        "avg_privacy_loss_training_algo",
    ]:
        # Generate privacy risk report for auditing the model
        if len(audit_results) == 1 and configs["privacy_game"] == "privacy_loss_model":
            ROCCurveReport.generate_report(
                metric_result=audit_results[0],
                save=True,
                filename=f"{save_path}/ROC.png",
                configs=configs,
            )
            SignalHistogramReport.generate_report(
                metric_result=audit_results[0],
                save=True,
                filename=f"{save_path}/Histogram.png",
            )
        else:
            raise ValueError(
                f"{len(audit_results)} results are not enough for {configs['privacy_game']})"
            )
    else:
        raise NotImplementedError(f"{configs['privacy_game']} is not implemented yet")
