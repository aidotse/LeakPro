"""Contains the AttackResult class, which stores the results of an attack."""

# typing package not available form < python-3.11, typing_extensions backports new and experimental type hinting features to older Python versions
try:
    from typing import  Any, List, Self
except ImportError:
    from typing_extensions import  Any, List, Self

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

########################################################################################################################
# METRIC_RESULT CLASS
########################################################################################################################


class AttackResult:
    """Contains results related to the performance of the attack."""

    def __init__(  # noqa: PLR0913
        self:Self,
        predicted_labels: list,
        true_labels: list,
        predictions_proba: List[List[float]] = None,
        signal_values:List[Any]=None,
        threshold: float = None,
    ) -> None:
        """Compute and stores the accuracy, ROC AUC score, and the confusion matrix for a metric.

        Args:
        ----
            metric_id: ID of the metric that was used (c.f. the report_files/explanations.json file).
            predicted_labels: Membership predictions of the metric.
            true_labels: True membership labels used to evaluate the metric.
            predictions_proba: Continuous version of the predicted_labels.
            signal_values: Values of the signal used by the metric.
            threshold: Threshold computed by the metric.

        """
        self.predicted_labels = predicted_labels
        self.true_labels = true_labels
        self.predictions_proba = predictions_proba
        self.signal_values = signal_values
        self.threshold = threshold

        self.accuracy = accuracy_score(y_true=true_labels, y_pred=predicted_labels)

        if self.predictions_proba is None:
            self.roc = roc_curve(y_true=true_labels, y_score=predicted_labels)
        else:
            self.roc = roc_curve(y_true=true_labels, y_score=predictions_proba)

        if self.predictions_proba is None:
            self.roc_auc = roc_auc_score(y_true=true_labels, y_score=predicted_labels)
        else:
            self.roc_auc = roc_auc_score(y_true=true_labels, y_score=predictions_proba)

        self.tn, self.fp, self.fn, self.tp = confusion_matrix(
            y_true=true_labels, y_pred=predicted_labels
        ).ravel()

    def __str__(self:Self) -> str:
        """Return a string describing the metric result."""
        txt = [
            f'{" METRIC RESULT OBJECT ":=^48}',
            f"Accuracy          = {self.accuracy}",
            f"ROC AUC Score     = {self.roc_auc}",
            f"FPR               = {self.fp / (self.fp + self.tn)}",
            f"TN, FP, FN, TP    = {self.tn, self.fp, self.fn, self.tp}",
        ]
        return "\n".join(txt)


class CombinedMetricResult:
    """Contains results related to the performance of the metric. It contains the results for multiple fpr."""

    def __init__(  # noqa: PLR0913
        self:Self,
        predicted_labels: list,
        true_labels: list,
        predictions_proba:list=None,
        signal_values:list=None,
        threshold: list = None,
    )-> None:
        """Compute and store the accuracy, ROC AUC score, and the confusion matrix for a metric.

        Args:
        ----
            predicted_labels: Membership predictions of the metric.
            true_labels: True membership labels used to evaluate the metric.
            predictions_proba: Continuous version of the predicted_labels.
            signal_values: Values of the signal used by the metric.
            threshold: Threshold computed by the metric.

        """
        self.predicted_labels = predicted_labels
        self.true_labels = true_labels
        self.predictions_proba = predictions_proba
        self.signal_values = signal_values
        self.threshold = threshold

        self.accuracy = np.mean(predicted_labels == true_labels, axis=1)
        self.tn = np.sum(true_labels == 0) - np.sum(
            predicted_labels[:, true_labels == 0], axis=1
        )
        self.tp = np.sum(predicted_labels[:, true_labels == 1], axis=1)
        self.fp = np.sum(predicted_labels[:, true_labels == 0], axis=1)
        self.fn = np.sum(true_labels == 1) - np.sum(
            predicted_labels[:, true_labels == 1], axis=1
        )

        self.roc_auc = auc(
            self.fp / (np.sum(true_labels == 0)), self.tp / (np.sum(true_labels == 1))
        )

    def __str__(self:Self) -> str:
        """Return a string describing the metric result."""
        txt_list = []
        for idx in range(len(self.accuracy)):
            txt = [
                f'{" METRIC RESULT OBJECT ":=^48}',
                f"Accuracy          = {self.accuracy[idx]}",
                f"ROC AUC Score     = {self.roc_auc}",
                f"FPR               = {self.fp[idx] / (self.fp[idx] + self.tn[idx])}",
                f"TN, FP, FN, TP    = {self.tn[idx], self.fp[idx], self.fn[idx], self.tp[idx]}",
            ]

            txt_list.append("\n".join(txt))
        return "\n\n".join(txt_list)
