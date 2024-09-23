"""Contains the AttackResult class, which stores the results of an attack."""

import os
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch import Tensor, clamp, stack
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.utils import save_image

from leakpro.utils.import_helper import Any, List, Self

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
        # TODO REDIFINE THE CLASS SO IT DOSE NOT STORE MATRICIES BUT VECTORS 

        self.predicted_labels = predicted_labels
        self.true_labels = true_labels
        self.predictions_proba = predictions_proba
        self.signal_values = signal_values
        self.threshold = threshold
        self.id = None

        self.accuracy = np.mean(predicted_labels == true_labels, axis=1)
        self.tn = np.sum(true_labels == 0) - np.sum(
            predicted_labels[:, true_labels == 0], axis=1
        )
        self.tp = np.sum(predicted_labels[:, true_labels == 1], axis=1)
        self.fp = np.sum(predicted_labels[:, true_labels == 0], axis=1)
        self.fn = np.sum(true_labels == 1) - np.sum(
            predicted_labels[:, true_labels == 1], axis=1
        )
        self.fpr = self.fp / (np.sum(true_labels == 0))
        self.tpr = self.tp / (np.sum(true_labels == 1))
        # In case the fpr are not sorted in ascending order.
        sorted_indices = np.argsort(self.fpr)
        self.fpr = self.fpr[sorted_indices]
        self.tpr = self.tpr[sorted_indices]

        self.roc_auc = auc(self.fpr, self.tpr)

    def _get_primitives(self:Self):
        """Return the primitives of the CombinedMetricResult class."""
        return {"predicted_labels": self.predicted_labels.tolist(), 
            "true_labels": self.true_labels.tolist(),
            "predictions_proba": self.predictions_proba.tolist() if isinstance(self.predictions_proba, np.ndarray) else None, 
            "signal_values": self.signal_values.tolist() if isinstance(self.signal_values, np.ndarray) else None,
            "threshold": self.threshold.tolist() if isinstance(self.threshold, np.ndarray) else None,
        }
    
    def save(self:Self, path: str, name: str, config:dict):
        """Save the CombinedMetricResult class to disk."""

        # Primitives are the set of variables to re-create the class from scratch
        primitives = self._get_primitives()

        # Data to be saved
        data = {
            "resulttype": self.__class__.__name__,
            "primitives": primitives,
            "config": config
        }

        # Get the name for the attack configuration
        config_name = get_config_name(config["attack_list"][name])

        # Check if path exists, otherwise create it.
        if not os.path.exists(f'{path}/{name}/{name}{config_name}'):
            os.makedirs(f'{path}/{name}/{name}{config_name}')

        # Save the results to a file
        with open(f'{path}/{name}/{name}{config_name}/data.json', 'w') as f:
            json.dump(data, f)

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

class GIAResults:
    """Contains results for a GIA attack."""

    def __init__(self: Self, original_data: DataLoader, recreated_data: DataLoader,
                 psnr_score: float, data_mean: float, data_std: float) -> None:
        self.original_data = original_data
        self.recreated_data = recreated_data
        self.PSNR_score = psnr_score
        self.data_mean = data_mean
        self.data_std = data_std

    def prepare_privacy_risk_report(self: Self, attack_name: str, save_path: str) -> None:
        """Risk report for GIA. WIP."""

        def extract_tensors_from_subset(dataset: Dataset) -> Tensor:
            all_tensors = []
            if isinstance(dataset, Subset):
                for idx in dataset.indices:
                    all_tensors.append(dataset.dataset[idx][0])

            else:
                for idx in range(len(dataset)):
                    all_tensors.append(dataset[idx][0])
            return stack(all_tensors)

        recreated_data = extract_tensors_from_subset(self.recreated_data.dataset)
        original_data = extract_tensors_from_subset(self.original_data.dataset)

        output_denormalized = clamp(recreated_data * self.data_std + self.data_mean, 0, 1)
        save_image(output_denormalized, os.path.join(save_path, "recreated_image.png"))

        gt_denormalized = clamp(original_data * self.data_std + self.data_mean, 0, 1)
        save_image(gt_denormalized, os.path.join(save_path, "original_image.png"))

        return attack_name

def get_config_name(config):
    config = dict(sorted(config.items()))

    exclude = ["attack_data_dir"]

    config_name = ""
    for key, value in zip(list(config.keys()), list(config.values())):
        if key in exclude:
            pass
        else:
            if type(value) is bool:
                config_name += f"-{key}"
            else:
                config_name += f"-{key}={value}"
    return config_name