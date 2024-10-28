"""Contains the AttackResult class, which stores the results of an attack."""

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
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
        if not os.path.exists(f"{path}/{name}/{name}{config_name}"):
            os.makedirs(f"{path}/{name}/{name}{config_name}")

        # Save the results to a file
        with open(f"{path}/{name}/{name}{config_name}/data.json", "w") as f:
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

class MIAResult:
    """Contains results related to the performance of the metric. It contains the results for multiple fpr."""

    def __init__(  # noqa: PLR0913
        self:Self,
        predicted_labels: list=None,
        true_labels: list=None,
        predictions_proba:list=None,
        signal_values:list=None,
        threshold: list = None,
        audit_indices: list = None,
        metadata: dict = None,
        resultname: str = None,
        id: str = None,
        load: bool = False,
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
        self.audit_indices = audit_indices
        self.metadata = metadata
        self.resultname = resultname
        self.id = id

        if load:
            return

        self.tn = np.sum(true_labels == 0) - np.sum(
            predicted_labels[:, true_labels == 0], axis=1
        )
        self.tp = np.sum(predicted_labels[:, true_labels == 1], axis=1)
        self.fp = np.sum(predicted_labels[:, true_labels == 0], axis=1)
        self.fn = np.sum(true_labels == 1) - np.sum(
            predicted_labels[:, true_labels == 1], axis=1
        )

        self.fpr = self.fp / (self.fp + self.tn)
        self.tpr = self.tp / (self.tp + self.fn)
        self.roc_auc = auc(self.fpr, self.tpr)

    def load(self, data):
        self.resultname = data["resultname"]
        self.resulttype = data["resulttype"]
        self.tpr = data["tpr"]
        self.fpr = data["fpr"]
        self.roc_auc = data["roc_auc"]
        self.config = data["config"]
        self.fixed_fpr_table = data["fixed_fpr"]
        self.audit_indices = data["audit_indices"]
        self.signal_values = data["signal_values"]
        self.true_labels = data["true_labels"]
        self.threshold = data["threshold"]

    def save(self:Self, path: str, name: str, config:dict = None):
        """Save the MIAResults to disk."""

        print(config)

        result_config = config["attack_list"][name]
        fixed_fpr_table = get_result_fixed_fpr(self.fpr, self.tpr)

        # Get the name for the attack configuration
        config_name = get_config_name(result_config)

        self.id = f"{name}{config_name}"
        save_path = f"{path}/{name}/{self.id}"

        # Data to be saved
        data = {
            "resulttype": self.__class__.__name__,
            "resultname": name,
            "tpr": self.tpr.tolist(),
            "fpr": self.fpr.tolist(),
            "roc_auc": self.roc_auc,
            "config": config,
            "fixed_fpr": fixed_fpr_table,
            "audit_indices": self.audit_indices.tolist(),
            "signal_values": self.signal_values.tolist(),
            "true_labels": self.true_labels.tolist(),
            "threshold": self.threshold.tolist() if self.threshold is not None else None,
            "id": name,
        }

        # Check if path exists, otherwise create it.
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Save the results to a file
        with open(f"{save_path}/data.json", "w") as f:
            json.dump(data, f)

        # Create ROC plot for MIAResult
        filename = f"{save_path}/ROC"
        temp_res = MIAResult(load=True)
        temp_res.tpr = self.tpr
        temp_res.fpr = self.fpr
        temp_res.id = self.id
        self.create_plot(results = [temp_res],
                        filename = filename
                        )

        # Create SignalHistogram plot for MIAResult
        filename = f"{save_path}/SignalHistogram.png"
        self.create_signal_histogram(filename = filename,
                                    signal_values = self.signal_values,
                                    true_labels = self.true_labels,
                                    threshold = self.threshold
                                    )

    @classmethod
    def get_strongest(self, results) -> list:
        """Method for selecting the strongest attack."""
        return max((res for res in results), key=lambda d: d.roc_auc)

    def create_signal_histogram(self, filename, signal_values, true_labels, threshold) -> None:

        values = np.array(signal_values).ravel()
        labels = np.array(true_labels).ravel()
        threshold = threshold

        data = pd.DataFrame(
                {
                    "Signal": values,
                    "Membership": ["Member" if y == 1 else "Non-member" for y in labels],
                }
            )

        bin_edges = np.histogram_bin_edges(values, bins=1000)

        histogram = sn.histplot(
            data=data,
            x="Signal",
            hue="Membership",
            element="step",
            kde=True,
            bins = bin_edges
        )

        if threshold is not None and isinstance(threshold, float):
            histogram.axvline(x=threshold, linestyle="--", color="C{}".format(2))
            histogram.text(
                x=threshold - (np.max(values) - np.min(values)) / 30,
                y=0.8,
                s="Threshold",
                rotation=90,
                color="C{}".format(2),
                transform=histogram.get_xaxis_transform(),
            )

        plt.grid()
        plt.xlabel("Signal value")
        plt.ylabel("Number of samples")
        plt.title("Signal histogram")
        plt.savefig(fname=filename, dpi=1000)
        plt.clf()

    def create_plot(self, results, filename = "", save_name = "") -> None:

        # Create plot for results
        reduced_labels = reduce_to_unique_labels(results)
        for res, label in zip(results, reduced_labels):

            plt.fill_between(res.fpr, res.tpr, alpha=0.15)
            plt.plot(res.fpr, res.tpr, label=label)

        # Plot random guesses
        range01 = np.linspace(0, 1)
        plt.plot(range01, range01, "--", label="Random guess")

        # Set plot parameters
        plt.yscale("log")
        plt.xscale("log")
        plt.xlim(left=1e-5)
        plt.ylim(bottom=1e-5)
        plt.tight_layout()
        plt.grid()
        plt.legend(bbox_to_anchor =(0.5,-0.27), loc="lower center")

        plt.xlabel("False positive rate (FPR)")
        plt.ylabel("True positive rate (TPR)")
        plt.title(save_name+"ROC Curve")
        plt.savefig(fname=f"{filename}.png", dpi=1000, bbox_inches="tight")
        plt.clf()

    @classmethod
    def create_results(
            self: Self,
            results: list,
            save_dir: str = "./",
            save_name: str = "foo",
        ):

        filename = f"{save_dir}/{save_name}"

        self.create_plot(results, filename, save_name)

        return self._latex(results, save_name, filename)

    def _latex(self, results, subsection, filename):
        """Latex method for MIAResult."""

        latex_content = ""
        latex_content += f"""
        \\subsection{{{" ".join(subsection.split("_"))}}}
        \\begin{{figure}}[ht]
        \\includegraphics[width=0.8\\textwidth]{{{filename}.png}}
        \\end{{figure}}
        """

        latex_content += """
        \\resizebox{\\linewidth}{!}{%
        \\begin{tabularx}{\\textwidth}{l c l l l l}
        Attack name & attack config & TPR: 1.0\\%FPR & 0.1\\%FPR & 0.01\\%FPR & 0.0\\%FPR \\\\
        \\hline 
        """

        def config_latex_style(config):
            config = " \\\\ ".join(config.split("-")[1:])
            config = "-".join(config.split("_"))
            return f"""\\shortstack{{{config}}}"""

        for res in results:
            config = config_latex_style(res.id)
            latex_content += f"""{"-".join(res.resultname.split("_"))} & {config} & {res.fixed_fpr_table["TPR@1.0%FPR"]} & {res.fixed_fpr_table["TPR@0.1%FPR"]} & {res.fixed_fpr_table["TPR@0.01%FPR"]} & {res.fixed_fpr_table["TPR@0.0%FPR"]} \\\\ \\hline 
            """
        latex_content += """
        \\end{tabularx}
        }
        \\newline
        """
        return latex_content



class GIAResults:
    """Contains results for a GIA attack."""

    def __init__(self: Self, original_data: DataLoader, recreated_data: DataLoader,
                 psnr_score: float, data_mean: float, data_std: float, load: bool) -> None:
        self.original_data = original_data
        self.recreated_data = recreated_data
        self.PSNR_score = psnr_score
        self.data_mean = data_mean
        self.data_std = data_std

        if load:
            return

    def load(self, data):
        self.original = data["original"]
        self.resulttype = data["resulttype"]
        self.recreated = data["recreated"]
        self.id = data["id"]

    def save(self: Self, save_path: str, name: str, config: dict):
        """Save the GIAResults to disk."""

        result_config = config["attack_list"][name]

        # Get the name for the attack configuration
        config_name = get_config_name(result_config)
        self.id = f"{name}{config_name}"
        save_path = f"{save_path}/{name}/{self.id}"

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
        recreated = os.path.join(save_path, "recreated_image.png")
        save_image(output_denormalized, recreated)

        gt_denormalized = clamp(original_data * self.data_std + self.data_mean, 0, 1)
        original = os.path.join(save_path, "original_image.png")
        save_image(gt_denormalized, original)

        # Data to be saved
        data = {
            "resulttype": self.__class__.__name__,
            "original": original,
            "recreated": recreated,
            "id": self.id,
        }

        # Check if path exists, otherwise create it.
        if not os.path.exists(f"{save_path}"):
            os.makedirs(f"{save_path}")

        # Save the results to a file
        with open(f"{save_path}/data.json", "w") as f:
            json.dump(data, f)

        pass

    @classmethod
    def create_result(self: Self, attack_name: str, save_path: str) -> None:
        """Result method for GIA."""

        def _latex(attack_name, original, recreated):
            latex_content = f"""
            \\subsection{{{" ".join(attack_name.split("_"))}}}
            \\begin{{figure}}[ht]
            \\includegraphics[width=0.8\\textwidth]{{{original}}}
            \\caption{{Original}}
            \\end{{figure}}

            \\begin{{figure}}[ht]
            \\includegraphics[width=0.8\\textwidth]{{{recreated}}}
            \\caption{{Original}}
            \\end{{figure}}
            """
            return latex_content

        return _latex(attack_name=attack_name, original=save_path+"recreated_image.png", recreated=save_path+"original_image.png")

class SyntheticResult:
    """Contains results related to the performance of the metric. It contains the results for multiple fpr."""

    def __init__(  # noqa: PLR0913
        self:Self,
        load: bool = False,
    )-> None:
        """Initalze Result method

        Args:
        ----

        """
        # Initialize values to result object
        # self.values = values

        # Have a method to return if the results are to be loaded
        if load:
            return

        # Create some result
        # self.result_values = some_result

    def load(self, data: dict):
        """Load the TEMPLATEResult class to disk."""
        # self.result_values = data["some_result"]
        pass

    def save(self:Self, path: str, name: str, config:dict = None):
        """Save the TEMPLATEResult class to disk."""

        result_config = config["attack_list"][name]

        # Data to be saved
        data = {
            "some_result": self.result_values
        }

        # Get the name for the attack configuration
        config_name = get_config_name(result_config)
        self.id = f"{name}{config_name}"
        save_path = f"{path}/{name}/{self.id}"

        # Check if path exists, otherwise create it.
        if not os.path.exists(f"{save_path}"):
            os.makedirs(f"{save_path}")

        # Save the results to a file
        with open(f"{save_path}/data.json", "w") as f:
            json.dump(data, f)

class TEMPLATEResult:
    """Contains results related to the performance of the metric. It contains the results for multiple fpr."""

    def __init__(  # noqa: PLR0913
        self:Self,
        load: bool = False,
    )-> None:
        """Initalze Result method

        Args:
        ----

        """
        # Initialize values to result object
        # self.values = values

        # Have a method to return if the results are to be loaded
        if load:
            return

        # Create some result
        # self.result_values = some_result

    def load(self, data: dict):
        """Load the TEMPLATEResult class to disk."""
        # self.result_values = data["some_result"]
        pass

    def save(self:Self, path: str, name: str, config:dict = None):
        """Save the TEMPLATEResult class to disk."""

        result_config = config["attack_list"][name]

        # Data to be saved
        data = {
            "some_result": self.result_values
        }

        # Get the name for the attack configuration
        config_name = get_config_name(result_config)
        self.id = f"{name}{config_name}"

        # Check if path exists, otherwise create it.
        if not os.path.exists(f"{path}/{name}/{self.id}"):
            os.makedirs(f"{path}/{name}/{self.id}")

        # Save the results to a file
        with open(f"{path}/{name}/{self.id}/data.json", "w") as f:
            json.dump(data, f)

    @classmethod
    def create_result(self, results):
        """Method for results."""
        def _latex(results):
            """Latex method for TEMPLATEResult"""
            pass
        pass

def get_result_fixed_fpr(fpr, tpr):

    # Function to find TPR at given FPR thresholds
    def find_tpr_at_fpr(fpr_array:np.ndarray, tpr_array:np.ndarray, threshold:float): #-> Optional[str]:
        try:
            # Find the last index where FPR is less than the threshold
            valid_index = np.where(fpr_array < threshold)[0][-1]
            return float(f"{tpr_array[valid_index] * 100:.4f}")
        except IndexError:
            # Return None or some default value if no valid index found
            return "N/A"

    # Compute TPR values at various FPR thresholds
    return {"TPR@1.0%FPR": find_tpr_at_fpr(fpr, tpr, 0.01),
            "TPR@0.1%FPR": find_tpr_at_fpr(fpr, tpr, 0.001),
            "TPR@0.01%FPR": find_tpr_at_fpr(fpr, tpr, 0.0001),
            "TPR@0.0%FPR": find_tpr_at_fpr(fpr, tpr, 0.0)}

def get_config_name(config):
    config = dict(sorted(config.items()))

    exclude = ["attack_data_dir"]

    config_name = ""
    for key, value in zip(list(config.keys()), list(config.values())):
        if key in exclude:
            pass
        elif type(value) is bool:
            config_name += f"-{key}"
        else:
            config_name += f"-{key}={value}"
    return config_name

def reduce_to_unique_labels(results):
    """Reduce very long labels to unique and distinct ones."""
    strings = [res.id for res in results]

    # Dictionary to store name as key and a list of configurations as value
    name_configs = defaultdict(list)

    # Parse each string and store configurations
    for s in strings:
        parts = s.split("-")
        name = parts[0]  # The first part is the name
        config = "-".join(parts[1:]) if len(parts) > 1 else ""  # The rest is the configuration
        name_configs[name].append(config)  # Store the configuration under the name

    def find_common_suffix(configs):
        """Helper function to find the common suffix among multiple configurations"""
        if not configs:
            return ""

        # Split each configuration by "-" and zip them in reverse to compare backwards
        reversed_configs = [config.split("-")[::-1] for config in configs]
        common_suffix = []

        for elements in zip(*reversed_configs):
            if all(e == elements[0] for e in elements):
                common_suffix.append(elements[0])
            else:
                break

        # Return the common suffix as a string, reversed back to normal order
        return "-".join(common_suffix[::-1])

    result = []

    # Process each name and its configurations
    for name, configs in name_configs.items():
        if len(configs) > 1:
            # Find the common suffix for the configurations
            common_suffix = find_common_suffix(configs)

            # Remove the common suffix from each configuration
            trimmed_configs = [config[:-(len(common_suffix) + 1)] if common_suffix and config.endswith(common_suffix) else config for config in configs]

            # Process configurations based on whether they share the same pattern
            for config in trimmed_configs:
                if config:
                    result.append(f"{name}-{config}")
                else:
                    result.append(name)
        else:
            # If only one configuration, just return the string as is
            result.append(f"{name}")

    return result


