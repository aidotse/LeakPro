"""Contains the AttackResult class, which stores the results of an attack."""

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from pydantic import BaseModel
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

    def _get_primitives(self:Self) -> dict:
        """Return the primitives of the CombinedMetricResult class."""
        return {"predicted_labels": self.predicted_labels.tolist(),
            "true_labels": self.true_labels.tolist(),
            "predictions_proba": self.predictions_proba.tolist() if isinstance(self.predictions_proba, np.ndarray) else None,
            "signal_values": self.signal_values.tolist() if isinstance(self.signal_values, np.ndarray) else None,
            "threshold": self.threshold.tolist() if isinstance(self.threshold, np.ndarray) else None,
        }

    def save(self:Self, path: str, name: str, config:dict) -> None:
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
    )-> None:
        """Compute and store the accuracy, ROC AUC score, and the confusion matrix for a metric.

        Args:
        ----
            predicted_labels: Membership predictions of the metric.
            true_labels: True membership labels used to evaluate the metric.
            predictions_proba: Continuous version of the predicted_labels.
            signal_values: Values of the signal used by the metric.
            threshold: Threshold computed by the metric.
            audit_indices: The corresponding dataset indices for the results
            id: The identity of the attack
            load: If the data should be loaded
            metadata: Metadata about the results
            resultname: The name of the attack and result

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

        if true_labels is None or predicted_labels is None:
            self.tn, self.tp, self.fn, self.fp = 0.0, 0.0, 0.0, 0.0
            self.fpr, self.tpr, self.roc_auc = 0.0, 0.0, 0.0
            return

        self.tn = np.sum(true_labels == 0) - np.sum(
            predicted_labels[:, true_labels == 0], axis=1
        )
        self.tp = np.sum(predicted_labels[:, true_labels == 1], axis=1)
        self.fp = np.sum(predicted_labels[:, true_labels == 0], axis=1)
        self.fn = np.sum(true_labels == 1) - np.sum(
            predicted_labels[:, true_labels == 1], axis=1
        )

        self.fpr = np.divide(self.fp.astype(float), (self.fp + self.tn).astype(float),
                            out=np.full_like(self.fp, np.nan, dtype=float),
                            where=(self.fp + self.tn) != 0.0)
        self.tpr = np.divide(self.tp.astype(float), (self.tp + self.fn).astype(float),
                            out=np.full_like(self.tp, np.nan, dtype=float),
                            where=(self.tp + self.fn) != 0.0)

        # In case denominator is zero in fpr/tpr calculations
        not_nan = ~(np.isnan(self.fpr) | np.isnan(self.tpr))
        self.fpr = self.fpr[not_nan]
        self.tpr = self.tpr[not_nan]

        # In case the fpr are not sorted in ascending order.
        sorted_indices = np.argsort(self.fpr)
        self.fpr = self.fpr[sorted_indices]
        self.tpr = self.tpr[sorted_indices]

        self.roc_auc = auc(self.fpr, self.tpr)


    @staticmethod
    def load(data: dict) -> None:
        """Load the MIAResults to disk."""

        miaresult = MIAResult()

        miaresult.resultname = data["resultname"]
        miaresult.resulttype = data["resulttype"]
        miaresult.tpr = data["tpr"]
        miaresult.fpr = data["fpr"]
        miaresult.roc_auc = data["roc_auc"]
        miaresult.config = data["config"]
        miaresult.fixed_fpr_table = data["fixed_fpr"]
        miaresult.audit_indices = data["audit_indices"]
        miaresult.signal_values = data["signal_values"]
        miaresult.true_labels = data["true_labels"]
        miaresult.threshold = data["threshold"]

        miaresult.id = data["id"]

        return miaresult

    def save(self:Self, path: str, name: str, config:dict = None, show_plot:bool = False) -> None:
        """Save the MIAResults to disk."""

        result_config = config["attack_list"][name]
        fixed_fpr_table = get_result_fixed_fpr(self.fpr, self.tpr)

        # Get the name for the attack configuration
        config_name = get_config_name(result_config)

        self.id = f"{name}{config_name}".replace("/", "__")
        save_path = f"{path}/{name}/{self.id}"

        # Data to be saved
        data = {
            "resulttype": self.__class__.__name__,
            "resultname": name,
            "tpr": self.tpr.tolist(),
            "fpr": self.fpr.tolist(),
            "roc_auc": self.roc_auc,
            "config": result_config,
            "fixed_fpr": fixed_fpr_table,
            "audit_indices": self.audit_indices.tolist() if self.audit_indices is not None else None,
            "signal_values": self.signal_values.tolist() if self.signal_values is not None else None,
            "true_labels": self.true_labels.tolist() if self.true_labels is not None else None,
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
        temp_res = MIAResult()
        temp_res.tpr = self.tpr
        temp_res.fpr = self.fpr
        temp_res.id = self.id
        self.create_plot(results = [temp_res],
                        save_dir = save_path,
                        save_name = name,
                        show_plot = show_plot
                        )

        # Create SignalHistogram plot for MIAResult
        self.create_signal_histogram(save_path = save_path,
                                    save_name = "SignalHistogram",
                                    signal_values = self.signal_values,
                                    true_labels = self.true_labels,
                                    threshold = self.threshold,
                                    show_plot = show_plot,
                                    )

    @staticmethod
    def get_strongest(results: list) -> list:
        """Method for selecting the strongest attack."""
        return max((res for res in results), key=lambda d: d.roc_auc)

    def create_signal_histogram(
            self:Self,
            save_path: str,
            save_name: str,
            signal_values: list,
            true_labels: list,
            threshold: float,
            show_plot: bool = False,
            ) -> None:
        """Method to create Signal Histogram."""

        filename = f"{save_path}/{save_name}"
        values = np.array(signal_values).ravel()
        labels = np.array(true_labels).ravel()

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
        plt.savefig(fname=filename+".png", dpi=1000)
        if show_plot:
            plt.show()
        else:
            plt.clf()

    @staticmethod
    def create_plot(
            results: list,
            save_dir: str = "",
            save_name: str = "",
            show_plot: bool = False
        ) -> None:
        """Plot method for MIAResult."""

        filename = f"{save_dir}/{save_name}"

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
        plt.title(save_name+" ROC Curve")
        plt.savefig(fname=f"{filename}.png", dpi=1000, bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.clf()

    @staticmethod
    def _get_all_attacknames(
            results: list
            ) -> list:
        attack_name_list = []
        for result in results:
            if result.resultname not in attack_name_list:
                attack_name_list.append(result.resultname)
        return attack_name_list

    @staticmethod
    def _get_results_of_name(
            results: list,
            resultname_value: str
            ) -> list:
        indices = [idx for (idx, result) in enumerate(results) if result.resultname == resultname_value]
        return [results[idx] for idx in indices]

    @staticmethod
    def create_results(
            results: list,
            save_dir: str = "./",
            save_name: str = "foo", # noqa: ARG004
            show_plot: bool = False,
        ) -> str:
        """Result method for MIAResult."""
        latex = ""

        # Create plot for all results
        MIAResult.create_plot(results, save_dir, save_name="all_results", show_plot=show_plot)
        latex += MIAResult._latex(results, save_dir, save_name="all_results")

        # Create plot for results grouped by name
        all_attack_names = MIAResult._get_all_attacknames(results)
        for name in all_attack_names:
            results_name_grouped = MIAResult._get_results_of_name(results, name)
            MIAResult.create_plot(results_name_grouped, save_dir, save_name=name, show_plot=show_plot)
            latex += MIAResult._latex(results_name_grouped, save_dir, save_name=name)

        # Create plot for results grouped by name
        grouped_results = [MIAResult._get_results_of_name(results, name) for name
                           in all_attack_names]
        strongest_results = [MIAResult.get_strongest(result) for result in grouped_results]
        MIAResult.create_plot(strongest_results, save_dir, save_name="strongest", show_plot=show_plot)
        latex += MIAResult._latex(strongest_results, save_dir, save_name="strongest")

        return latex

    @staticmethod
    def _latex(
            results: list,
            save_dir: str, # noqa: ARG004
            save_name: str
        ) -> str:
        """Latex method for MIAResult."""

        # Input mia results image
        latex_content = f"""
        \\subsection{{{" ".join(save_name.split("_"))}}}
        \\begin{{figure}}[ht]
        \\includegraphics[width=0.8\\textwidth]{{{save_name}.png}}
        \\end{{figure}}
        """

        # Initialize latex table
        latex_content += """
        \\resizebox{\\linewidth}{!}{%
        \\begin{tabularx}{\\textwidth}{l c l l l l}
        Attack name & attack config & TPR: 1.0\\%FPR & 0.1\\%FPR & 0.01\\%FPR & 0.0\\%FPR \\\\ \\hline """ # noqa: W291

        # Convert config to latex table input
        def config_latex_style(config: str) -> str:
            config = " \\\\ ".join(config.split("-")[1:])
            config = "-".join(config.split("_"))
            return f"""\\shortstack{{{config}}}"""

        # Append all mia results to table
        for res in results:
            config = config_latex_style(get_config_name(res.config))
            latex_content += f"""
            {"-".join(res.resultname.split("_"))} & {config} & {res.fixed_fpr_table["TPR@1.0%FPR"]} & {res.fixed_fpr_table["TPR@0.1%FPR"]} & {res.fixed_fpr_table["TPR@0.01%FPR"]} & {res.fixed_fpr_table["TPR@0.0%FPR"]} \\\\ \\hline """ # noqa: E501
        latex_content += """
        \\end{tabularx}
        }
        \\newline
        """
        return latex_content

class GIAResults:
    """Contains results for a GIA attack."""

    def __init__(
            self: Self,
            original_data: DataLoader = None,
            recreated_data: DataLoader = None,
            psnr_score: float = None,
            ssim_score: float = None,
            data_mean: float = None,
            data_std: float = None,
            config: dict = None,
        ) -> None:

        self.original_data = original_data
        self.recreated_data = recreated_data
        self.PSNR_score = psnr_score
        self.SSIM_score = ssim_score
        self.data_mean = data_mean
        self.data_std = data_std
        self.config = config

    @staticmethod
    def load(
            data:dict
        ) -> None:
        """Load the GIAResults from disk."""

        giaresult = GIAResults()

        giaresult.original = data["original"]
        giaresult.resulttype = data["resulttype"]
        giaresult.recreated = data["recreated"]
        giaresult.id = data["id"]
        giaresult.result_config = data["result_config"]

        return giaresult

    def save(
            self: Self,
            name: str,
            path: str,
            config: dict,
            show_plot: bool = False # noqa: ARG002
        ) -> None:
        """Save the GIAResults to disk."""

        def get_gia_config(instance: Any, skip_keys: List[str] = None) -> dict:
            """Extract manually typed variables and their values from a class instance with options to skip keys."""
            if skip_keys is None:
                skip_keys = []

            cls_annotations = instance.__class__.__annotations__  # Get typed attributes
            return {
                var: getattr(instance, var)
                for var in cls_annotations
                if var not in skip_keys  # Exclude skipped keys
            }

        result_config = get_gia_config(config, skip_keys=["optimizer", "criterion"])

        # Get the name for the attack configuration
        config_name = get_config_name(result_config)
        self.id = f"{name}{config_name}"
        path = f"{path}/gradient_inversion/{self.id}"

        # Check if path exists, otherwise create it.
        if not os.path.exists(f"{path}"):
            os.makedirs(f"{path}")

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
        recreated = os.path.join(path, "recreated_image.png")
        save_image(output_denormalized, recreated)

        gt_denormalized = clamp(original_data * self.data_std + self.data_mean, 0, 1)
        original = os.path.join(path, "original_image.png")
        save_image(gt_denormalized, original)

        # Data to be saved
        data = {
            "resulttype": self.__class__.__name__,
            "original": original,
            "recreated": recreated,
            "result_config": result_config,
            "id": self.id,
        }

        # Save the results to a file
        with open(f"{path}/data.json", "w") as f:
            json.dump(data, f)

    @staticmethod
    def create_results(
            results: list,
            save_dir: str = "./", # noqa: ARG004
            save_name: str = "foo", # noqa: ARG004
        ) -> str:
        """Result method for GIA."""
        latex = ""
        def _latex(
                save_name: str,
                original: str,
                recreated: str
            ) -> str:
            """Latex method for GIAResults."""
            return f"""
            \\subsection{{{" ".join(save_name.split("_"))}}}
            \\begin{{figure}}[ht]
            \\includegraphics[width=0.6\\textwidth]{{{original}}}
            \\caption{{Original}}
            \\end{{figure}}

            \\begin{{figure}}[ht]
            \\includegraphics[width=0.6\\textwidth]{{{recreated}}}
            \\caption{{Recreated}}
            \\end{{figure}}
            """
        unique_names = reduce_to_unique_labels(results)
        for res, name in zip(results, unique_names):
            latex += _latex(save_name=name, original=res.original, recreated=res.recreated)
        return latex

class MinvResult:
    """Contains results for a MI attack."""

    pass



class TEMPLATEResult:
    """Contains results related to the performance of the metric. It contains the results for multiple fpr."""

    def __init__(  # noqa: PLR0913
            self:Self,
            values: list =  None,
        ) -> None:
        """Initialize the result method."""

        # Initialize values to result object
        self.values = values

        # Create some latex result
        self.result_values = self.create_result(self.values)

    @staticmethod
    def load(
            data: dict
        ) -> None:
        """Load the TEMPLATEResult class to disk."""
        template_res = TEMPLATEResult()
        template_res.values = data["some_values"]
        return template_res

    def save(
            self:Self,
            path: str,
            name: str,
            config:dict = None
        ) -> None:
        """Save the TEMPLATEResult class to disk."""

        # Data to be saved
        data = {
            "some_values": self.values
        }

        # Get the name for the attack configuration
        config_name = get_config_name(config)
        self.id = f"{name}{config_name}"

        # Check if path exists, otherwise create it.
        if not os.path.exists(f"{path}/{name}/{self.id}"):
            os.makedirs(f"{path}/{name}/{self.id}")

        # Save the results to a file
        with open(f"{path}/{name}/{self.id}/data.json", "w") as f:
            json.dump(data, f)

    @staticmethod
    def create_results(results: list) -> str:
        """Method for results."""
        def _latex(results: list) -> str:
            """Latex method for TEMPLATEResult."""
            return results
        return _latex(results)

def find_tpr_at_fpr(fpr_array: np.ndarray, tpr_array:np.ndarray, threshold:float) -> float:
    """Find TPR for a given FPR."""
    # Find the last index where FPR is less than or equal to the threshold
    less_equal = np.nonzero(fpr_array <= threshold)[0]
    if len(less_equal) == 0: # If no fpr at given threshold exists return 0.0%
        return 0.0
    max_tpr = np.max(tpr_array[less_equal])
    return float(f"{max_tpr * 100:.4f}")

def get_result_fixed_fpr(fpr: np.ndarray, tpr: np.ndarray) -> dict:
    """Find TPR values for fixed FPRs."""

    # Compute TPR values at various FPR thresholds
    return {"TPR@1.0%FPR": find_tpr_at_fpr(fpr, tpr, 0.01),
            "TPR@0.1%FPR": find_tpr_at_fpr(fpr, tpr, 0.001),
            "TPR@0.01%FPR": find_tpr_at_fpr(fpr, tpr, 0.0001),
            "TPR@0.0%FPR": find_tpr_at_fpr(fpr, tpr, 0.0)}

def get_config_name(config: BaseModel) -> str:
    """Create id from the attack config."""

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

def reduce_to_unique_labels(results: list) -> list:
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

    def find_common_suffix(configs: list) -> str:
        """Helper function to find the common suffix among multiple configurations."""
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
            trimmed_configs = [config[:-(len(common_suffix) + 1)] if common_suffix and config.endswith(common_suffix)
                                                                                    else config for config in configs]

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


