"""Contains the Result classes for MIA, MiNVA, and GIA attacks."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import auc

from leakpro.reporting.report_utils import get_config_name, get_result_fixed_fpr, reduce_to_unique_labels
from leakpro.utils.import_helper import Self


class MIAResult:
    """Contains results related to the performance of the metric."""

    def __init__(  # noqa: PLR0913
        self:Self,
        predicted_labels: list=None,
        true_labels: list=None,
        signal_values:list=None,
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
        self.signal_values = signal_values
        self.audit_indices = audit_indices
        self.metadata = metadata
        self.resultname = resultname
        self.id = id

        if true_labels is None or predicted_labels is None:
            self.tn = self.tp = self.fn = self.fp = 0.0
            self.fpr = self.tpr = self.roc_auc = 0.0
            return

        self.tp = np.sum(predicted_labels & (true_labels == 1), axis=1)
        self.fp = np.sum(predicted_labels & (true_labels == 0), axis=1)
        self.fn = np.sum((true_labels == 1) & ~predicted_labels, axis=1)
        self.tn = np.sum((true_labels == 0) & ~predicted_labels, axis=1)

        self.fpr = np.where(self.fp + self.tn != 0, self.fp / (self.fp + self.tn), 0.0)
        self.tpr = np.where(self.tp + self.fn != 0, self.tp / (self.tp + self.fn), 0.0)

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

        miaresult.id = data["id"]

        return miaresult

    def save(self:Self, path: str, name: str, config:dict = None, show_plot:bool = False) -> None:
        """Save the MIAResults to disk."""

        result_config = config.attack_list[name]
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
