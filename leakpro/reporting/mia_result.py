"""Contains the Result classes for MIA, MiNVA, and GIA attacks."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

from leakpro.reporting.report_utils import create_roc_plot, get_config_name
from leakpro.utils.import_helper import Self


class MIAResult:
    """Contains results related to the performance of the metric."""

    def __init__(  # noqa: PLR0913
        self:Self,
        true_membership: list=None,
        signal_values:list=None,
        audit_indices: list = None,
        metadata: dict = None,
        resultname: str = None,
        id: str = None,
    )-> None:
        """Compute and store the accuracy, ROC AUC score, and the confusion matrix for a metric.

        Args:
        ----
            true_membership: True membership labels used to evaluate the metric.
            signal_values: Values of the signal used by the metric.
            threshold: Threshold computed by the metric.
            audit_indices: The corresponding dataset indices for the results
            id: The identity of the attack
            load: If the data should be loaded
            metadata: Metadata about the results
            resultname: The name of the attack and result

        """
        self.true = np.ravel(true_membership)
        self.signal_values = signal_values
        self.audit_indices = audit_indices
        self.metadata = metadata
        self.resultname = resultname
        self.id = id

        # Compute results
        thresholds = np.unique(signal_values) # ensure we consider each unique value as a threshold
        self.tp, self.fp, self.fn, self.tn = [], [], [], []

        for threshold in thresholds :
            pred = np.ravel(signal_values >= threshold)

            self.tp.append(np.sum(pred & (self.true == 1)))
            self.fp.append(np.sum(pred & (self.true == 0)))
            self.fn.append(np.sum((self.true == 1) & ~pred))
            self.tn.append(np.sum((self.true == 0) & ~pred))

        self.tp = np.array(self.tp)
        self.fp = np.array(self.fp)
        self.fn = np.array(self.fn)
        self.tn = np.array(self.tn)

        self.fpr = np.where(self.fp + self.tn != 0, self.fp / (self.fp + self.tn), 0.0)
        self.tpr = np.where(self.tp + self.fn != 0, self.tp / (self.tp + self.fn), 0.0)

        self.roc_auc = auc(self.fpr, self.tpr)

    def _get_result_fixed_fpr(self: Self, fpr_targets: list[float]) -> dict:
        """Find TPR values for fixed FPRs."""


        results = {}

        for fpr_target in fpr_targets:
            # Get indices where FPR is less than or equal to the target
            valid_indices = np.where(self.fpr <= fpr_target)[0]

            if len(valid_indices) > 0:
                valid_index = valid_indices[-1]
                tpr = self.tpr[valid_index]
            else:
                tpr = 0.0

            results[f"TPR@{fpr_target:.1%}FPR"] = tpr

        return results


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

    def save(self:Self, path: str, name: str, config:dict = None) -> None:
        """Save the MIAResults to disk."""

        result_config = config.attack_list[name]

        fpr_targets = [0.0, 0.0001, 0.001, 0.01, 0.1]
        fixed_fpr_table = self._get_result_fixed_fpr(fpr_targets)

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
            "true_labels": self.true.tolist() if self.true is not None else None,
            "id": name,
        }

        # Check if path exists, otherwise create it.
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Save the results to a file
        with open(f"{save_path}/data.json", "w") as f:
            json.dump(data, f)

        # Create ROC plot
        create_roc_plot([self], save_dir = save_path, save_name = name)

        # Create SignalHistogram plot for MIAResult
        self.create_signal_histogram(save_path = save_path)

    @staticmethod
    def get_strongest(results: list) -> list:
        """Method for selecting the strongest attack."""
        return max((res for res in results), key=lambda d: d.roc_auc)

    def create_signal_histogram(self:Self, save_path: str) -> None:
        """Method to create Signal Histogram."""

        filename = f"{save_path}/SignalHistogram"
        values = np.array(self.signal_values).ravel()
        labels = np.array(self.true).ravel()

        # Split values by membership
        member_values = values[labels == 1]
        non_member_values = values[labels == 0]

        # Compute bin edges (shared for both histograms)
        bin_edges = np.histogram_bin_edges(values, bins=1000)

        # Plot histograms
        plt.hist(non_member_values, bins=bin_edges, histtype="step", label="out-member", density=False)
        plt.hist(member_values, bins=bin_edges, histtype="step", label="in-member", density=False)

        plt.grid()
        plt.xlabel("Signal value")
        plt.ylabel("Number of samples")
        plt.title("Signal histogram")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fname=filename + ".png", dpi=1000)
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
