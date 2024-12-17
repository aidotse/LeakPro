"""Module containing classes to generate reports from metric results."""
import datetime
import json
import os
import subprocess
from abc import ABC, abstractmethod
from typing import Optional

import jinja2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from scipy import interpolate

from leakpro.metrics.attack_result import AttackResult, CombinedMetricResult
from leakpro.utils.import_helper import Dict, List, Tuple, Union

########################################################################################################################
# GLOBAL SETTINGS
########################################################################################################################

# Temporary parameter pointing to the report_file directory (for compatibility)
REPORT_FILES_DIR = "report_files"

# Configure jinja for LaTex
latex_jinja_env = jinja2.Environment(
    block_start_string=r"\BLOCK{",
    block_end_string="}",
    variable_start_string=r"\VAR{",
    variable_end_string="}",
    comment_start_string=r"\#{",
    comment_end_string="}",
    line_statement_prefix="%%",
    line_comment_prefix="%#",
    trim_blocks=True,
    autoescape=True,
    loader=jinja2.FileSystemLoader(os.path.abspath(".")),
)

# Config the name for metric id.
EXPLANATIONS = {
    "metric": {
        "shadow_metric": {
            "name": "Shadow metric",
            "details": "The shadow metric is a membership inference metric, that uses the algorithm described in \\cite{https://doi.org/10.48550/arxiv.1610.05820}",
        },
        "population_metric": {
            "name": "Population metric",
            "details": "[TODO] Add details in explanations.json (you can also edit citations.bib).",
        },
        "reference_metric": {
            "name": "Reference metric",
            "details": "[TODO] Add details in explanations.json (you can also edit citations.bib).",
        },
    },
    "figure": {
        "roc_curve": {
            "name": "ROC curve",
            "details": "shows the ROC (Receiver Operating Characteristic) curve, a graph illustrating the performance of a classification model at various decision thresholds. The AUC (Area Under the Curve), represented in blue, is a threshold-independant measure of the classifier performance.\n\nA higher AUC is an indicator of a system vulnerable to the chosen metric. For reference, a random classifier yields an AUC of 0.5, while a perfect classifier yields an AUC of 1.0",  # noqa: E501
        },
        "confusion_matrix": {
            "name": "Confusion matrix",
            "details": "shows the confusion matrix, a graph illustrating the performance of a classification model for a specific decision threshold.\n\nHigher values on the top-left to bottom-right diagonal is an indicator of a system vulnerable to the chosen metric, while higher values on the top-right to bottom-left diagonal is an indicator of a system less vulnerable to the chosen metric.",  # noqa: E501
        },
        "signal_histogram": {
            "name": "Signal histogram",
            "details": "shows the histogram of the signal used by the chosen metric, on both members and non-member samples.\n\nA clear separation between the two groups is an indicator of a system vulnerable to the chosen metric.",  # noqa: E501
        },
        "vulnerable_points": {
            "name": "Vulnerable points",
            "details": "shows points that are most vulnerable to the chosen metric.\n\nThe score depends on the chosen metric, but is always between 0 and 1, with 0 meaning low vulnerability and 1 high vulnerability.",  # noqa: E501
        },
    },
}

########################################################################################################################
# AUDIT_REPORT CLASS
########################################################################################################################


class AuditReport(ABC):
    """An abstract class to display and/or save some elements of a metric result object."""

    @staticmethod
    @abstractmethod
    def generate_report(
        metric_result: Union[
            AttackResult, List[AttackResult], dict, CombinedMetricResult
        ]
    ) -> None:
        """Core function of the AuditReport class that actually generates the report.

        Args:
        ----
            metric_result: MetricResult object, containing data for the report.

        """
        pass


########################################################################################################################
# ROC_CURVE_REPORT CLASS
########################################################################################################################


class ROCCurveReport(AuditReport):
    """An interface class to display and/or save some elements of a metric result object.

    This particular class is used to generate a ROC (Receiver Operating Characteristic) curve.
    """

    @staticmethod
    def __avg_roc(
        fpr_2d_list: List[List[float]], tpr_2d_list: List[List[float]], n: int = 200
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Private helper function, to average a ROC curve from non-aligned list.

        Args:
        ----
            fpr_2d_list: A 2D list of fpr values.
            tpr_2d_list: A 2D list of fpr values.
            n: Number of points in the resulting lists.

        Returns:
        -------
            A tuple of aligned 1D numpy arrays, fpr and tpr.

        """
        functions = [
            interpolate.interp1d(fpr, tpr)
            for (fpr, tpr) in zip(fpr_2d_list, tpr_2d_list)
        ]
        fpr = np.linspace(0, 1, n)
        tpr = np.mean([f(fpr) for f in functions], axis=0)
        return fpr, tpr

    @staticmethod
    def generate_report(
        metric_result: Union[
            AttackResult,
            List[AttackResult],
            List[List[AttackResult]],
            CombinedMetricResult,
        ],
        show: bool = False,
        save: bool = True,
        filename: str = "roc_curve.jpg",
        configs: dict = None,  # noqa: ARG004,
    ) -> None:
        """Core function of the AuditReport class that actually generates the report.

        Args:
        ----
            metric_result: A list of MetricResult objects, containing data for the report.
            inference_game_type: Value from the InferenceGame ENUM type, indicating which inference game was used.
            show: Boolean specifying if the plot should be displayed on screen.
            save: Boolean specifying if the plot should be saved as a file.
            filename: File name to be used if the plot is saved as a file.
            configs: Dictionary containing the configuration of the audit.

        """
        # Check if it is the combined report:
        if not isinstance(metric_result, list):
            metric_result = [metric_result]
        if not isinstance(metric_result[0], CombinedMetricResult) and not isinstance(
            metric_result[0][0], CombinedMetricResult
        ):
            # Casts type to a 2D list
            if not isinstance(metric_result[0], list):
                metric_result = [metric_result]
            # Computes fpr, tpr and auc in different ways, depending on the available information and inference game
            if metric_result[0][0].predictions_proba is None:
                fpr = [mr.fp / (mr.fp + mr.tn) for mr in metric_result[0]]
                tpr = [mr.tp / (mr.tp + mr.fn) for mr in metric_result[0]]
                roc_auc = np.trapz(x=fpr, y=tpr)
            else:
                fpr, tpr, _ = metric_result[0][0].roc
                roc_auc = metric_result[0][0].roc_auc
        elif metric_result[0].predictions_proba is None:
            mr = metric_result[0]
            fpr = mr.fp / (mr.fp + mr.tn)
            tpr = mr.tp / (mr.tp + mr.fn)
            roc_auc = np.trapz(x=fpr, y=tpr)

        # save the data to a csv file
        directory = os.path.dirname(filename)

        # If the directory doesn't exist, create it
        if not os.path.exists(directory):
            os.makedirs(directory)


        with open(filename.replace(".png", ".csv"), "w") as f:
            f.write("fpr,tpr\n")
            for i in range(len(fpr)):
                f.write(f"{fpr[i]},{tpr[i]}\n")


        fixed_fpr_results(fpr, tpr, configs, filename)

        # Gets metric ID
        # TODO: add metric ID to the CombinedMetricResult class
        metric_id = "population_metric"

        # Generate plot
        range01 = np.linspace(0, 1)
        plt.fill_between(fpr, tpr, alpha=0.15)
        plt.plot(fpr, tpr, label=EXPLANATIONS["metric"][metric_id]["name"])
        plt.plot(range01, range01, "--", label="Random guess")
        plt.yscale("log")
        plt.xscale("log")
        plt.tight_layout()
        plt.grid()
        plt.legend()
        plt.xlabel("False positive rate (FPR)")
        plt.ylabel("True positive rate (TPR)")
        plt.title("ROC curve")
        plt.text(
            0.7,
            0.3,
            f"AUC = {roc_auc:.03f}",
            horizontalalignment="center",
            verticalalignment="center",
            bbox={"facecolor": "white", "alpha": 0.5},
        )
        plt.xlim(left=1e-5)
        plt.ylim(bottom=1e-5)
        if save:
            plt.savefig(fname=filename, dpi=1000)
        if show:
            plt.show()
        plt.clf()


########################################################################################################################
# CONFUSION_MATRIX_REPORT CLASS
########################################################################################################################


class ConfusionMatrixReport(AuditReport):
    """An interface class to display and/or save some elements of a metric result object.

    This particular class is used to generate a confusion matrix.
    """

    @staticmethod
    def generate_report(
        metric_result: Union[AttackResult, List[AttackResult]],
        show: bool = False,
        save: bool = True,
        filename: str = "confusion_matrix.jpg",
    ) -> None:
        """Core function of the AuditReport class that actually generates the report.

        Args:
        ----
            metric_result: MetricResult object, containing data for the report.
            inference_game_type: Value from the InferenceGame ENUM type, indicating which inference game was used.
            show: Boolean specifying if the plot should be displayed on screen.
            save: Boolean specifying if the plot should be saved as a file.
            filename: File name to be used if the plot is saved as a file.

        """
        if not isinstance(metric_result, AttackResult):
            raise ValueError("metric_result must be an instance of AttackResult"
                             )
        cm = np.array(
            [
                [metric_result.tn, metric_result.fp],
                [metric_result.fn, metric_result.tp],
            ]
        )

        cm = 100 * cm / np.sum(cm)
        index = ["Non-member", "Member"]
        df_cm = pd.DataFrame(cm, index, index)
        sn.heatmap(df_cm, annot=True, cmap=plt.cm.Blues)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Confusion matrix (in %)")
        if save:
            plt.savefig(fname=filename, dpi=1000)
        if show:
            plt.show()
        plt.clf()


########################################################################################################################
# SIGNAL_HISTOGRAM_REPORT CLASS
########################################################################################################################


class SignalHistogramReport(AuditReport):
    """An interface class to display and/or save some elements of a metric result object.

    This particular class is used to generate a histogram of the signal values.
    """

    @staticmethod
    def generate_report(
        metric_result: Union[AttackResult, List[AttackResult]],
        show: bool = False,
        save: bool = True,
        filename: str = "signal_histogram.jpg",
    ) -> None:
        """Core function of the AuditReport class that actually generates the report.

        Args:
        ----
            metric_result: MetricResult object, containing data for the report.
            inference_game_type: Value from the InferenceGame ENUM type, indicating which inference game was used.
            show: Boolean specifying if the plot should be displayed on screen.
            save: Boolean specifying if the plot should be saved as a file.
            filename: File name to be used if the plot is saved as a file.

        """
        values = np.array(metric_result.signal_values).ravel()
        labels = np.array(metric_result.true_labels).ravel()
        threshold = metric_result.threshold

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
        if save:
            plt.savefig(fname=filename, dpi=1000)
        if show:
            plt.show()
        plt.clf()


########################################################################################################################
# VULNERABLE_POINTS_REPORT CLASS
########################################################################################################################


class VulnerablePointsReport(AuditReport):
    """An interface class to display and/or save some elements of a metric result object.

    This particular class is used to identify the most vulnerable points.
    """

    @staticmethod
    def generate_report(  # noqa: PLR0913
        metric_results: List[AttackResult],
        number_of_points: int = 10,
        save_tex: bool = False,
        filename: str = "vulnerable_points.tex",
        return_raw_values: bool = True,
        point_type: str = "any",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Core function of the AuditReport class that actually generates the report.

        Args:
        ----
            metric_results: A dict of lists of MetricResult objects, containing data for the report.
            target_info_source: The InformationSource associated with the audited model training.
            target_model_to_train_split_mapping: The mapping associated with target_info_source.
            number_of_points: Number of vulnerable to be selected.
            save_tex: Boolean specifying if a partial .tex file should be generated.
            filename: Filename of the partial .tex file.
            return_raw_values: Boolean specifying if the points indices and scores should be returned.
            point_type: Can be "any" or "image". If "image", then the images are displayed as such in the report.

        Returns:
        -------
            Indices of the vulnerable points and their scores.

        """
        # Objects to be returned if return_raw_values is True
        indices, scores = [], []

        # If only one metric was used (i.e. we have access to the prediction probabilities)
        if len(metric_results) == 1:
            mr = metric_results[0]
            # Sort the training points that were identified as such by their prediction probabilities
            adjusted_values = np.where(
                (np.array(mr.predicted_labels) == np.array(mr.true_labels))
                & (np.array(mr.true_labels) == 1),
                -mr.predictions_proba,
                10,
            )
            indices = np.argsort(adjusted_values)[:number_of_points]
            # Get the associated scores
            scores = mr.predictions_proba[indices]

        # If multiple metrics were used (i.e. we don't have access to the prediction probabilities)
        else:
            # Use the various metric, from the one with lowest fpr to the one with highest fpr
            fp_indices = np.argsort([mr.fp for mr in metric_results])
            for k in range(len(metric_results)):
                mr = metric_results[fp_indices[k]]
                # Get the training points that were identified as such
                new_indices = np.argwhere(
                    (np.array(mr.predicted_labels) == np.array(mr.true_labels))
                    & (np.array(mr.true_labels) == 1)
                )
                indices.extend(list(new_indices.ravel()))
                # Get the associated scores
                fpr = mr.fp / (mr.fp + mr.tn)
                scores.extend([1 - fpr] * new_indices.shape[0])
            # Only keep number_of_points points
            indices, scores = indices[:number_of_points], scores[:number_of_points]

        # Map indices stored in the metric_result object to indices in the training set
        indices_to_train_indices = []
        counter = 0
        for _, v in enumerate(metric_results[0].true_labels):
            indices_to_train_indices.append(counter)
            counter += v
        indices = np.array(indices_to_train_indices)[np.array(indices)]

        # If we are creating a LaTex
        if save_tex:
            # Load template
            template = latex_jinja_env.get_template(
                f"{REPORT_FILES_DIR}/vulnerable_points_template.tex"
            )

            # Render the template (i.e. generate the corresponding string)
            latex_content = template.render(
                points=[
                    {
                        "index": index,
                        "score": f"{score:.3f}",
                        "type": point_type,
                        "path": f"point{k:03d}.jpg" if point_type == "image" else None,
                    }
                    for (k, (index, score)) in enumerate(zip(indices, scores))
                ]
            )

            # Write the result (the string) to a .tex file
            with open(filename, "w") as f:
                f.write(latex_content)

        # If we required the values to be returned
        if return_raw_values:
            return indices, scores
        return None


########################################################################################################################
# PDF_REPORT CLASS
########################################################################################################################


class PDFReport(AuditReport):
    """An interface class to display and/or save some elements of a metric result object.

    This particular class is used to generate a user-friendly report, with multiple plots and some explanations.
    """

    @staticmethod
    def generate_report(  # noqa: PLR0913, D417
        metric_results: Dict[
            str, Union[AttackResult, List[AttackResult], List[List[AttackResult]]]
        ],
        figures_dict: dict,
        system_name: str,
        call_pdflatex: bool = True,
        show: bool = False,  # noqa: ARG004
        save: bool = True,  # noqa: ARG004
        filename_no_extension: str = "report",
        point_type: str = "any",
    ) -> None:
        """Core function of the AuditReport class that actually generates the report.

        Args:
        ----
            metric_results: A dict of lists of MetricResult objects, containing data for the report.
            inference_game_type: Value from the InferenceGame ENUM type, indicating which inference game was used.
            figures_dict: A dictionary containing the figures to include, for each metric result.
                E.g. {"shadow_metric": ["roc_curve", "confusion_matrix", "signal_histogram"]}.
            system_name: Name of the system being audited. E.g. "Purchase100 classifier".
            call_pdflatex: Boolean to specify if the pdflatex compiler should be called (to get a PDF file from the
                TEX file).
            show: Boolean specifying if the plot should be displayed on screen.
            save: Boolean specifying if the plot should be saved as a file.
            filename_no_extension: File name to be used if the plot is saved as a file, without the file extension.

        """
        for metric in metric_results:
            if not isinstance(metric_results[metric], list):
                metric_results[metric] = [metric_results[metric]]
            if not isinstance(metric_results[metric][0], list):
                metric_results[metric] = [[mr] for mr in metric_results[metric]]

        # Generate all plots, and save their filenames
        files_dict = {}
        for metric in metric_results:
            files_dict[metric] = {}
            result = metric_results[metric]

            # Select one instance to display when necessary (e.g. for a confusion matrix with a PopulationMetric)
            best_index = np.argmax([r.accuracy for r in result])
            best_result = result[best_index]

            if "roc_curve" in figures_dict[metric]:
                figure = "roc_curve"
                filename = f"{metric}_{figure}.jpg"
                files_dict[metric][figure] = filename
                ROCCurveReport.generate_report(
                    metric_result=result,
                    filename=filename,
                )
            if "confusion_matrix" in figures_dict[metric]:
                figure = "confusion_matrix"
                filename = f"{metric}_{figure}.jpg"
                files_dict[metric][figure] = filename
                ConfusionMatrixReport.generate_report(
                    metric_result=best_result,
                    filename=filename,
                )
            if "signal_histogram" in figures_dict[metric]:
                figure = "signal_histogram"
                filename = f"{metric}_{figure}.jpg"
                files_dict[metric][figure] = filename
                SignalHistogramReport.generate_report(
                    metric_result=best_result,
                    filename=filename,
                )
            if "vulnerable_points" in figures_dict[metric]:
                figure = "vulnerable_points"
                filename = f"{metric}_{figure}.tex"
                files_dict[metric][figure] = filename
                VulnerablePointsReport.generate_report(
                    metric_results=result,
                    save_tex=True,
                    filename=filename,
                    point_type=point_type,
                )

        # Load template
        template = latex_jinja_env.get_template(
            f"{REPORT_FILES_DIR}/report_template.tex"
        )

        # Render the template (i.e. generate the corresponding string)
        latex_content = template.render(
            bib_file=os.path.abspath(f"{REPORT_FILES_DIR}/citations.bib"),
            image_folder=os.path.abspath("."),
            name=system_name,
            tool_version="1.0",
            report_date=datetime.datetime.now().date().strftime("%b-%d-%Y"),  # noqa: DTZ005
            explanations=EXPLANATIONS,
            figures_dict=figures_dict,
            files_dict=files_dict,
        )

        # Write the result (the string) to a .tex file
        with open(f"{filename_no_extension}.tex", "w") as f:
            f.write(latex_content)

        print(f'LaTex file created:\t{os.path.abspath(f"{filename_no_extension}.tex")}')  # noqa: T201

        if call_pdflatex:
            # Compile the .tex file to a .pdf file. Several rounds are required to get the references (to papers, to
            # page numbers, and to figure numbers)

            process = subprocess.Popen(  # noqa: S603
                ["pdflatex", os.path.abspath(f"{filename_no_extension}.tex")],  # noqa: S607, S603
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = process.communicate()

            process = subprocess.Popen( # noqa: S603
                ["biber", os.path.abspath(f"{filename_no_extension}")], # noqa: S607, S603
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = process.communicate()

            process = subprocess.Popen( # noqa: S603
                ["pdflatex", os.path.abspath(f"{filename_no_extension}.tex")], # noqa: S607, S603
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = process.communicate()

            process = subprocess.Popen( # noqa: S603
                ["pdflatex", os.path.abspath(f"{filename_no_extension}.tex")], # noqa: S607, S603
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = process.communicate()

            print(  # noqa: T201
                f'PDF file created:\t{os.path.abspath(f"{filename_no_extension}.pdf")}'
            )

def read_and_parse_data(filename:str) -> dict:
    """Read and parse data from a file.

    Args:
    ----
        filename (str): The name of the file to read.
        logger (logging.Logger): The logger object for logging messages.

    Returns:
    -------
        dict: A dictionary containing the parsed data.

    """
    data = {}
    try:
        with open(filename, "r") as file:
            content = file.read().strip().split("\n\n")  # Split by empty row
            for block in content:
                if block.strip():
                    lines = block.split("\n")
                    config = lines[0]
                    fpr_tpr = lines[1]
                    data[config] = fpr_tpr
    except FileNotFoundError:
        print(f"No existing file named '{filename}'. A new file will be created.")  # noqa: T201
    return data

# Main logic to process and save results
def fixed_fpr_results(fpr:np.ndarray, tpr:np.ndarray, configs:dict, filename:str = None) -> None:
    """Compute and save fixed FPR results.

    Args:
    ----
        fpr (np.ndarray): Array of false positive rates.
        tpr (np.ndarray): Array of true positive rates.
        configs (dict): Dictionary of attack configurations.
        filename (str): Name of the file to save the results.
        logger (logging.Logger): The logger object for logging messages.

    Returns:
    -------
        None

    """
    # Split the path into components
    path_components = filename.split("/")

    # Make the path for "results.txt"
    path_components[-1] = "results.txt"

    # Join the components back into a full path
    filename = "/".join(path_components)

    # Serialize configuration
    attack_name = list(configs["attack_list"].keys())[0]
    attack_configs = configs["attack_list"][attack_name]
    config_key = json.dumps(attack_configs, sort_keys=True)

    # Function to find TPR at given FPR thresholds
    def find_tpr_at_fpr(fpr_array:np.ndarray, tpr_array:np.ndarray, threshold:float) -> Optional[str]:
        try:
            # Find the last index where FPR is less than the threshold
            valid_index = np.where(fpr_array < threshold)[0][-1]
            return f"{tpr_array[valid_index] * 100:.4f}%"
        except IndexError:
            # Return None or some default value if no valid index found
            return "N/A"

    # Compute TPR values at various FPR thresholds
    results = f"TPR@1.0%FPR: {find_tpr_at_fpr(fpr, tpr, 0.01)}%, " \
              f"TPR@0.1%FPR: {find_tpr_at_fpr(fpr, tpr, 0.001)}%, " \
              f"TPR@0.01%FPR: {find_tpr_at_fpr(fpr, tpr, 0.0001)}%, " \
              f"TPR@0.0%FPR: {find_tpr_at_fpr(fpr, tpr, 0.00001)}%"

    # Load existing data
    data = read_and_parse_data(filename)

    # Update or append data
    data[config_key] = results

    # Save updated data
    with open(filename, "w") as file:
        for config, results in data.items():
            file.write(f"{config}\n{results}\n\n")
