import json
import logging
import numpy as np
import os
import subprocess

# from leakpro.reporting.utils import get_config_name
from leakpro.metrics.attack_result import CombinedMetricResult

import matplotlib.pyplot as plt

def load_mia_results(path: str, name: str):
    with open(f'{path}/{name}_data') as f:
        result_data = json.load(f)
    return result_data

# Report Handler
class report_handler():
    """Implementation of the report handler."""

    def __init__(self, report_dir: str, logger:logging.Logger) -> None:
        self.logger = logger
        self.report_dir = report_dir
        self.image_paths = []

    def save_results(self, attack_name: str, result_data: dict, config: dict) -> None:
        """Save attack results. """
    
        self.logger.info(f'Saving results for {attack_name}')
        result_data.save(self.report_dir, attack_name, config)

    def load_results(self):
        self.results = []
        for parentdir in os.scandir(f"{self.report_dir}"):
            if parentdir.is_dir():
                for subdir in os.scandir(f"{self.report_dir}/{parentdir.name}"):
                    if subdir.is_dir():
                        try:
                            with open(f"{self.report_dir}/{parentdir.name}/{subdir.name}/data.json") as f:
                                data = json.load(f)

                            # Extract class name and data
                            resulttype = data["resulttype"]
                            primitives = data["primitives"]
                            config = data["config"]

                            # Dynamically get the class from its name (resulttype)
                            # This assumes that the class is already defined in the current module or imported
                            if resulttype in globals() and callable(globals()[resulttype]):
                                cls = globals()[resulttype]
                            else:
                                raise ValueError(f"Class '{resulttype}' not found.")
                            
                            # Initialize the class using the saved primitives
                            instance = cls(
                                predicted_labels=np.array(primitives["predicted_labels"]),
                                true_labels=np.array(primitives["true_labels"]),
                                predictions_proba=np.array(primitives["predictions_proba"]) if primitives["predictions_proba"] is not None else None,
                                signal_values=np.array(primitives["signal_values"]) if primitives["signal_values"] is not None else None,
                                threshold=np.array(primitives["threshold"]) if primitives["threshold"] is not None else None,
                            )
                            instance.config = config
                            instance.id = subdir.name
                            instance.resultname = parentdir.name
                            self.results.append(instance)

                        except Exception as e:
                            self.logger.info(f"Not able to load data, Error: {e}")

    def _plot_merged_results(
            self,
            merged_results,
            title = "ROC curve",
            save_name = "",
        ):
        
        filename = f"{self.report_dir}/{save_name}"

        for res in merged_results:

            fpr = res.fp / (res.fp + res.tn)
            tpr = res.tp / (res.tp + res.fn)

            range01 = np.linspace(0, 1)
            plt.fill_between(fpr, tpr, alpha=0.15)
            plt.plot(fpr, tpr, label=res.id)

        plt.plot(range01, range01, "--", label="Random guess")
        plt.yscale("log")
        plt.xscale("log")
        plt.tight_layout()
        plt.grid()
        plt.legend()
        plt.xlabel("False positive rate (FPR)")
        plt.ylabel("True positive rate (TPR)")
        plt.title(title)
        plt.xlim(left=1e-5)
        plt.ylim(bottom=1e-5)
        plt.savefig(fname=f"{filename}.png", dpi=1000, bbox_inches='tight')
        plt.clf()
        return filename

    def _get_results_of_name(self, results, resultname_value) -> list:
        indices = [idx for (idx, result) in enumerate(results) if result.resultname == resultname_value]
        return [results[idx] for idx in indices]
    
    def _get_all_attacknames(self):
        attack_name_list = []
        for result in self.results:
            if result.resultname not in attack_name_list:
                attack_name_list.append(result.resultname)
        return attack_name_list

    def create_results_all(self) -> None:
        names = self._plot_merged_results(merged_results=self.results, save_name="all_results")
        self.image_paths.append(names)
        pass
    
    def get_strongest(self, results) -> list:
        return max((res for res in results), key=lambda d: d.roc_auc)

    def create_results_strong(self):
        attack_name_grouped_results = [self._get_results_of_name(self.results, name) for name in self._get_all_attacknames()] 
        strongest_results = [self.get_strongest(attack_name) for attack_name in attack_name_grouped_results]
        names = self._plot_merged_results(merged_results=strongest_results, save_name="strongest_attacks")
        self.image_paths.append(names)
        pass

    def create_results_attackname_grouped(self):
        all_attack_names = self._get_all_attacknames()
        print(all_attack_names)
        for name in all_attack_names:
            attack_results = self._get_results_of_name(self.results, name)
            names = self._plot_merged_results(merged_results=attack_results, save_name="all_"+name)
            self.image_paths.append(names)
        pass

    # TODO: Make other useful groupings of results
    def create_results_numshadowmodels(self):
        pass

    def create_report(self):
        self._init_pdf()

        for image in self.image_paths:
            self._append_to_pdf(image_path=image)

        self._compile_pdf()
        pass

    def _init_pdf(self,):
        self.latex_content = f"""
        \\documentclass{{article}}
        \\usepackage{{graphicx}}
        
        \\begin{{document}}
        """
        pass

    def _append_to_pdf(self, image_path=None, table=None):
        self.latex_content += f"""
        \\begin{{figure}}[ht]
        \\includegraphics[width=0.9\\textwidth]{{{image_path}.png}}
        \\end{{figure}}
        """
        pass

    def _compile_pdf(self):
        self.latex_content += f"""
        \\end{{document}}
        """
        with open(f'{self.report_dir}/LeakPro_output.tex', 'w') as f:
            f.write(self.latex_content)

        cmd = ['pdflatex', '-interaction', 'nonstopmode', f'{self.report_dir}/LeakPro_output.tex']
        proc = subprocess.Popen(cmd)
        pass