"""Implementation of the Report module."""

import os
import pickle
import subprocess

from leakpro.attacks.mia_attacks.attack_factory_mia import AttackFactoryMIA
from leakpro.reporting.gia_result import GIAResults
from leakpro.reporting.mia_result import MIAResult
from leakpro.schemas import AuditConfig
from leakpro.synthetic_data_attacks.inference_utils import InferenceResults
from leakpro.synthetic_data_attacks.linkability_utils import LinkabilityResults
from leakpro.synthetic_data_attacks.singling_out_utils import SinglingOutResults
from leakpro.utils.import_helper import Self, Union
from leakpro.utils.logger import logger


# Report Handler
class ReportHandler():
    """Implementation of the report handler."""

    def __init__(self:Self, report_dir: str = None) -> None:
        logger.info("Initializing report handler...")

        self.report_dir = self._try_find_rep_dir() if report_dir is None else report_dir
        logger.info(f"report_dir set to: {self.report_dir}")

        self.pdf_results = {}
        self.leakpro_types = ["MIAResult",
                           "GIAResults",
                           "SinglingOutResults",
                            "InferenceResults",
                            "LinkabilityResults"
                           ]

        # Initiate empty lists for the different types of LeakPro attack types
        for key in self.leakpro_types:
            self.pdf_results[key] = []

    def _try_find_rep_dir(self:Self) -> str:
        save_path = "../leakpro_output/results"
        # Check if path exists, otherwise create it.
        for _ in range(3):
            if os.path.exists(save_path):
                return save_path
            save_path = "../"+save_path

        # If no result folder can be found
        if not os.path.exists(save_path):
            save_path = "../../leakpro_output/results"
            os.makedirs(save_path)

        return save_path

    def save_results(self:Self,
                    attack_name: str = None,
                    result_data: Union[MIAResult,
                                       GIAResults,
                                       InferenceResults,
                                       LinkabilityResults,
                                       SinglingOutResults] = None,
                    config: AuditConfig = None) -> None:
        """Save method for results."""

        assert isinstance(result_data, (MIAResult,
                                       GIAResults,
                                       InferenceResults,
                                       LinkabilityResults,
                                       SinglingOutResults))
        assert isinstance(config, AuditConfig)

        logger.info(f"Saving results for {attack_name}")
        attack_config = config.attack_list.get(attack_name, None)

        if attack_config is None:
            attack_config = AttackFactoryMIA.attack_classes[attack_name].AttackConfig()
            config.attack_list[attack_name] = attack_config
        result_data.save(path=self.report_dir, name=attack_name, config=config)

    def load_results(self:Self) -> None:
        """Load method for results."""

        self.results = []
        for parentdir in os.scandir(f"{self.report_dir}"):
            if parentdir.is_dir():
                for subdir in os.scandir(f"{self.report_dir}/{parentdir.name}"):
                    if subdir.is_dir():
                        try:
                            with open(f"{self.report_dir}/{parentdir.name}/{subdir.name}/data.json", "rb") as f:
                                data = pickle.load(f)  # noqa: S301

                            # Extract class name and data
                            result_type = data.result_type

                            # Dynamically get the class from its name (resulttype)
                            # This assumes that the class is already defined in the current module or imported to context
                            if result_type in globals() and callable(globals()[result_type]):
                                cls = globals()[result_type]
                            else:
                                raise ValueError(f"Class '{result_type}' not found.")

                            instance =  cls.load(data)
                            self.results.append(instance)

                        except Exception as e:
                            logger.info(f"In ReportHandler.load_results(), Not able to load data, Error: {e}")

    def create_results(self:Self) -> None:
        """Result method to group all attacks."""

        for result_type in self.leakpro_types:
            try:
            # Get all results of type result_type
                results = [res for res in self.results if res.__class__.__name__ == result_type]

                # If no results of type "result_type" is found, skip to next result_type
                if not results:
                    logger.info(f"No results of type {result_type} found.")
                    continue

                # Check if the result type has a 'create_results' method
                try:
                    result_class = globals().get(result_type)
                except Exception as e:
                    logger.info(f"No {result_type} class could be found or exists. Error: {e}")
                    continue

                if hasattr(result_class, "create_results") and callable(result_class.create_results):

                    # Create all results
                    latex_results = result_class.create_results(results=results, save_dir=self.report_dir)
                    self.pdf_results[result_type].append(latex_results)

            except Exception as e:
                logger.info(f"Error in results all: {result_class}, {e}")

    def create_results_mia(
        self:Self,
        ) -> None:
        """Method to create MIAResult results."""
        self.create_results(types=["MIAResult"])

    def create_results_gia(
        self:Self,
        ) -> None:
        """Method to create GIAResults results."""
        self.create_results(types=["GIAResults"])

    def create_results_syn(
        self:Self,
        ) -> None:
        """Method to create Synthetic results."""

        self.create_results(types=["SinglingOutResults",
                                   "InferenceResults",
                                   "LinkabilityResults"])

    def create_report(self:Self) -> None:
        """Method to create PDF report."""

        # Make sure results have been read and created
        if not hasattr(self, "results"):
            self.load_results()
        if self.pdf_results and all(not value for value in self.pdf_results.values()):
            self.create_results()

        # Create initial part of the document.
        self._init_pdf()

        # Append all results to the document
        for result_type in self.leakpro_types:
            if len(self.pdf_results[result_type]) > 0:
                self.latex_content += f"""\\section{{{result_type}}}"""
                for res in self.pdf_results[result_type]:
                    self.latex_content += res

        # Compile the PDF
        self._compile_pdf()

        # Reset result variables
        self._reset_result()

    def _init_pdf(self:Self) -> None:
        self.latex_content = """
        \\documentclass{article}
        \\usepackage{tabularx}
        \\usepackage{graphicx}
        \\usepackage{graphics}
        \\begin{document}
        """

    def _compile_pdf(self:Self) -> None:
        """Method to compile PDF."""

        # Support for an empty pdf to compile
        if self.latex_content.strip()[-16:] == "\\begin{document}":
            self.latex_content += "\\null"
            logger.info("Warning! You are about to compile an empty pdf.")
            logger.info("Please ensure that you append your results!")

        self.latex_content += """
        \\end{document}
        """
        with open(f"{self.report_dir}/LeakPro_output.tex", "w") as f:
            f.write(self.latex_content)

        try:
            # Check if pdflatex is installed
            check = subprocess.check_output(["which", "pdflatex"], universal_newlines=True) # noqa: S607 S603
            if "pdflatex" not in check:
                logger.info("Could not find pdflatex installed\
                                 \nPlease install pdflatex with apt install texlive-latex-base")

            cmd = ["pdflatex", "-interaction", "nonstopmode", "LeakPro_output.tex"]
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, cwd=f"{self.report_dir}") # noqa: S603
            proc.communicate()
            logger.info("PDF compiled")

        except Exception as e:
            logger.info(f"Could not compile PDF: {e}")
            logger.info("Make sure to install pdflatex with apt install texlive-latex-base")

    def _reset_result(self:Self) -> None:
        del self.results
        for key in self.pdf_results:
            self.pdf_results[key] = []
