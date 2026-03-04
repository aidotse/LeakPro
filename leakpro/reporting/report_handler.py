"""Implementation of the Report module."""

import subprocess

from leakpro.metrics.attack_result import GIAResults
from leakpro.reporting.mia_result import MIAResult
from leakpro.synthetic_data_attacks.inference_utils import InferenceResults
from leakpro.synthetic_data_attacks.linkability_utils import LinkabilityResults
from leakpro.synthetic_data_attacks.singling_out_utils import SinglingOutResults
from leakpro.utils.import_helper import Self, Union
from leakpro.utils.logger import logger

ResultList = Union[list[MIAResult],
                    list[GIAResults],
                    list[SinglingOutResults],
                    list[InferenceResults],
                    list[LinkabilityResults]
                    ]

# Report Handler
class ReportHandler():
    """Implementation of the report handler."""

    def __init__(self:Self, results:ResultList, report_dir: str) -> None:
        logger.info("Initializing report handler...")

        self.report_dir = report_dir
        logger.info(f"report_dir set to: {self.report_dir}")

        self.pdf_results = {}
        self.leakpro_types = {"MIAResult" : MIAResult,
                              "GIAResults" : GIAResults,
                              "SinglingOutResults" : SinglingOutResults,
                              "InferenceResults" : InferenceResults,
                              "LinkabilityResults" : LinkabilityResults}

        # Initiate empty lists for the different types of LeakPro attack types
        for key in self.leakpro_types:
            self.pdf_results[key] = []

        self.results = []
        for res in results:
            assert isinstance(res, (MIAResult, GIAResults, InferenceResults, LinkabilityResults, SinglingOutResults))
            self.results.append(res)

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

                result_class = self.leakpro_types[result_type]
                assert hasattr(result_class, "create_results"), f"No create_results in result class {result_class}."
                assert callable(result_class.create_results), f" create_results is not callable in result class {result_class}."

                # Create all results
                latex_results = result_class.create_results(results=results, save_dir=self.report_dir)
                self.pdf_results[result_type].append(latex_results)

            except Exception as e:
                logger.info(f"Error in results all: {result_class}, {e}")

    def create_report(self:Self) -> None:
        """Method to create PDF report."""

        assert self.results, "No results found. Please run the audit first."
        assert self.report_dir, "No report directory found. Please set the report directory first."

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
