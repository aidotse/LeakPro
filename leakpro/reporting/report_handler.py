import json
import logging
import os
import subprocess

from leakpro.utils.import_helper import Self


# Report Handler
class ReportHandler():
    """Implementation of the report handler."""

    def __init__(self:Self, report_dir: str, logger:logging.Logger) -> None:
        self.logger = logger
        self.report_dir = report_dir
        self.pdf_results = {}
        self.leakpro_types = ["MIAResult",
                           "GIAResults",
                           "SyntheticResult"
                           ]

        # Initiate empty lists for the different types of LeakPro attack types
        for key in self.leakpro_types:
            self.pdf_results[key] = []

    def save_results(self:Self, attack_name: str, result_data: dict, config: dict) -> None:
        """Save attack results."""

        self.logger.info(f"Saving results for {attack_name}")
        result_data.save(self.report_dir, attack_name, config)

    def load_results(self:Self):
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

                            # Dynamically get the class from its name (resulttype)
                            # This assumes that the class is already defined in the current module or imported
                            if resulttype in globals() and callable(globals()[resulttype]):
                                cls = globals()[resulttype]
                            else:
                                raise ValueError(f"Class '{resulttype}' not found.")

                            # Initialize the class using the saved primitives
                            instance = cls(load=True)
                            instance.load(data)

                            if instance.id is None:
                                instance.id = subdir.name

                            if instance.resultname is None:
                                instance.resultname = parentdir.name

                            self.results.append(instance)

                        except Exception as e:
                            self.logger.info(f"Not able to load data, Error: {e}")

    def _get_results_of_name(self:Self, results, resultname_value) -> list:
        indices = [idx for (idx, result) in enumerate(results) if result.resultname == resultname_value]
        return [results[idx] for idx in indices]

    def _get_all_attacknames(self:Self):
        attack_name_list = []
        for result in self.results:
            if result.resultname not in attack_name_list:
                attack_name_list.append(result.resultname)
        return attack_name_list

    def create_results_all(self:Self) -> None:
        for result_type in self.leakpro_types:
            try:
                # Get all results of type "Result"
                results = [res for res in self.results if res.resulttype == result_type]

                # If no results of type "result_type" is found, skip to next result_type
                if not results:
                    self.logger.info(f"No results of type {result_type} found.")
                    continue

                # Check if the result type has a 'create_results' method
                try:
                    result_class = globals().get(result_type)
                except:
                    self.logger.info(f"No {result_type} class could be found or exists")
                    continue

                if hasattr(result_class, "create_results") and callable(result_class.create_results):

                    # Create all results
                    merged_result = result_class.create_results(results=results, save_dir=self.report_dir, save_name="all_results")
                    self.pdf_results[result_type].append(merged_result)

            except Exception as e:
                print("all", e)

    def create_results_strong(self:Self):
        for result_type in self.leakpro_types:
            try:
                # Get all results of type "Result"
                results = [res for res in self.results if res.resulttype == result_type]

                # If no results of type "result_type" is found, skip to next result_type
                if not results:
                    self.logger.info(f"No 'strong' results of type {result_type} found.")
                    continue

                try:
                    result_class = globals().get(result_type)
                except:
                    self.logger.info(f"No {result_type} class could be found or exists")
                    continue

                # Get all attack names
                attack_name_grouped_results = [self._get_results_of_name(results, name) for name in self._get_all_attacknames()]

                # Get the strongest result for each attack name
                if hasattr(result_class, "get_strongest") and callable(result_class.get_strongest):
                    strongest_results = [result_class.get_strongest(result) for result in attack_name_grouped_results]

                    # Create the strongest results
                    merged_result = result_class.create_results(results=strongest_results, save_dir=self.report_dir, save_name="strong_results")
                    self.pdf_results[result_type].append(merged_result)

            except Exception as e:
                print("results_strong", e)

    def create_results_attackname_grouped(self:Self):
        # Get all attack names
        all_attack_names = self._get_all_attacknames()

        for result_type in self.leakpro_types:

                # Get all results of type "Result"
                results = [res for res in self.results if res.resulttype == result_type]

                # If no results of type "result_type" is found, skip to next result_type
                if not results:
                    self.logger.info(f"No results of type {result_type} to group.")
                    continue

                # Check if the result type has a 'create_results' method
                try:
                    result_class = globals().get(result_type)
                except:
                    self.logger.info(f"No {result_type} class could be found or exists")
                    continue

                for name in all_attack_names:
                    if hasattr(result_class, "create_results") and callable(result_class.create_results):
                        try:
                            # Get result for each attack names
                            attack_results = self._get_results_of_name(results, name)

                            # Create results
                            merged_result = result_class.create_results(results=attack_results, save_dir=self.report_dir, save_name="grouped_"+name)
                            self.pdf_results[result_type].append(merged_result)

                        except Exception as e:
                            print("create_results_attackname_grouped", e)

    def create_report(self:Self):
        """Method to create PDF report"""

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

    def _init_pdf(self:Self):
        self.latex_content = """
        \\documentclass{article}
        \\usepackage{tabularx}
        \\usepackage{graphicx}
        \\usepackage{graphics}        
        \\begin{document}
        """

    def _compile_pdf(self:Self, install_flag: bool = False):
        """Method to compile PDF."""

        self.latex_content += """
        \\end{document}
        """
        with open(f"{self.report_dir}/LeakPro_output.tex", "w") as f:
            f.write(self.latex_content)

        # Check if pdflatex is installed
        try:
            check = subprocess.check_output(["which", "pdflatex"], universal_newlines=True)
            assert "pdflatex" in check
        except:
            # Option to install pdflatex
            self.logger.info('Could not find pdflatex installed\nPlease install pdflatex with "apt install texlive-latex-base"')
            choice = input("Do you want to install pdflatex? (Y/n): ").lower()
            if (choice in {"y", "yes"} or install_flag==True):
                proc = subprocess.Popen(["apt", "install", "-y", "texlive-latex-base"], stdout=subprocess.DEVNULL)
                proc.communicate()

        # Compile PDF if possible
        try:
            cmd = ["pdflatex", "-interaction", "nonstopmode", f"{self.report_dir}/LeakPro_output.tex"]
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL)
            proc.communicate()
            self.logger.info("PDF compiled")
        except Exception as e:
            print(e)
            self.logger.info("Could not compile PDF")
