"""Tests for the report_handler module."""

import logging
import os
import tempfile

import numpy as np

from leakpro.reporting.report_handler import ReportHandler
from leakpro.utils.import_helper import Self
from leakpro.reporting.mia_result import MIAResult


class TestReportHandler:
    """Test class of the ReportHandler."""

    def setup_method(self:Self) -> None:
        """Set up temporary directory and logger for ReportHandler."""
        
        true_labels = np.array([False,  True,  True, True,  False, False])
        signal_values =  np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        result_name = "dummy"
        id = "dummy-test"

        miaresult = MIAResult.from_full_scores(true_membership = true_labels,
                                    signal_values = signal_values,
                                    result_name = result_name,
                                    id = id)
        
        
        self.temp_dir = tempfile.TemporaryDirectory()
        self.logger = logging.getLogger("test_logger")
        self.logger.setLevel(logging.INFO)
        self.report_handler = ReportHandler(results=[miaresult], report_dir=self.temp_dir.name)
        


    def teardown_method(self:Self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_report_handler_initialization(self:Self) -> None:
        """Test the initialization of ReportHandler."""
        assert self.report_handler is not None
        assert self.report_handler.report_dir == self.temp_dir.name

        types = ["MIAResult", "GIAResults", "SinglingOutResults", "InferenceResults", "LinkabilityResults"]
        
        # check that pdf_results include all types and is empty
        for key in self.report_handler.pdf_results:
            assert key in types
            assert self.report_handler.pdf_results[key] == []

        assert len(self.report_handler.results) == 1

    def test_init_pdf(self:Self) -> None:
        """Test the initialization method of the ReportHandler."""

        if hasattr(self.report_handler, "latex_content"):
            raise AssertionError

        self.report_handler._init_pdf()
        assert "documentclass" in self.report_handler.latex_content
        assert "begin" in self.report_handler.latex_content

    def test_compile_pdf(self:Self) -> None:
        """Test PDF compilation."""

        self.report_handler._init_pdf()
        assert ("documentclass" and "begin") in self.report_handler.latex_content

        # Add body to latex content        
        self.report_handler.latex_content += "body text example"

        self.report_handler._compile_pdf()
        assert "end" in self.report_handler.latex_content
        assert os.path.isfile(f"{self.report_handler.report_dir}/LeakPro_output.tex")
        assert os.path.isfile(f"{self.report_handler.report_dir}/LeakPro_output.pdf")

    def test_create_pdf(self:Self) -> None:
        # Create pdf report
        self.report_handler.create_report()

        assert os.path.isfile(f"{self.report_handler.report_dir}/LeakPro_output.pdf")