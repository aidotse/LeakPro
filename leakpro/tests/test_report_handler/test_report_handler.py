"""Tests for the report_handler module."""

import logging
import os
import tempfile
from pytest_mock import MockerFixture

from leakpro.reporting.report_handler import ReportHandler
from leakpro.utils.import_helper import Self


class TestReportHandler:
    """Test class of the ReportHandler."""

    def setup_method(self:Self) -> None:
        """Set up temporary directory and logger for ReportHandler."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.logger = logging.getLogger("test_logger")
        self.logger.setLevel(logging.INFO)
        self.report_handler = ReportHandler(report_dir=self.temp_dir.name, logger=self.logger)

    def teardown_method(self:Self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_report_handler_initialization(self:Self) -> None:
        """Test the initialization of ReportHandler."""
        assert self.report_handler is not None
        assert self.report_handler.report_dir == self.temp_dir.name
        assert isinstance(self.report_handler.logger, logging.Logger)

        types = ["MIAResult", "GIAResults", "SinglingOutResults", "InferenceResults", "LinkabilityResults"]
        assert False not in [_type in types for _type in self.report_handler.leakpro_types]
        assert True not in [bool(self.report_handler.pdf_results[key]) for key in self.report_handler.leakpro_types]

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

        self.report_handler._compile_pdf()
        assert "end" in self.report_handler.latex_content
        assert os.path.isfile(f"{self.report_handler.report_dir}/LeakPro_output.tex")