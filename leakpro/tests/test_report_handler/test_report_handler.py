"""Tests for the report_handler module."""

import logging
import os
import tempfile
from unittest.mock import MagicMock

from leakpro.reporting.report_handler import ReportHandler
from leakpro.utils.import_helper import Self


class TestReportHandler():
    """Test class of the ReportHandler."""

    def setUp(self:Self) -> None:
        """Set up temporary directory and logger for ReportHandler."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.logger = logging.getLogger("test_logger")
        self.logger.setLevel(logging.INFO)
        self.report_handler = ReportHandler(report_dir=self.temp_dir.name, logger=self.logger)

    def tearDown(self:Self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_report_handler_initialization(self:Self) -> None:
        """Test the initialization of ReportHandler."""
        assert self.report_handler is not None
        assert self.report_handler.report_dir == self.temp_dir.name
        assert isinstance(self.report_handler.logger, logging.Logger)

        types = ["MIAResult", "GIAResults", "SyntheticResult"]
        assert False not in [_type in types for _type in self.report_handler.leakpro_types]
        assert True not in [bool(self.report_handler.pdf_results[key]) for key in self.report_handler.leakpro_types]
        assert False not in [_type in globals() for _type in types]

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
        self.report_handler._compile_pdf(install_flag=True)

        assert "end" in self.report_handler.latex_content
        assert os.path.isfile(f"{self.report_handler.report_dir}/LeakPro_output.tex")
        assert os.path.isfile("./LeakPro_output.pdf")

    def test_get_all_attacknames(self:Self) -> None:
        """Test retrieval of all attack names."""
        result_mock_1 = MagicMock(resultname="Attack1")
        result_mock_2 = MagicMock(resultname="Attack2")
        self.report_handler.results = [result_mock_1, result_mock_2, result_mock_1]

        attack_names = self.report_handler._get_all_attacknames()

        assert attack_names == ["Attack1", "Attack2"]

    def test_get_results_of_name(self:Self) -> None:
        """Test retrieval of all attack names."""
        result_mock_1 = MagicMock(resultname="Attack1")
        result_mock_2 = MagicMock(resultname="Attack2")
        result_mock_3 = MagicMock(resultname="Attack2")
        result_mock_4 = MagicMock(resultname="Attack3")
        result_mock_5 = MagicMock(resultname="Attack3")
        result_mock_6 = MagicMock(resultname="Attack3")

        self.report_handler.results = [result_mock_1, result_mock_2, result_mock_3,
                                       result_mock_4, result_mock_5, result_mock_6]

        assert len(self.report_handler._get_results_of_name(self.report_handler.results, "Attack1")) == 1
        assert len(self.report_handler._get_results_of_name(self.report_handler.results, "Attack2")) == 2
        assert len(self.report_handler._get_results_of_name(self.report_handler.results, "Attack3")) == 3
