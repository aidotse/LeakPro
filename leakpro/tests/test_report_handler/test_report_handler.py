import logging
import os
import tempfile
from pytest_mock import MockerFixture

from leakpro.reporting.report_handler import ReportHandler

@pytest.fixture
def temp_dir():
    """Fixture for creating and cleaning a temporary directory."""
    dir = tempfile.TemporaryDirectory()
    yield dir
    dir.cleanup()

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

    types = ["MIAResult", "GIAResults", "SinglingOutResults", "InferenceResults", "LinkabilityResults"]
    # ensure all types are in the leakpro_types
    assert all(_type in types for _type in report_handler.leakpro_types)
    # ensure all types are initialized to False
    assert all(not report_handler.pdf_results[key] for key in report_handler.leakpro_types)
    # ensure all types are in the global namespace
    assert all(_type in globals() for _type in types)

        types = ["MIAResult", "GIAResults", "SinglingOutResults", "InferenceResults", "LinkabilityResults"]
        assert False not in [_type in types for _type in self.report_handler.leakpro_types]
        assert True not in [bool(self.report_handler.pdf_results[key]) for key in self.report_handler.leakpro_types]

def test_compile_pdf(report_handler):
    """Test PDF compilation."""
    report_handler._init_pdf()
    report_handler._compile_pdf(install_flag=True)

    assert "end" in report_handler.latex_content
    assert os.path.isfile(f"{report_handler.report_dir}/LeakPro_output.tex")
    assert os.path.isfile("./LeakPro_output.pdf")

def test_get_all_attacknames(report_handler, mocker):
    """Test retrieval of all attack names."""
    result_mock_1 = mocker.MagicMock(resultname="Attack1")
    result_mock_2 = mocker.MagicMock(resultname="Attack2")
    report_handler.results = [result_mock_1, result_mock_2, result_mock_1]

    attack_names = report_handler._get_all_attacknames()

        self.report_handler._init_pdf()
        assert ("documentclass" and "begin") in self.report_handler.latex_content

        self.report_handler._compile_pdf()
        assert "end" in self.report_handler.latex_content
        assert os.path.isfile(f"{self.report_handler.report_dir}/LeakPro_output.tex")
