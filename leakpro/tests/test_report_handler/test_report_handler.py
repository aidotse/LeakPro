import logging
import os
import tempfile
import pytest

from leakpro.reporting.report_handler import ReportHandler

@pytest.fixture
def temp_dir():
    """Fixture for creating and cleaning a temporary directory."""
    dir = tempfile.TemporaryDirectory()
    yield dir
    dir.cleanup()

@pytest.fixture
def logger(mocker):
    """Fixture for setting up a mocked logger."""
    mock_logger = mocker.MagicMock(spec=logging.Logger)
    mock_logger.name = "test_logger"
    return mock_logger

@pytest.fixture
def report_handler(temp_dir, logger):
    """Fixture for initializing ReportHandler."""
    return ReportHandler(report_dir=temp_dir.name, logger=logger)

def test_report_handler_initialization(report_handler, temp_dir):
    """Test the initialization of ReportHandler."""
    assert report_handler is not None
    assert report_handler.report_dir == temp_dir.name
    assert isinstance(report_handler.logger, logging.Logger)

    types = ["MIAResult", "GIAResults", "SinglingOutResults", "InferenceResults", "LinkabilityResults"]
    # ensure all types are in the leakpro_types
    assert all(_type in types for _type in report_handler.leakpro_types)
    # ensure all types are initialized to False
    assert all(not report_handler.pdf_results[key] for key in report_handler.leakpro_types)
    # ensure all types are in the global namespace
    assert all(_type in globals() for _type in types)

def test_init_pdf(report_handler):
    """Test the initialization method of the ReportHandler."""
    if hasattr(report_handler, "latex_content"):
        raise AssertionError
    report_handler._init_pdf()
    assert "documentclass" in report_handler.latex_content
    assert "begin" in report_handler.latex_content

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

    assert attack_names == ["Attack1", "Attack2"]

def test_get_results_of_name(report_handler, mocker):
    """Test retrieval of all attack names."""
    result_mock_1 = mocker.MagicMock(resultname="Attack1")
    result_mock_2 = mocker.MagicMock(resultname="Attack2")
    result_mock_3 = mocker.MagicMock(resultname="Attack2")
    result_mock_4 = mocker.MagicMock(resultname="Attack3")
    result_mock_5 = mocker.MagicMock(resultname="Attack3")
    result_mock_6 = mocker.MagicMock(resultname="Attack3")

    report_handler.results = [result_mock_1, result_mock_2, result_mock_3,
                               result_mock_4, result_mock_5, result_mock_6]

    assert len(report_handler._get_results_of_name(report_handler.results, "Attack1")) == 1
    assert len(report_handler._get_results_of_name(report_handler.results, "Attack2")) == 2
    assert len(report_handler._get_results_of_name(report_handler.results, "Attack3")) == 3
