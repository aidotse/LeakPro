"""Tests for the attack_result module."""

import os
import tempfile
from pytest_mock import MockerFixture

import numpy as np

from leakpro.reporting.mia_result import MIAResult
from leakpro.utils.import_helper import Self

class TestMIAResult:
    """Test class for MIAResult."""

    def setup_method(self:Self) -> None:
        """Set up temporary directory and logger for MIAResult."""
        self.temp_dir = tempfile.TemporaryDirectory()

        true_labels = np.array([False,  True,  True, True,  False, False])
        signal_values =  np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        result_name = "dummy-test"

        self.miaresult_full = MIAResult.from_full_scores(true_membership = true_labels,
                                                        signal_values = signal_values,
                                                        result_name = result_name+ "-full")
        
        self.miaresult_fixed = MIAResult.from_fixed_thresholds(true_membership = true_labels,
                                                            signal_values = signal_values,
                                                            result_name = result_name + "-fixed",
                                                            thresholds = [signal_values[-1], signal_values[0]])
        
        self.miaresult_confusion = MIAResult.from_confusion_counts(true_membership = true_labels,
                                                                result_name = result_name+ "-confusion",
                                                                tp= 3,
                                                                fp= 3,
                                                                tn= 2,
                                                                fn= 1,)

        self.config = {"random_seed": 1234, "attack_list":
                            {"lira":
                                    {"training_data_fraction": 0.5,
                                     "num_shadow_models": 3,
                                     "online": True}
                            },
                        "report_log":
                                "./leakpro_output/results",
                                "config_log":
                                        "./leakpro_output/config",
                                        "target_model_folder":
                                                        "./target",
                                                        "attack_folder":
                                                                "attack_objects",
                                                                "attack_type":
                                                                        "mia",
                                                                        "split_method":
                                                                                "no_overlapping"
                        }
        
        self.fpr_array = np.array([0.33333333, 0.66666667, 0.66666667, 0.66666667, 0.66666667, 1.0])
        self.tpr_array = np.array([0.0, 0.0,0.33333333, 0.66666667, 1.0, 1.0])
        self.fp = [1,2,2,2,2,3]
        self.tn = [2,1,1,1,1,0]
        
        self.fixed_fpr = [self.fpr_array[0], self.fpr_array[-1]]
        self.fixed_tpr = [self.tpr_array[0], self.tpr_array[-1]]
        self.fixed_fp = [self.fp[0], self.fp[-1]]
        self.fixed_tn = [self.tn[0], self.tn[-1]]
        

    def teardown_method(self:Self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_miaresult_init(self:Self) -> None:
        """Test the initialization of MIAResult."""
        assert self.miaresult_full.result_name == "dummy-test-full"
        assert self.miaresult_fixed.result_name == "dummy-test-fixed"
        assert self.miaresult_confusion.result_name == "dummy-test-confusion"

    def test_check_tpr_fpr(self:Self) -> None:
        """Test fpr and tpr."""

        # Check the full version
        assert np.allclose(self.miaresult_full.fpr, self.fpr_array)
        assert np.allclose(self.miaresult_full.tpr, self.tpr_array)
        np.testing.assert_array_equal(self.miaresult_full.fp, self.fp)
        np.testing.assert_array_equal(self.miaresult_full.tn, self.tn)
        
        # Check the fixed version
        assert np.allclose(self.miaresult_fixed.fpr, self.fixed_fpr)
        assert np.allclose(self.miaresult_fixed.tpr, self.fixed_tpr)
        np.testing.assert_array_equal(self.miaresult_fixed.fp, self.fixed_fp)
        np.testing.assert_array_equal(self.miaresult_fixed.tn, self.fixed_tn)
        

    def test_save_load_miaresult(self:Self, mocker: MockerFixture) -> None:
        """Test load and save functionality."""

        name = "lira"
        save_path = f"{self.temp_dir}/results/{self.miaresult_full.id}"
        data_storage_path = f"{self.temp_dir}/data_objects/"

        # Test saving
        attack_mock = mocker.Mock(attack_id=self.miaresult_full.id)
        self.miaresult_full.save(attack_mock, self.temp_dir)

        assert os.path.isdir(save_path)
        assert os.path.exists(data_storage_path)
        assert os.path.exists(f"{save_path}/ROC.png")
        assert os.path.exists(f"{save_path}/SignalHistogram.png")

        # Test loading
        data_path = f"{data_storage_path}/{self.miaresult_full.id}.json"
        self.miaresult_new = MIAResult.load(data_path)
        assert np.allclose(self.miaresult_new.fpr, self.fpr_array)
        assert np.allclose(self.miaresult_new.tpr, self.tpr_array)

    def test_latex(self:Self, mocker: MockerFixture) -> None:
        """Test if the LaTeX content is generated correctly."""

        result = [mocker.Mock(id="attack-config-1", result_name="test_attack_1",\
                     fixed_fpr_table={"TPR@10%FPR": 0.90, "TPR@1%FPR": 0.80, "TPR@0.1%FPR": 0.70, "TPR@0%FPR": 0.60},
                     config={"training_data_fraction": 0.5, "num_shadow_models": 3, "online": True})]

        name = "attack_comparison"

        latex_content = MIAResult._latex(result, save_dir=self.temp_dir.name, section_title=name)

        # Check that the subsection is correctly included
        assert "\\subsection{attack comparison}" in latex_content

        # Check that the figure is correctly included
        assert f"\\includegraphics[width=0.8\\textwidth]{{{self.temp_dir.name}/ROC.png}}" in latex_content

        # Check that the table header is correct
        assert "Attack name & attack config & TPR: 10.0\\%FPR & 1.0\\%FPR & 0.1\\%FPR & 0.0\\%FPR" in latex_content

        # Check if the results for mock_result are included correctly
        assert "test-attack-1" in latex_content
        assert "0.9" in latex_content
        assert "0.8" in latex_content
        assert "0.7" in latex_content
        assert "0.6" in latex_content

        # Ensure the LaTeX content ends properly
        assert "\\newline\n"  in latex_content