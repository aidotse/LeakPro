"""Tests for the attack_result module."""

import pickle
import os
import tempfile
from pytest_mock import MockerFixture

import numpy as np

from leakpro.reporting.mia_result import MIAResult
from leakpro.reporting.report_utils import get_config_name
from leakpro.utils.import_helper import Self
from leakpro.schemas import AuditConfig

class TestMIAResult:
    """Test class for MIAResult."""

    def setup_method(self:Self) -> None:
        """Set up temporary directory and logger for MIAResult."""
        self.temp_dir = tempfile.TemporaryDirectory()

        true_labels = np.array([False,  True,  True, True,  False, False])
        signal_values =  np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        result_name = "dummy"
        id = "dummy-test"

        self.miaresult = MIAResult(true_membership = true_labels,
                                    signal_values = signal_values,
                                    result_name = result_name,
                                    id = id)

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

    def teardown_method(self:Self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_miaresult_init(self:Self) -> None:
        """Test the initialization of MIAResult."""
        assert self.miaresult.id is "dummy-test"

    def test_check_tpr_fpr(self:Self) -> None:
        """Test fpr and tpr."""

        assert np.allclose(self.miaresult.fpr, self.fpr_array)
        assert np.allclose(self.miaresult.tpr, self.tpr_array)
        np.testing.assert_array_equal(self.miaresult.fp, self.fp)
        np.testing.assert_array_equal(self.miaresult.tn, self.tn)

    def test_save_load_miaresult(self:Self) -> None:
        """Test load and save functionality."""

        name = "lira"
        config_name = get_config_name(self.config["attack_list"][name])
        save_path = f"{self.temp_dir}/{name}/{name}{config_name}"

        # Test saving
        config = {}
        config["random_seed"] = 1234
        config["attack_list"] = self.config["attack_list"]
        config["attack_type"] = "mia"
        config["data_modality"] = "tabular"
        config["output_dir"] = save_path
        config_schema = AuditConfig(**config)

        self.miaresult.save(self.temp_dir, name, config_schema)

        assert os.path.isdir(save_path)
        assert os.path.exists(f"{save_path}/data.json")
        assert os.path.exists(f"{save_path}/{name}.png")
        assert os.path.exists(f"{save_path}/SignalHistogram.png")

        # Test loading
        with open(f"{save_path}/data.json", "rb") as f:
            data = pickle.load(f)

        self.miaresult_new = MIAResult.load(data)
        assert np.allclose(self.miaresult_new.fpr, self.fpr_array)
        assert np.allclose(self.miaresult_new.tpr, self.tpr_array)

    def test_get_strongest_miaresult(self:Self, mocker: MockerFixture) -> None:
        """Test selecting the strongest attack based on ROC AUC."""
        result_1 = mocker.Mock(roc_auc=0.75)
        result_2 = mocker.Mock(roc_auc=0.85)
        result_3 = mocker.Mock(roc_auc=0.65)

        strongest = MIAResult.get_strongest([result_1, result_2, result_3])
        print(strongest)
        # The strongest attack should be the one with the highest ROC AUC
        assert strongest == result_2

    def test_latex(self:Self, mocker: MockerFixture) -> None:
        """Test if the LaTeX content is generated correctly."""

        result = [mocker.Mock(id="attack-config-1", result_name="test_attack_1",\
                     fixed_fpr_table={"TPR@10.0%FPR": 0.90, "TPR@1.0%FPR": 0.80, "TPR@0.1%FPR": 0.70, "TPR@0.0%FPR": 0.60},
                     config={"training_data_fraction": 0.5, "num_shadow_models": 3, "online": True})]

        name = "attack_comparison"

        latex_content = MIAResult._latex(result, save_dir=self.temp_dir, save_name=name)

        # Check that the subsection is correctly included
        assert "\\subsection{attack comparison}" in latex_content

        # Check that the figure is correctly included
        assert f"\\includegraphics[width=0.8\\textwidth]{{{name}.png}}" in latex_content

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

    def test_get_all_attacknames(self:Self, mocker: MockerFixture) -> None:
        """Test retrieval of all attack names."""
        result_mock_1 = mocker.Mock(result_name="Attack1")
        result_mock_2 = mocker.Mock(result_name="Attack2")
        results = [result_mock_1, result_mock_2, result_mock_1]

        attack_names = MIAResult._get_all_attacknames(results)

        assert attack_names == ["Attack1", "Attack2"]

    def test_get_results_of_name(self:Self, mocker: MockerFixture) -> None:
        """Test retrieval of all attack names."""
        result_mock_1 = mocker.Mock(result_name="Attack1")
        result_mock_2 = mocker.Mock(result_name="Attack2")
        result_mock_3 = mocker.Mock(result_name="Attack2")
        result_mock_4 = mocker.Mock(result_name="Attack3")
        result_mock_5 = mocker.Mock(result_name="Attack3")
        result_mock_6 = mocker.Mock(result_name="Attack3")

        results = [result_mock_1, result_mock_2, result_mock_3,
                                       result_mock_4, result_mock_5, result_mock_6]

        assert len(MIAResult._get_results_of_name(results, "Attack1")) == 1
        assert len(MIAResult._get_results_of_name(results, "Attack2")) == 2
        assert len(MIAResult._get_results_of_name(results, "Attack3")) == 3
