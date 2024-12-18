"""Tests for the attack_result module."""

import json
import os
import tempfile
from pytest_mock import MockerFixture

import numpy as np

from leakpro.metrics.attack_result import MIAResult, get_config_name
from leakpro.utils.import_helper import Self


class TestMIAResult:
    """Test class for MIAResult."""

    def setup_method(self:Self) -> None:
        """Set up temporary directory and logger for MIAResult."""
        self.temp_dir = tempfile.TemporaryDirectory()

        predicted_labels = np.array([[False, False, False, False, False, False],
                                    [False, False, False, False, False, False],
                                    [False, False,  True, False, False, False],
                                    [False,  True,  True, False,  True, False],
                                    [ True,  True,  True,  True,  True,  True],
                                    [ True,  True,  True,  True,  True,  True],
                                    [ True,  True,  True,  True,  True,  True],
                                    [ True,  True,  True,  True,  True,  True],
                                    [ True,  True,  True,  True,  True,  True],
                                    [ True,  True,  True,  True,  True,  True]])
        true_labels = np.ones((6))
        signal_values =  np.array([[-0.00614866],
                                    [-0.45619705],
                                    [-2.30781003],
                                    [ 0.46973035],
                                    [-0.1584589 ],
                                    [ 0.14289466]])


        predictions_proba = None
        threshold = None
        audit_indices = np.array([0, 1, 2, 3])
        resultname = None
        id = None

        self.miaresult = MIAResult(predicted_labels = predicted_labels,
                                    true_labels = true_labels,
                                    signal_values = signal_values,
                                    predictions_proba = predictions_proba,
                                    threshold = threshold,
                                    audit_indices = audit_indices,
                                    resultname = resultname,
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

    def teardown_method(self:Self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_miaresult_init(self:Self) -> None:
        """Test the initialization of MIAResult."""
        assert self.miaresult.id is None

    def test_check_tpr_fpr(self:Self) -> None:
        """Test fpr and tpr."""

        assert np.allclose(self.miaresult.tpr, np.array([0., 0., 0.16666667, 0.5, 1., 1., 1., 1., 1., 1.]))
        assert self.miaresult.fp.all() == 0.
        assert self.miaresult.tn.all() == 0.

    def test_save_load_miaresult(self:Self) -> None:
        """Test load and save functionality."""

        name = "lira"
        config_name = get_config_name(self.config["attack_list"][name])
        save_path = f"{self.temp_dir}/{name}/{name}{config_name}"

        # Test saving
        self.miaresult.save(self.temp_dir, name, self.config)

        assert os.path.isdir(save_path)
        assert os.path.exists(f"{save_path}/data.json")
        assert os.path.exists(f"{save_path}/{name}.png")
        assert os.path.exists(f"{save_path}/SignalHistogram.png")

        # Test loading
        with open(f"{save_path}/data.json") as f:
            data = json.load(f)

        self.miaresult_new = MIAResult()
        assert self.miaresult_new.predicted_labels is None
        assert self.miaresult_new.true_labels is None
        assert self.miaresult_new.signal_values is None

        self.miaresult_new = MIAResult.load(data)
        assert np.allclose(self.miaresult_new.tpr, np.array([0., 0., 0.16666667, 0.5, 1., 1., 1., 1., 1., 1.]))

    def test_get_strongest_miaresult(self:Self, mocker: MockerFixture) -> None:
        """Test selecting the strongest attack based on ROC AUC."""
        result_1 = mocker.Mock(roc_auc=0.75)
        result_2 = mocker.Mock(roc_auc=0.85)
        result_3 = mocker.Mock(roc_auc=0.65)

        mia_result = MIAResult()
        strongest = mia_result.get_strongest([result_1, result_2, result_3])
        print(strongest)
        # The strongest attack should be the one with the highest ROC AUC
        assert strongest == result_2

    def test_latex(self:Self, mocker: MockerFixture) -> None:
        """Test if the LaTeX content is generated correctly."""

        result = [mocker.Mock(id="attack-config-1", resultname="test_attack_1",\
                     fixed_fpr_table={"TPR@1.0%FPR": 0.90, "TPR@0.1%FPR": 0.80, "TPR@0.01%FPR": 0.70, "TPR@0.0%FPR": 0.60},
                     config={"training_data_fraction": 0.5, "num_shadow_models": 3, "online": True})]

        name = "attack_comparison"
        filename = f"{self.temp_dir}/{name}"

        latex_content = MIAResult()._latex(result, save_dir=self.temp_dir, save_name=name)

        # Check that the subsection is correctly included
        assert "\\subsection{attack comparison}" in latex_content

        # Check that the figure is correctly included
        assert f"\\includegraphics[width=0.8\\textwidth]{{{name}.png}}" in latex_content

        # Check that the table header is correct
        assert "Attack name & attack config & TPR: 1.0\\%FPR & 0.1\\%FPR & 0.01\\%FPR & 0.0\\%FPR" in latex_content

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
        result_mock_1 = mocker.Mock(resultname="Attack1")
        result_mock_2 = mocker.Mock(resultname="Attack2")
        results = [result_mock_1, result_mock_2, result_mock_1]

        attack_names = MIAResult._get_all_attacknames(results)

        assert attack_names == ["Attack1", "Attack2"]

    def test_get_results_of_name(self:Self, mocker: MockerFixture) -> None:
        """Test retrieval of all attack names."""
        result_mock_1 = mocker.Mock(resultname="Attack1")
        result_mock_2 = mocker.Mock(resultname="Attack2")
        result_mock_3 = mocker.Mock(resultname="Attack2")
        result_mock_4 = mocker.Mock(resultname="Attack3")
        result_mock_5 = mocker.Mock(resultname="Attack3")
        result_mock_6 = mocker.Mock(resultname="Attack3")

        results = [result_mock_1, result_mock_2, result_mock_3,
                                       result_mock_4, result_mock_5, result_mock_6]

        assert len(MIAResult._get_results_of_name(results, "Attack1")) == 1
        assert len(MIAResult._get_results_of_name(results, "Attack2")) == 2
        assert len(MIAResult._get_results_of_name(results, "Attack3")) == 3
