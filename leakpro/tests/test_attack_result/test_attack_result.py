"""Tests for the attack_result module."""

import json
import os
import tempfile
from pytest_mock import MockerFixture

import numpy as np

from leakpro.metrics.attack_result import MIAResult, GIAResults, get_config_name
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
        """Test the true positive rate (TPR) and false positive rate (FPR) of MIAresult.

        This function verifies:
        - The `tpr` (true positive rate) values are close to the expected array.
        - The `fp` (false positives) values are all zeros.
        - The `tn` (true negatives) values are all zeros.
        """

        # Check if the calculated TPR values match the expected ones (within numerical tolerance)
        assert np.allclose(
            self.miaresult.tpr, 
            np.array([0., 0., 0.16666667, 0.5, 1., 1., 1., 1., 1., 1.])
        )

        # Ensure all false positive values are exactly zero
        assert self.miaresult.fp.all() == 0.

        # Ensure all true negative values are exactly zero
        assert self.miaresult.tn.all() == 0.

    def test_save_load_miaresult(self: Self) -> None:
        """Test the save and load functionality of MIAResult.

        This function verifies:
        - The `MIAResult.save` method correctly creates the expected directory and files.
        - The `MIAResult.load` method correctly restores a saved instance.
        """

        # Define the attack name
        name = "lira"

        # Generate the configuration name based on the attack configuration
        config_name = get_config_name(self.config["attack_list"][name])

        # Construct the expected save path
        save_path = f"{self.temp_dir}/{name}/{name}{config_name}"

        # ---- Test Saving ----
        # Save the MIAResult instance
        self.miaresult.save(self.temp_dir, name, self.config)

        # Verify that the save directory exists
        assert os.path.isdir(save_path), "Save directory was not created."

        # Verify that expected files exist in the save path
        assert os.path.exists(f"{save_path}/data.json"), "data.json file missing."
        assert os.path.exists(f"{save_path}/{name}.png"), f"{name}.png file missing."
        assert os.path.exists(f"{save_path}/SignalHistogram.png"), "SignalHistogram.png file missing."

        # ---- Test Loading ----
        # Load the saved data from JSON file
        with open(f"{save_path}/data.json") as f:
            data = json.load(f)

        # Create a new MIAResult instance
        self.miaresult_new = MIAResult()

        # Ensure the new instance has no data before loading
        assert self.miaresult_new.predicted_labels is None, "predicted_labels should be None before loading."
        assert self.miaresult_new.true_labels is None, "true_labels should be None before loading."
        assert self.miaresult_new.signal_values is None, "signal_values should be None before loading."

        # Load the data into the new instance
        self.miaresult_new = MIAResult.load(data)

        # Verify that the loaded instance contains the expected TPR values
        expected_tpr = np.array([0., 0., 0.16666667, 0.5, 1., 1., 1., 1., 1., 1.])
        assert np.allclose(self.miaresult_new.tpr, expected_tpr), "Loaded TPR values do not match expected values."

    def test_get_strongest_miaresult(self: Self, mocker: MockerFixture) -> None:
        """Test selecting the strongest attack based on ROC AUC.

        This function verifies that the `get_strongest` method correctly selects 
        the attack with the highest ROC AUC score.
        """

        # Create mock MIAResult objects with different ROC AUC values
        result_1 = mocker.Mock(roc_auc=0.75)
        result_2 = mocker.Mock(roc_auc=0.85)  # This one has the highest ROC AUC
        result_3 = mocker.Mock(roc_auc=0.65)

        # Create an instance of MIAResult
        mia_result = MIAResult()

        # Call the method to get the strongest attack
        strongest = mia_result.get_strongest([result_1, result_2, result_3])

        # The strongest attack should be the one with the highest ROC AUC
        assert strongest == result_2, "The strongest attack was not correctly selected based on ROC AUC."


    def test_latex(self: Self, mocker: MockerFixture) -> None:
        """Test if the LaTeX content is generated correctly."""

        # Mock an attack result with necessary attributes
        result = [mocker.Mock(
            id="attack-config-1",
            resultname="test_attack_1",
            fixed_fpr_table={
                "TPR@1.0%FPR": 0.90,
                "TPR@0.1%FPR": 0.80,
                "TPR@0.01%FPR": 0.70,
                "TPR@0.0%FPR": 0.60
            },
            config={
                "training_data_fraction": 0.5,
                "num_shadow_models": 3,
                "online": True
            }
        )]

        # Define the attack comparison name
        name = "attack_comparison"

        # Generate LaTeX content
        latex_content = MIAResult()._latex(result, save_dir=self.temp_dir, save_name=name)

        # ---- Assertions to verify correct LaTeX content ----

        # Check if subsection header is included
        assert "\\subsection{attack comparison}" in latex_content, "Subsection header is missing in LaTeX content."

        # Check if the figure reference is included
        assert f"\\includegraphics[width=0.8\\textwidth]{{{name}.png}}" in latex_content, "Figure inclusion is incorrect."

        # Check if the LaTeX table header is properly formatted
        assert "Attack name & attack config & TPR: 1.0\\%FPR & 0.1\\%FPR & 0.01\\%FPR & 0.0\\%FPR" in latex_content, \
            "Table header is incorrect."

        # Verify that expected attack details appear in LaTeX content
        assert "test-attack-1" in latex_content, "Attack name is missing in LaTeX content."
        assert "0.9" in latex_content, "TPR@1.0%FPR is missing in LaTeX content."
        assert "0.8" in latex_content, "TPR@0.1%FPR is missing in LaTeX content."
        assert "0.7" in latex_content, "TPR@0.01%FPR is missing in LaTeX content."
        assert "0.6" in latex_content, "TPR@0.0%FPR is missing in LaTeX content."

        # Ensure the LaTeX content ends properly with a newline
        assert "\\newline\n" in latex_content, "Final newline is missing in LaTeX content."

    def test_get_all_attacknames(self: Self, mocker: MockerFixture) -> None:
        """Test retrieval of all unique attack names."""

        # Create mock attack results with names
        result_mock_1 = mocker.Mock(resultname="Attack1")
        result_mock_2 = mocker.Mock(resultname="Attack2")

        # Include a duplicate attack name
        results = [result_mock_1, result_mock_2, result_mock_1]

        # Call the function to get unique attack names
        attack_names = MIAResult._get_all_attacknames(results)

        # Check that duplicate names are removed
        assert attack_names == ["Attack1", "Attack2"], "Unique attack names were not retrieved correctly."

    def test_get_results_of_name(self: Self, mocker: MockerFixture) -> None:
        """Test retrieval of all attack results with a specific name."""

        # Create mock attack results with different names
        result_mock_1 = mocker.Mock(resultname="Attack1")
        result_mock_2 = mocker.Mock(resultname="Attack2")
        result_mock_3 = mocker.Mock(resultname="Attack2")
        result_mock_4 = mocker.Mock(resultname="Attack3")
        result_mock_5 = mocker.Mock(resultname="Attack3")
        result_mock_6 = mocker.Mock(resultname="Attack3")

        # Store all results in a list
        results = [result_mock_1, result_mock_2, result_mock_3, result_mock_4, result_mock_5, result_mock_6]

        # Check that filtering by attack name returns the correct count
        assert len(MIAResult._get_results_of_name(results, "Attack1")) == 1, "Incorrect number of results for Attack1."
        assert len(MIAResult._get_results_of_name(results, "Attack2")) == 2, "Incorrect number of results for Attack2."
        assert len(MIAResult._get_results_of_name(results, "Attack3")) == 3, "Incorrect number of results for Attack3."


class TestGIAResult:
    """Test class for GIAResult."""

    def setup_method(self:Self) -> None:
        """Set up temporary directory and logger for GIAResult."""
        self.temp_dir = tempfile.TemporaryDirectory()
        
        import torch
        from torch.nn import BCEWithLogitsLoss
        from leakpro.attacks.gia_attacks.invertinggradients import InvertingConfig
        
        configs = InvertingConfig()
        configs.at_iterations = 24000
        configs.tv_reg = 1.0e-06
        configs.attack_lr = 0.1
        configs.criterion = BCEWithLogitsLoss() #CrossEntropyLoss()
        configs.epochs = 1
        self.config = configs
        
        org_data = torch.tensor([2.0271e+00, 2.0112e+00, 2.0271e+00, 1.0625e-01, -4.8768e-03, -3.2237e-01])
        self.GIAresult = GIAResults(org_data,
                org_data*2,
                torch.tensor(12.8943),
                0.8470116640585275,
                torch.tensor([0.4914]),
                torch.tensor([0.2470]),
                self.config)

    def teardown_method(self:Self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_giaresult_init(self:Self) -> None:
        """Test the initialization of MIAResult."""
        assert self.GIAresult is not None

    def test_save_load_giaresult(self:Self) -> None:
        """Test load and save functionality."""

        name = "gia"
        config_name = get_config_name(self.config["attack_list"][name])
        save_path = f"{self.temp_dir}/{name}/{name}{config_name}"

        # Test saving
        self.GIAresult.save(self.temp_dir, name, self.config)

        assert os.path.isdir(save_path)
        assert os.path.exists(f"{save_path}/data.json")
        assert os.path.exists(f"{save_path}/{name}.png")
        assert os.path.exists(f"{save_path}/SignalHistogram.png")

        # Test loading
        with open(f"{save_path}/data.json") as f:
            data = json.load(f)

        self.giaresult_new = GIAResults()
        assert self.giaresult_new.original_data is None
        assert self.giaresult_new.recreated_data is None
        assert self.giaresult_new.psnr_score is None
        assert self.giaresult_new.ssim_score is None
        assert self.giaresult_new.data_mean is None
        assert self.giaresult_new.data_std is None
        assert self.giaresult_new.config is None

        self.giaresult_new = GIAResults.load(data)
        assert self.giaresult_new.result_config == self.GIAresult.config
        
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