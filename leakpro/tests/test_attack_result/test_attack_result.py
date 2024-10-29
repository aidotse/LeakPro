import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

from leakpro.metrics.attack_result import *
from leakpro.utils.import_helper import Self


class TestMIAResult(unittest.TestCase):

    def setUp(self:Self) -> None:
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

    def tearDown(self:Self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_MIAResult_init(self:Self) -> None:
        """Test the initialization of MIAResult."""
        assert self.miaresult.id == None

    def test_check_tpr_fpr(self:Self) -> None:
        assert np.allclose(self.miaresult.tpr, np.array([0., 0., 0.16666667, 0.5, 1., 1., 1., 1., 1., 1.]))
        assert self.miaresult.fp.all() == 0.
        assert self.miaresult.tn.all() == 0.

    def test_save_load_MIAResult(self:Self) -> None:

        name = "lira"
        config_name = get_config_name(self.config["attack_list"][name])
        save_path = f"{self.temp_dir}/{name}/{name}{config_name}"

        # Test saving
        self.miaresult.save(self.temp_dir, name, self.config)

        assert os.path.isdir(save_path)
        assert os.path.exists(f"{save_path}/data.json")
        assert os.path.exists(f"{save_path}/ROC.png")
        assert os.path.exists(f"{save_path}/SignalHistogram.png")

        # Test loading
        with open(f"{save_path}/data.json") as f:
            data = json.load(f)

        self.miaresult_new = MIAResult(load=True)
        assert self.miaresult_new.predicted_labels == None
        assert self.miaresult_new.true_labels == None
        assert self.miaresult_new.signal_values == None

        self.miaresult_new.load(data)
        assert np.allclose(self.miaresult_new.tpr, np.array([0., 0., 0.16666667, 0.5, 1., 1., 1., 1., 1., 1.]))

    def test_get_strongest_MIAResult(self:Self) -> None:
        """Test selecting the strongest attack based on ROC AUC."""
        result_1 = MagicMock(roc_auc=0.75)
        result_2 = MagicMock(roc_auc=0.85)
        result_3 = MagicMock(roc_auc=0.65)

        mia_result = MIAResult(load=True)
        strongest = mia_result.get_strongest([result_1, result_2, result_3])

        # The strongest attack should be the one with the highest ROC AUC
        assert strongest == result_2

    def test_latex(self:Self) -> None:
        """Test if the LaTeX content is generated correctly."""

        result = [MagicMock(id="attack-config-1", resultname="test_attack_1", fixed_fpr_table={"TPR@1.0%FPR": 0.90, "TPR@0.1%FPR": 0.80, "TPR@0.01%FPR": 0.70, "TPR@0.0%FPR": 0.60})]
        subsection = "attack_comparison"
        filename = f"{self.temp_dir}/test.png"

        latex_content = MIAResult(load=True)._latex(result, subsection, filename)

        # Check that the subsection is correctly included
        self.assertIn("\\subsection{attack comparison}", latex_content)

        # Check that the figure is correctly included
        self.assertIn(f"\\includegraphics[width=0.8\\textwidth]{{{filename}.png}}", latex_content)

        # Check that the table header is correct
        self.assertIn("Attack name & attack config & TPR: 1.0\\%FPR & 0.1\\%FPR & 0.01\\%FPR & 0.0\\%FPR", latex_content)

        # Check if the results for mock_result are included correctly
        self.assertIn("test-attack-1", latex_content)
        self.assertIn("0.9", latex_content)
        self.assertIn("0.8", latex_content)
        self.assertIn("0.7", latex_content)
        self.assertIn("0.6", latex_content)

        # Ensure the LaTeX content ends properly
        self.assertIn("\\newline\n", latex_content)
