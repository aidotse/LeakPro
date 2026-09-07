#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
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

        # Check that the figure is correctly included (path has underscores escaped for LaTeX)
        expected_path = self.temp_dir.name.replace("_", "\\_")
        assert f"\\includegraphics[width=0.8\\textwidth]{{{expected_path}/ROC.png}}" in latex_content

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

class TestTiedScoreROC:
    """Regression tests: ROC vertices must sit at the END of each tie block.

    A threshold at value v admits every point scoring >= v, so a tie block is
    admitted whole or not at all. The old code snapshotted the block START
    (np.unique first occurrence), counting exactly one arbitrary element per
    block — an operating point no threshold can realize. Consequences: phantom
    vertices near the origin, curves that never reach (1, 1), and headline
    TPRs (e.g. TPR@0%FPR) that depended on argsort tie order. Invisible on
    all-distinct scores; bites on clamped/quantized signals such as DP-SGD
    saturation.
    """

    @staticmethod
    def _full(scores, labels) -> MIAResult:
        return MIAResult.from_full_scores(true_membership=np.asarray(labels, dtype=bool),
                                          signal_values=np.asarray(scores, dtype=float),
                                          result_name="tied")

    def test_minimal_tied_case_counts_whole_blocks(self:Self) -> None:
        """Scores [9,9,5,5], labels [1,0,1,0]: threshold 9 admits one member AND
        one nonmember (they are tied); threshold 5 admits everything."""
        result = self._full([9, 9, 5, 5], [1, 0, 1, 0])
        np.testing.assert_array_equal(result.tp, [1, 2])
        np.testing.assert_array_equal(result.fp, [1, 2])
        assert np.allclose(result.fpr, [0.5, 1.0])
        assert np.allclose(result.tpr, [0.5, 1.0])
        # The old block-start rule reported fp=[0,1], i.e. a phantom vertex at
        # FPR 0 for an attack that cannot tell members from nonmembers.

    def test_vertices_match_sklearn_on_heavy_ties(self:Self) -> None:
        """Randomized heavy-tie cases: the (fpr, tpr) vertex set must equal
        sklearn's roc_curve(..., drop_intermediate=False) minus its (0,0) anchor."""
        from sklearn.metrics import roc_curve
        rng = np.random.default_rng(0)
        for _ in range(50):
            n = int(rng.integers(20, 300))
            # Quantized scores -> plenty of ties, including at the extremes.
            scores = rng.integers(0, 6, size=n).astype(float)
            labels = rng.integers(0, 2, size=n).astype(bool)
            if labels.all() or not labels.any():
                continue
            result = self._full(scores, labels)
            fpr_sk, tpr_sk, _ = roc_curve(labels, scores, drop_intermediate=False)
            np.testing.assert_allclose(result.fpr, fpr_sk[1:], atol=1e-12)
            np.testing.assert_allclose(result.tpr, tpr_sk[1:], atol=1e-12)

    def test_result_is_invariant_to_input_order(self:Self) -> None:
        """The old rule made TPR at low FPR depend on which tied point argsort
        happened to place first. Reshuffling the inputs must not change anything."""
        rng = np.random.default_rng(1)
        n = 2000
        # Saturation shape: a shared tie block at the top score.
        scores = np.concatenate([np.full(120, 5.0), rng.random(n - 120),      # members
                                 np.full(60, 5.0), rng.random(n - 60)])        # nonmembers
        labels = np.concatenate([np.ones(n, dtype=bool), np.zeros(n, dtype=bool)])
        tables = []
        for _ in range(6):
            perm = rng.permutation(2 * n)
            tables.append(self._full(scores[perm], labels[perm]).fixed_fpr_table)
        assert all(t == tables[0] for t in tables[1:])
