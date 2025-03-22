"""Tests for the attack_result module."""

import json
import os
import tempfile
import pytest
from pytest_mock import MockerFixture

import numpy as np
import torch
import torchvision
from torch import Tensor, as_tensor, randperm
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
        
from leakpro.attacks.gia_attacks.utils import InvertingConfig, invertingconfigdictmap
from leakpro.fl_utils.data_utils import get_meanstd
from leakpro.fl_utils.gia_optimizers import MetaSGD, MetaAdam
from leakpro.metrics.attack_result import MIAResult, GIAResults, get_config_name, get_gia_config_name
from leakpro.utils.import_helper import Self


class TestMIAResult:
    """Test class for MIAResult."""

    def setup_method(self:Self) -> None:
        """Set up temporary directory and logger for MIAResult."""
        self.temp_dir = tempfile.TemporaryDirectory()

        predicted_labels = np.array([[False, False, False, False, False, False],
                                    [False, False, True, False, False, False],
                                    [False, False,  True, True, False, False],
                                    [False,  True,  True, True,  False, False],
                                    [ False,  True,  True,  True,  True,  False],
                                    [ False,  True,  True,  True,  True,  False],
                                    [ True,  True,  True,  True,  True,  False],
                                    [ True,  True,  True,  True,  True,  False],
                                    [ True,  True,  True,  True,  True,  True],
                                    [ True,  True,  True,  True,  True,  True]])
        true_labels = np.array([False,  True,  True, True,  False, False])
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
        
        self.fpr_array = np.array([0., 0., 0., 0., 0.33333333, 0.33333333, 0.66666667, 0.66666667, 1., 1.])
        self.tpr_array = np.array([0., 0.33333333, 0.66666667, 1., 1., 1., 1., 1., 1., 1.])

    def teardown_method(self:Self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_miaresult_init(self:Self) -> None:
        """Test the initialization of MIAResult."""
        assert self.miaresult.id is None

    def test_check_tpr_fpr(self:Self) -> None:
        """Test fpr and tpr."""

        assert np.allclose(self.miaresult.fpr, self.fpr_array)
        assert np.allclose(self.miaresult.tpr, self.tpr_array)
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

        # Verify that the loaded instance contains the expected TPR and FPR values
        assert np.allclose(self.miaresult_new.fpr, self.fpr_array)
        assert np.allclose(self.miaresult_new.tpr, self.tpr_array)

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
        
        configs = InvertingConfig()
        configs.at_iterations = 24000
        configs.tv_reg = 1.0e-06
        configs.attack_lr = 0.1
        configs.criterion = BCEWithLogitsLoss()
        configs.epochs = 1
        self.config = configs

        def get_cifar10_loader(num_images:int =1, batch_size:int = 1, num_workers:int = 2 ) -> tuple[DataLoader, Tensor, Tensor]:
            """Get the full dataset for CIFAR10."""
            trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
            data_mean, data_std = get_meanstd(trainset)
            transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(data_mean, data_std)])
            trainset.transform = transform

            total_examples = len(trainset)
            random_indices = randperm(total_examples)[:num_images]
            subset_trainset = Subset(trainset, random_indices)
            trainloader = DataLoader(subset_trainset, batch_size=batch_size,
                                                    shuffle=False, drop_last=True, num_workers=num_workers)
            data_mean = as_tensor(data_mean)[:, None, None]
            data_std = as_tensor(data_std)[:, None, None]
            return trainloader, data_mean, data_std
        
        client_dataloader, _, _ = get_cifar10_loader(num_images=1, batch_size=1, num_workers=2)
        self.GIAresult = GIAResults(
            original_data = client_dataloader, 
            recreated_data = client_dataloader, 
            psnr_score = 12.0, 
            ssim_score = 0.84, 
            data_mean = torch.tensor(2.0), 
            data_std = torch.tensor(1.0), 
            config = self.config)

    def teardown_method(self:Self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_giaresult_init(self:Self) -> None:
        """Test the initialization of GIAResults."""
        
        assert isinstance(self.GIAresult, GIAResults)
        assert isinstance(self.GIAresult.original_data, DataLoader)
        assert isinstance(self.GIAresult.recreated_data, DataLoader)
        assert self.GIAresult.PSNR_score == 12.0
        assert self.GIAresult.SSIM_score == 0.84
        assert torch.isclose(self.GIAresult.data_mean, torch.tensor(2.0))
        assert torch.isclose(self.GIAresult.data_std, torch.tensor(1.0))
        assert isinstance(self.GIAresult.config, InvertingConfig)
        assert isinstance(self.GIAresult.config.criterion, BCEWithLogitsLoss)

    def test_save_load_giaresult(self:Self) -> None:
        """Test load and save functionality."""

        name = "gia"
        config_name = get_gia_config_name(self.config)
        save_path = f"{self.temp_dir}/{name}/{name}{config_name}"

        # Test saving
        self.GIAresult.save(self.temp_dir, name, self.config)

        assert os.path.isdir(save_path)
        assert os.path.exists(f"{save_path}/data.json")
        assert os.path.exists(f"{save_path}/original_image.png")
        assert os.path.exists(f"{save_path}/recreated_image.png")

        # Test loading
        with open(f"{save_path}/data.json") as f:
            data = json.load(f)

        self.giaresult_new = GIAResults()
        assert self.giaresult_new.original_data is None
        assert self.giaresult_new.recreated_data is None
        assert self.giaresult_new.PSNR_score  is None
        assert self.giaresult_new.SSIM_score is None
        assert self.giaresult_new.data_mean is None
        assert self.giaresult_new.data_std is None
        assert self.giaresult_new.config is None

        self.giaresult_new = GIAResults.load(data)

        # Compare all fields after loading
        assert isinstance(self.giaresult_new.original, str)
        assert isinstance(self.giaresult_new.recreated, str)
        assert self.giaresult_new.PSNR_score == self.GIAresult.PSNR_score
        assert self.giaresult_new.SSIM_score == self.GIAresult.SSIM_score
        assert self.giaresult_new.data_mean == self.GIAresult.data_mean 
        assert self.giaresult_new.data_std == self.GIAresult.data_std
        assert isinstance(self.giaresult_new.config, InvertingConfig)

        # Compare all InvertingConfig fields after loading
        assert self.giaresult_new.config.tv_reg == self.GIAresult.config.tv_reg
        assert self.giaresult_new.config.attack_lr == self.GIAresult.config.attack_lr
        assert self.giaresult_new.config.at_iterations == self.GIAresult.config.at_iterations
        assert isinstance(self.giaresult_new.config.optimizer, type(self.GIAresult.config.optimizer))
        assert isinstance(self.giaresult_new.config.criterion, type(self.GIAresult.config.criterion))
        assert self.giaresult_new.config.epochs == self.GIAresult.config.epochs
        assert self.giaresult_new.config.median_pooling == self.GIAresult.config.median_pooling
        assert self.giaresult_new.config.top10norms == self.GIAresult.config.top10norms

    def test_latex(self:Self, mocker: MockerFixture) -> None:
        """Test if the LaTeX content is generated correctly."""

        # Set name
        name = "gia"

        # Run save function
        config_name = get_gia_config_name(self.config)
        save_path = f"{self.temp_dir}/{name}/{name}{config_name}"
        self.GIAresult.save(self.temp_dir, name)

        # Load data
        with open(f"{save_path}/data.json") as f:
            data = json.load(f)
        self.giaresult = GIAResults.load(data)

        # Create results and examine latex text
        latex_content = GIAResults.create_results([self.giaresult], save_dir=self.temp_dir, save_name=name)

        # Check that the subsection is correctly included
        assert f"\\subsection{{{name}}}" in latex_content

        # Check that the original figure is correctly included
        assert "Original" in latex_content

        # Check that the figure is recreated included
        assert "Recreated" in latex_content

        # Check that the config is included
        assert config_name in latex_content

class TestIntermediateAndMergedGIAResult:
    """Test class for intermediate and merged GIAResults."""

    def setup_method(self:Self) -> None:
        """Set up temporary directory and logger for GIAResult."""
        self.temp_dir = tempfile.TemporaryDirectory()
        
        configs = InvertingConfig()
        configs.at_iterations = 24000
        configs.tv_reg = 1.0e-06
        configs.attack_lr = 0.1
        configs.criterion = BCEWithLogitsLoss() 
        configs.epochs = 1
        self.config = configs

        def get_cifar10_loader(num_images:int =1, batch_size:int = 1, num_workers:int = 2 ) -> tuple[DataLoader, Tensor, Tensor]:
            """Get the full dataset for CIFAR10."""
            trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
            data_mean, data_std = get_meanstd(trainset)
            transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(data_mean, data_std)])
            trainset.transform = transform

            total_examples = len(trainset)
            random_indices = randperm(total_examples)[:num_images]
            subset_trainset = Subset(trainset, random_indices)
            trainloader = DataLoader(subset_trainset, batch_size=batch_size,
                                                    shuffle=False, drop_last=True, num_workers=num_workers)
            data_mean = as_tensor(data_mean)[:, None, None]
            data_std = as_tensor(data_std)[:, None, None]
            return trainloader, data_mean, data_std
        
        client_dataloader, _, _ = get_cifar10_loader(num_images=1, batch_size=1, num_workers=2)

        self.GIAresult = GIAResults(
            original_data = client_dataloader, 
            recreated_data = client_dataloader, 
            psnr_score = 12.0, 
            ssim_score = 0.84, 
            data_mean = torch.tensor(2.0), 
            data_std = torch.tensor(1.0), 
            config = self.config
            )

        self.GIAresult_multiple = GIAResults(
            original_data = client_dataloader, 
            recreated_data = [client_dataloader, client_dataloader, client_dataloader, client_dataloader],
            psnr_score = [12.8, 12.8, 12.8, 12.8], 
            ssim_score = [0.84, 0.84, 0.84, 0.84],
            data_mean = torch.tensor(2.0), 
            data_std = torch.tensor(1.0), 
            config = self.config 
            )

    def teardown_method(self:Self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_merged_giaresult_init(self:Self) -> None:
        """Test the initialization of a merged GIAResult."""

        assert isinstance(self.GIAresult_multiple, GIAResults)
        assert isinstance(self.GIAresult_multiple.original_data, DataLoader)
        assert isinstance(self.GIAresult_multiple.recreated_data, list)
        assert isinstance(self.GIAresult_multiple.recreated_data[0], DataLoader)
        assert isinstance(self.GIAresult_multiple.config, InvertingConfig)
        assert isinstance(self.GIAresult_multiple.config.criterion, BCEWithLogitsLoss)

    def test_save_load_merged_giaresults(self:Self) -> None:
        """Test load and save functionality."""

        name = "gia"
        config_name = get_gia_config_name(self.config)
        save_path = f"{self.temp_dir}/{name}/{name}{config_name}"

        # Test saving
        self.GIAresult_multiple.save(self.temp_dir, name, self.config)

        assert os.path.isdir(save_path)
        assert os.path.exists(f"{save_path}/data.json")
        assert os.path.exists(f"{save_path}/original_image.png")
        assert os.path.exists(f"{save_path}/recreated_image_0.png")
        assert os.path.exists(f"{save_path}/recreated_image_1.png")
        assert os.path.exists(f"{save_path}/recreated_image_2.png")
        assert os.path.exists(f"{save_path}/recreated_image_3.png")

        # Test loading
        with open(f"{save_path}/data.json") as f:
            data = json.load(f)

        self.GIAresult_multiple_new = GIAResults()
        assert self.GIAresult_multiple_new.original_data is None
        assert self.GIAresult_multiple_new.recreated_data is None
        assert self.GIAresult_multiple_new.PSNR_score  is None
        assert self.GIAresult_multiple_new.SSIM_score is None
        assert self.GIAresult_multiple_new.data_mean is None
        assert self.GIAresult_multiple_new.data_std is None
        assert self.GIAresult_multiple_new.config is None

        self.GIAresult_multiple_new = GIAResults.load(data)

        # Compare all fields after loading
        assert isinstance(self.GIAresult_multiple_new.original, str)
        assert isinstance(self.GIAresult_multiple_new.recreated, list)
        assert self.GIAresult_multiple_new.PSNR_score == self.GIAresult_multiple.PSNR_score
        assert self.GIAresult_multiple_new.SSIM_score == self.GIAresult_multiple.SSIM_score
        assert self.GIAresult_multiple_new.data_mean == self.GIAresult_multiple.data_mean 
        assert self.GIAresult_multiple_new.data_std == self.GIAresult_multiple.data_std
        assert isinstance(self.GIAresult_multiple_new.config, InvertingConfig)

        # Compare all InvertingConfig fields after loading
        assert self.GIAresult_multiple_new.config.tv_reg == self.GIAresult_multiple.config.tv_reg
        assert self.GIAresult_multiple_new.config.attack_lr == self.GIAresult_multiple.config.attack_lr
        assert self.GIAresult_multiple_new.config.at_iterations == self.GIAresult_multiple.config.at_iterations
        assert isinstance(self.GIAresult_multiple_new.config.optimizer, type(self.GIAresult_multiple.config.optimizer))
        assert isinstance(self.GIAresult_multiple_new.config.criterion, type(self.GIAresult_multiple.config.criterion))
        assert self.GIAresult_multiple_new.config.epochs == self.GIAresult_multiple.config.epochs
        assert self.GIAresult_multiple_new.config.median_pooling == self.GIAresult_multiple.config.median_pooling
        assert self.GIAresult_multiple_new.config.top10norms == self.GIAresult_multiple.config.top10norms

    def test_merge_intermediate_giaresults(self: Self) -> None:
        """Test merging multiple GIAResults into a single result with intermediate values."""
        
        # Create mock intermediate results
        result1 = GIAResults(
            original_data=self.GIAresult.original_data,
            recreated_data=self.GIAresult.recreated_data, 
            psnr_score=torch.tensor(10.0),
            ssim_score=0.5,
            data_mean=self.GIAresult.data_mean,
            data_std=self.GIAresult.data_std,
            config=self.config
        )
        
        result2 = GIAResults(
            original_data=self.GIAresult.original_data,
            recreated_data=self.GIAresult.recreated_data,
            psnr_score=torch.tensor(11.0), 
            ssim_score=0.6,
            data_mean=self.GIAresult.data_mean,
            data_std=self.GIAresult.data_std,
            config=self.config
        )
        
        result3 = GIAResults(
            original_data=self.GIAresult.original_data, 
            recreated_data=self.GIAresult.recreated_data,
            psnr_score=torch.tensor(12.0),
            ssim_score=0.7,
            data_mean=self.GIAresult.data_mean,
            data_std=self.GIAresult.data_std, 
            config=self.config
        )

        # Create list of intermediate results
        intermediate_results = [result1, result2, result3]

        # Merge results
        merged = GIAResults.merge_intermediate(intermediate_results)

        # Verify merged result contains all intermediate values
        assert len(merged.recreated_data) == 3
        assert len(merged.PSNR_score) == 3
        assert len(merged.SSIM_score) == 3
        
        assert torch.allclose(merged.PSNR_score[0], torch.tensor(10.0))
        assert torch.allclose(merged.PSNR_score[1], torch.tensor(11.0)) 
        assert torch.allclose(merged.PSNR_score[2], torch.tensor(12.0))
        
        assert merged.SSIM_score[0] == 0.5
        assert merged.SSIM_score[1] == 0.6
        assert merged.SSIM_score[2] == 0.7

    def test_collect_generator(self: Self) -> None:
        """Test collecting results from a generator of GIAResults."""
        
        # Create mock generator that yields iteration number, tensor and result
        def mock_generator():
            for i in range(3):
                yield i, torch.randn(1), GIAResults(
                    original_data=self.GIAresult.original_data,
                    recreated_data=self.GIAresult.recreated_data,
                    psnr_score=torch.tensor(10.0 + i),
                    ssim_score=0.5 + i/10,
                    data_mean=self.GIAresult.data_mean,
                    data_std=self.GIAresult.data_std,
                    config=self.config
                )

        # Collect results from generator
        collected = GIAResults.collect_generator(mock_generator())
        
        # Verify collected contains results from all iterations
        assert len(collected.recreated_data) == 3
        assert len(collected.PSNR_score) == 3
        assert len(collected.SSIM_score) == 3
        
        # Verify values are collected in order
        assert torch.allclose(collected.PSNR_score[0], torch.tensor(10.0))
        assert torch.allclose(collected.PSNR_score[1], torch.tensor(11.0))
        assert torch.allclose(collected.PSNR_score[2], torch.tensor(12.0))
        
        assert collected.SSIM_score[0] == 0.5
        assert collected.SSIM_score[1] == 0.6  
        assert collected.SSIM_score[2] == 0.7

        
    def test_latex(self:Self, mocker: MockerFixture) -> None:
        """Test if the LaTeX content is generated correctly."""

        # Set name
        name = "gia"

        # Run save function
        config_name = get_gia_config_name(self.config)
        save_path = f"{self.temp_dir}/{name}/{name}{config_name}"
        self.GIAresult.save(self.temp_dir, name)

        # Load data
        with open(f"{save_path}/data.json") as f:
            data = json.load(f)
        self.giaresult = GIAResults.load(data)

        # Create results and examine latex text
        latex_content = GIAResults.create_results([self.giaresult], save_dir=self.temp_dir, save_name=name)

        # Check that the subsection is correctly included
        assert f"\\subsection{{{name}}}" in latex_content

        # Check that the original figure is correctly included
        assert "Original" in latex_content

        # Check that the figure is recreated included
        assert "Recreated" in latex_content

        # Check that the config is included
        assert config_name in latex_content

def test_get_gia_config_name():
    pass

def get_gia_config():
    pass

def test_invertingconfigdictmap():
    """Test invertingconfigdictmap functionality."""

    # Test mapping with custom values
    custom_config_dict = {
        'tv_reg': 1.0e-05,
        'attack_lr': 0.2,
        'at_iterations': 10000,
        'epochs': 2,
        'median_pooling': True,
        'top10norms': True,
        'optimizer': 'MetaAdam',
        'criterion': 'BCEWithLogitsLoss'
    }

    # Run mapping
    custom_config = invertingconfigdictmap(custom_config_dict)
    
    # Check if values are correctly mapped
    assert isinstance(custom_config, InvertingConfig)
    assert custom_config.tv_reg == 1.0e-05
    assert custom_config.attack_lr == 0.2
    assert custom_config.at_iterations == 10000
    assert custom_config.epochs == 2
    assert custom_config.median_pooling == True
    assert custom_config.top10norms == True
    assert isinstance(custom_config.optimizer, MetaAdam)
    assert isinstance(custom_config.criterion, BCEWithLogitsLoss)
    
    # Test with partial dict
    partial_config_dict = {
        'tv_reg': 2.0e-06,
        'attack_lr': 0.3
    }
    
    partial_config = invertingconfigdictmap(partial_config_dict)
    assert isinstance(partial_config, InvertingConfig)
    assert partial_config.tv_reg == 2.0e-06 
    assert partial_config.attack_lr == 0.3
    assert partial_config.at_iterations == 8000  # Default value
    assert partial_config.epochs == 1  # Default value
    assert partial_config.median_pooling == False  # Default value
    assert partial_config.top10norms == False  # Default value

    # Test with None dict (should raise ValueError)
    with pytest.raises(ValueError, match="Config must be a dictionary"):
        invertingconfigdictmap(None)

    # Test with empty dict (should raise ValueError)
    with pytest.raises(ValueError, match="Config dictionary cannot be empty"):
        _ = invertingconfigdictmap({})

    # Test with invalid optimizer string
    invalid_optimizer_dict = {
        'optimizer': 'InvalidOptimizer'
    }
    with pytest.raises(KeyError):
        invertingconfigdictmap(invalid_optimizer_dict)
        
    # Test with invalid criterion string 
    invalid_criterion_dict = {
        'criterion': 'InvalidCriterion'
    }
    with pytest.raises(KeyError):
        invertingconfigdictmap(invalid_criterion_dict)

    # Test with invalid attribute name
    invalid_attr_dict = {
        'invalid_attribute': 'some_value'
    }
    with pytest.raises(AttributeError):
        config = invertingconfigdictmap(invalid_attr_dict)
    
    # Test that all default values remain unchanged for attributes not in dict
    minimal_dict = {'tv_reg': 2.0e-05}
    config = invertingconfigdictmap(minimal_dict)
    assert config.tv_reg == 2.0e-05
    assert config.attack_lr == 0.1  # default
    assert config.at_iterations == 8000  # default 
    assert config.epochs == 1  # default
    assert config.median_pooling == False  # default
    assert config.top10norms == False  # default
    assert isinstance(config.optimizer, MetaSGD)  # default
    assert isinstance(config.criterion, CrossEntropyLoss)  # default