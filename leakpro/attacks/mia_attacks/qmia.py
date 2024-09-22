"""Implementation of the RMIA attack."""

from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.metrics.attack_result import CombinedMetricResult
from leakpro.signals.signal import ModelRescaledLogits
from leakpro.utils.import_helper import Any, Self, Tuple
from leakpro.utils.logger import logger


class QMIADataset(Dataset):
    """Dataset class for the QMIA attack."""

    def __init__(self:Self, dataloader:DataLoader, new_labels:np.ndarray)->None:
        self.features = None
        self.labels = None
        for data, _  in dataloader:
            self.features = data
            self.labels = new_labels

    def __len__(self:Self) -> int:
        """Get the length of the dataset.

        Returns
        -------
            int: The length of the dataset.

        """
        return len(self.features)

    def __getitem__(self:Self, idx:int) -> Tuple[Any, Any]:
        """Get an item from the dataset.

        Args:
        ----
            idx (int): Index of the item to retrieve.

        Returns:
        -------
            Tuple[Any, Any]: The features and labels of the item.

        """
        return self.features[idx], self.labels[idx]

class QuantileRegressor(nn.Module):
    """QuantileRegressor module for predicting quantiles."""

    def __init__(self:Self, model:nn.Module, dummy_dataloader:DataLoader, n_quantiles: int = 1) -> None:
        """Initialize the QuantileRegressor.

        Args:
        ----
            model (nn.Module): The model to extract features from.
            dummy_dataloader (DataLoader): A dummy input dataloader for inferring the penultimate layer size.
            n_quantiles (int): Number of quantiles to predict.

        """
        super(QuantileRegressor, self).__init__()

        # Create a regressor from the target model
        self.feature_extractor = deepcopy(model)
        layers = list(self.feature_extractor.children())[:-1]  # Remove the last layer
        self.feature_extractor = nn.Sequential(*layers) # Rebuild the model without the last layer

        # Figure out the size of the penultimate layer
        for x, _ in dummy_dataloader:
            dummy_output = self.feature_extractor(x)
        in_features = dummy_output.view(dummy_output.size(0), -1).size(1)

        # Add the regression layer
        self.regressor = nn.Linear(in_features, n_quantiles)  # Assuming we predict n quantiles

    def forward(self:Self, x:torch.Tensor) -> torch.Tensor:
        """Forward pass of the QuantileRegressor module.

        Args:
        ----
            x (torch.Tensor): Input tensor.

        Returns:
        -------
            torch.Tensor: Output tensor.

        """
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # Flatten the features
        return self.regressor(x)

class PinballLoss(torch.nn.Module):
    """Pinball loss for quantile regression."""

    def __init__(self:Self, quantiles:list) -> None:
        """Initialize the PinballLoss.

        Args:
        ----
            quantiles (list): List of quantiles to predict

        """
        super(PinballLoss, self).__init__()
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32)

    def forward(self:Self, predictions:torch.Tensor , targets:torch.Tensor) -> torch.Tensor:
        """Forward pass of the PinballLoss module.

        Args:
        ----
            predictions (torch.Tensor): Predicted values.
            targets (torch.Tensor): Target values.

        Returns:
        -------
            torch.Tensor: Loss value.

        """
        if predictions.shape[1] != len(self.quantiles):
            raise ValueError("Number of quantiles must match the predictions dimension")

        quantiles = self.quantiles.to(predictions.device)
        errors = targets.unsqueeze(1) - predictions
        losses = torch.max((quantiles * errors), ((quantiles - 1) * errors))
        return torch.mean(torch.sum(losses, dim=1))

class AttackQMIA(AbstractMIA):
    """Implementation of the RMIA attack."""

    def __init__(
        self:Self,
        handler: AbstractInputHandler,
        configs: dict
    ) -> None:
        """Initialize the QMIA attack.

        Args:
        ----
            handler (AbstractInputHandler): The input handler object.
            configs (dict): Configuration parameters for the attack.

        """
        # Initializes the parent metric
        super().__init__(handler)

        logger.info("Configuring the QMIA attack")
        self._configure_attack(configs)

        self.signal = ModelRescaledLogits()
        dummy_index = np.atleast_1d(np.ones(1)).astype(int)
        self.quantile_regressor = QuantileRegressor(self.handler.target_model,
                                                    self.handler.get_dataloader(dummy_index),
                                                    len(self.quantiles))

    def _configure_attack(self:Self, configs:dict) -> None:
        self.training_data_fraction = configs.get("training_data_fraction", 0.5)
        self.quantiles = configs.get("quantiles", [0, 0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99995, 0.99999])
        self.epochs = configs.get("epochs", 100)

        # Define the validation dictionary as: {parameter_name: (parameter, min_value, max_value)}
        validation_dict = {
            "min_quantile": (min(self.quantiles), 0.0, max(self.quantiles)),
            "max_quantile": (max(self.quantiles), min(self.quantiles), 1.0),
            "num_quantiles": (len(self.quantiles), 1, None),
            "epochs": (self.epochs, 0, None),
            "training_data_fraction": (self.training_data_fraction, 0, 1)
        }

        # Validate parameters
        for param_name, (param_value, min_val, max_val) in validation_dict.items():
            self._validate_config(param_name, param_value, min_val, max_val)

    def _validate_config(self: Self, name: str, value: float, min_val: float, max_val: float) -> None:
        if not (min_val <= value <= (max_val if max_val is not None else value)):
            raise ValueError(f"{name} must be between {min_val} and {max_val}")

    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "QMIA attack"
        reference_str = "Bertran, Martin, et al. Scalable membership inference attacks via quantile regression. Neurips, (2024)."
        summary_str = "The QMIA attack is a membership inference attack based on quantile regression."
        detailed_str = "The attack is executed according to: \
            1. Select a target FPR and a hypothesis set H for regression .\
            2. Train a model q in H by minimizing the pinball loss. \
            3. Classify as in-members if logit of (x,y) is above q(x)."
        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }

    def prepare_attack(self:Self) -> None:
        """Prepare data needed for running the attack on the target model and dataset.

        Signals are computed on the auxiliary model(s) and dataset.
        """
        # sample dataset to train quantile regressor
        logger.info("Preparing attack data for training the quantile regressor")
        self.attack_data_indices = self.sample_indices_from_population(include_train_indices = False,
                                                                       include_test_indices = False)

        # subsample the attack data based on the fraction
        logger.info(f"Subsampling attack data from {len(self.attack_data_indices)} points")
        n_points = int(self.training_data_fraction * len(self.attack_data_indices))
        chosen_attack_data_indices = np.random.choice(self.attack_data_indices, n_points, replace=False)
        logger.info(f"Number of attack data points after subsampling: {len(chosen_attack_data_indices)}")

        # create labels and change dataset to be used for regression
        regression_labels = np.array(self.signal([self.target_model],
                                                 self.handler,
                                                 chosen_attack_data_indices)).squeeze()

        # Create custom dataset for regression
        attack_dataloader = self.get_dataloader(chosen_attack_data_indices)
        quantile_regression_dataloader = DataLoader(QMIADataset(attack_dataloader, regression_labels), batch_size=64)

        # train quantile regressor
        logger.info("Training the quantile regressor")
        optimizer = torch.optim.Adam(self.quantile_regressor.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = PinballLoss(self.quantiles)
        self._train_quantile_regressor(quantile_regression_dataloader, criterion, optimizer, self.epochs)
        logger.info("Training of quantile regressor completed")

    def _train_quantile_regressor(
        self:Self,
        attack_dataloader: DataLoader,
        criterion:torch.nn.Module,
        optimizer:torch.optim,
        epochs: int
    ) -> None:
        """Train the quantile regressor model.

        Args:
        ----
            attack_dataloader (DataLoader): DataLoader for the attack dataset.
            criterion (torch.nn.Module): Loss criterion for training.
            optimizer (torch.optim): Optimizer for training.
            epochs (int, optional): Number of training epochs. Defaults to 200.

        Returns:
        -------
            None

        """
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.quantile_regressor.to(device)
        self.quantile_regressor.train()

        # Loop over each epoch
        for epoch_idx in tqdm(range(epochs)):
            train_loss = 0.0
            # Loop over the training set
            self.quantile_regressor.train()
            for data, target in attack_dataloader:
                # Move data to the device
                data, target = data.to(device, non_blocking=True), target.to(  # noqa: PLW2901
                    device, non_blocking=True
                )
                # Cast target to long tensor
                target = target.long()  # noqa: PLW2901

                # Set the gradients to zero
                optimizer.zero_grad(set_to_none=True)

                # Get the model output
                output = self.quantile_regressor(data)
                # Calculate the loss
                loss = criterion(output, target)
                # Perform the backward pass
                loss.backward()
                # Take a step using optimizer
                optimizer.step()
                # Add the loss to the total loss
                train_loss += loss.item()

            log_train_str = f"Epoch: {epoch_idx+1}/{epochs} | Train Loss: {train_loss/len(attack_dataloader):.8f}"  # noqa: E501
            logger.info(log_train_str)

        # Move the model back to the CPU
        self.quantile_regressor.to("cpu")

    def run_attack(self:Self) -> CombinedMetricResult:
        """Run the attack on the target model and dataset.

        Args:
        ----
            fpr_tolerance_rate_list (optional): List of FPR tolerance values that may be used by the threshold function
                to compute the attack threshold for the metric.

        Returns:
        -------
            Result(s) of the metric.

        """
        self.target_logits = np.array(self.signal([self.target_model],
                                                  self.handler,
                                                  self.audit_dataset["data"])).squeeze()

        audit_dataloader = self.get_dataloader(self.audit_dataset["data"])
        logger.info("Running the attack on the target model")
        score = []
        for data, _ in audit_dataloader:
            score.extend(self.quantile_regressor(data).detach().numpy())
        score = np.array(score).T

        logger.info("Attack completed")

        # pick out the in-members and out-members signals
        self.in_member_signals = self.target_logits[self.audit_dataset["in_members"]]
        self.out_member_signals = self.target_logits[self.audit_dataset["out_members"]]

        predictions = np.less(score, self.target_logits[np.newaxis, :])

        # set true labels for being in the training dataset
        true_labels = np.concatenate(
            [
                np.ones(len(self.audit_dataset["in_members"])),
                np.zeros(len(self.audit_dataset["out_members"])),
            ]
        )
        signal_values = np.hstack(
            [self.in_member_signals, self.out_member_signals]
        )

        # compute ROC, TP, TN etc
        return CombinedMetricResult(
            predicted_labels=predictions,
            true_labels=true_labels,
            predictions_proba=None,
            signal_values=signal_values,
        )


