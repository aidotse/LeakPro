"""Implementation of the RMIA attack."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.import_helper import Self
from leakpro.metrics.attack_result import CombinedMetricResult
from leakpro.signals.signal import ModelRescaledLogits
from leakpro.user_inputs.abstract_input_handler import AbstractInputHandler


class QuantileRegressor(nn.Module):
    """QuantileRegressor module for predicting quantiles."""

    def __init__(self:Self, n_quantiles: int = 1) -> None:
        """Initialize the QuantileRegressor.

        Args:
        ----
            n_quantiles (int): Number of quantiles to predict.

        """
        super(QuantileRegressor, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.regressor = nn.Linear(64 * 8 * 8, n_quantiles)  # Assuming we predict 3 quantiles

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

        self.logger.info("Configuring the QMIA attack")
        self._configure_attack(configs)

        self.signal = ModelRescaledLogits()
        self.quantile_regressor = QuantileRegressor(len(self.quantiles))

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
        self.logger.info("Preparing attack data for training the quantile regressor")
        self.attack_data_indices = self.sample_indices_from_population(include_train_indices = False,
                                                                       include_test_indices = False)

        # subsample the attack data based on the fraction
        self.logger.info(f"Subsampling attack data from {len(self.attack_data_indices)} points")
        n_points = int(self.training_data_fraction * len(self.attack_data_indices))
        chosen_attack_data_indices = np.random.choice(self.attack_data_indices, n_points, replace=False)
        self.logger.info(f"Number of attack data points after subsampling: {len(chosen_attack_data_indices)}")

        # create labels and change dataset to be used for regression
        regression_features = self.handler.get_features(chosen_attack_data_indices)
        regression_labels = np.array(self.signal([self.target_model],
                                                 self.handler,
                                                 self.attack_data_indices)).squeeze()

        # Create a TensorDataset
        attack_data = Dataset(regression_features, regression_labels)
        attack_dataloader = DataLoader(attack_data, batch_size=64, shuffle=True,)

        # train quantile regressor
        self.logger.info("Training the quantile regressor")
        optimizer = torch.optim.Adam(self.quantile_regressor.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = PinballLoss(self.quantiles)
        self.train_quantile_regressor(attack_dataloader, criterion, optimizer, self.epochs)
        self.logger.info("Training of quantile regressor completed")

    def train_quantile_regressor(
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
            self.logger.info(log_train_str)

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
        self.logger.info("Running the attack on the target model")
        score = []
        for data, _ in audit_dataloader:
            score.extend(self.quantile_regressor(data).detach().numpy())
        score = np.array(score).T

        self.logger.info("Attack completed")

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


