"""Implementation of the RMIA attack."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from leakpro.dataset import get_dataset_subset
from leakpro.import_helper import Self
from leakpro.metrics.attack_result import CombinedMetricResult
from leakpro.mia_attacks.attack_utils import AttackUtils
from leakpro.mia_attacks.attacks.attack import AttackAbstract
from leakpro.signals.signal import ModelRescaledLogits


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

class AttackQMIA(AttackAbstract):
    """Implementation of the RMIA attack."""

    def __init__(self:Self, attack_utils: AttackUtils, configs: dict) -> None:
        """Initialize the QMIA attack.

        Args:
        ----
            attack_utils (AttackUtils): Utility class for the attack.
            configs (dict): Configuration parameters for the attack.

        """
        # Initializes the parent metric
        super().__init__(attack_utils)

        self.f_attack_data_size = configs["audit"].get("f_attack_data_size", 0.3)
        self.quantiles = [0, 0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99995, 0.99999]
        self.signal = ModelRescaledLogits()
        self.quantile_regressor = QuantileRegressor(len(self.quantiles))
        self.logger = attack_utils.attack_objects.logger

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
        all_index = np.arange(self.population_size)
        attack_data_size = np.round(self.f_attack_data_size * self.population_size).astype(int)
        self.attack_data_index = np.random.choice(all_index, attack_data_size, replace=False)
        attack_data = get_dataset_subset(self.population, self.attack_data_index)

        # create labels and change dataset to be used for regression
        regression_labels = np.array(self.signal([self.target_model], attack_data)).squeeze()
        attack_data.y = regression_labels
        attack_data.task_type = "regression"
        attack_dataloader = DataLoader(attack_data, batch_size=64, shuffle=True,)

        # train quantile regressor
        optimizer = torch.optim.Adam(self.quantile_regressor.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = PinballLoss(self.quantiles)
        self.train_quantile_regressor(attack_dataloader, criterion, optimizer)

    def train_quantile_regressor(
        self:Self,
        attack_dataloader: DataLoader,
        criterion:torch.nn.Module,
        optimizer:torch.optim,
        epochs: int = 200
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
        for epoch_idx in range(epochs):
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
        audit_dataset = get_dataset_subset(self.population, self.audit_dataset["data"])
        self.target_logits = np.array(self.signal([self.target_model], audit_dataset)).squeeze()

        audit_dataloader = DataLoader(audit_dataset, batch_size=64, shuffle=False)
        score = []
        for data, _ in audit_dataloader:
            score.extend(self.quantile_regressor(data).detach().numpy())
        score = np.array(score).T

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


