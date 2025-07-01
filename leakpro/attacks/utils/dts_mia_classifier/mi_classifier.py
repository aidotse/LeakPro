"""Class used within the DTS attack."""

from typing import Any, Dict

import numpy as np
import torch
from torch import cuda, nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from leakpro.attacks.utils.dts_mia_classifier.models.inception_time import InceptionTime
from leakpro.attacks.utils.dts_mia_classifier.models.lstm_classifier import LSTMClassifier
from leakpro.utils.logger import logger


class MIClassifier():
    """Class for the MIC (Membership Inference Classification) model.

    A MIC model is essentially a time series binary classification model,
    here we provide an API for initing the model with different architectures, data, training parameters, etc.
    """

    def __init__(
            self,
            seq_len: int,
            num_input_variables: int,
            model: str,
            model_kwargs: Dict[str, Any] = None
        ) -> None:

        self.device = torch.device("cuda" if cuda.is_available() else "cpu")
        model_kwargs = model_kwargs or {}

        # Sanity check for InceptionTime
        if model.lower() == "inceptiontime" and "kernel_sizes" in model_kwargs:
            max_kernel_size = max(model_kwargs["kernel_sizes"])
            if max_kernel_size > seq_len:
                logger.warning(f"InceptionTime: Maximum kernel size ({max_kernel_size}) is greater than input sequence length ({seq_len}).")  # noqa: E501

        if model.lower() == "lstm":
            self.model = LSTMClassifier(
                num_input_variables,
                **model_kwargs
            )
        elif model.lower() == "inceptiontime":
            self.model = InceptionTime(
                num_input_variables,
                seq_len,
                **model_kwargs
            )
        else:
            raise ValueError(f"Unknown model: {model}. Must be one of ['LSTM', 'InceptionTime'].")

    def fit(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int,
            early_stopping_patience: int,
            verbose: int = 0
        ) -> None:
        """Fit the MIC model to supplied MIC data. Validation loader is used to employ early stopping."""

        self.model.to(self.device)
        self.model.train()

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters())

        best_val_loss = (-1, np.inf)  # (epoch, validation loss)
        best_state_dict = self.model.state_dict()

        for i in tqdm(range(epochs), desc="Training MI Classifier"):
            self.model.train()

            train_loss = 0.0
            correct = 0
            total = 0

            for data, target in train_loader:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                pred = self.model(data)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                # Accuracy
                pred_label = (pred >= 0.5).float()
                correct += (pred_label == target).sum().item()
                total += target.numel()

            train_loss /= len(train_loader)
            train_acc = correct / total

            val_loss, val_acc = self.evaluate(val_loader, criterion)

            if verbose > 0:
                logger.info(f"Epoch {i+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")  # noqa: E501

            if val_loss < best_val_loss[1]:
                best_val_loss = (i, val_loss)
                best_state_dict = self.model.state_dict()
            elif i - best_val_loss[0] > early_stopping_patience:
                logger.info(f"Training stopped early at epoch {i+1}.")
                break

        # Restore best weights
        self.model.load_state_dict(best_state_dict)
        logger.info("Best weights restored.")

    def evaluate(
            self,
            loader: DataLoader,
            criterion: nn.Module
        ) -> tuple[float, float]:
        """Evaluate MIC model on supplied data under specified loss criterion."""

        self.model.eval()
        self.model.to(self.device)
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                pred = self.model(data)
                total_loss += criterion(pred, target).item()

                # Compute accuracy
                pred_label = (pred >= 0.5).float()
                correct += (pred_label == target).sum().item()
                total += target.numel()

        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def predict(
            self,
            X_tensor: torch.Tensor,  # noqa: N803
            batch_size: int
        ) -> np.ndarray:
        """Predict membership label(s) given input X_tensor."""

        self.model.eval()
        self.model.to(self.device)
        all_preds = []

        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size].to(self.device)
                preds = self.model(batch).detach().cpu().numpy()
                all_preds.append(preds)

        return np.concatenate(all_preds, axis=0)
