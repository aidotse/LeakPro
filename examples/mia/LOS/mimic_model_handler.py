#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
import os
import pickle
from typing import Optional, Self

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from torch import  cuda, device, from_numpy, no_grad, nn, optim, sigmoid
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm


from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier

from leakpro import AbstractInputHandler
from leakpro.schemas import TrainingOutput, EvalOutput
from mimic_data_handler import MIMICUserDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau


def _resolve_dpsgd_metadata_path(configs, explicit_path: Optional[str] = None) -> str:
    """Resolve the DP-SGD metadata path from arg or target config."""
    if explicit_path:
        return explicit_path

    target_cfg = getattr(configs, "target", None)
    if target_cfg is not None:
        target_path = getattr(target_cfg, "dpsgd_path", None)
        if target_path:
            return target_path

    raise FileNotFoundError(
        "DP-SGD is enabled, but no DP-SGD metadata path could be resolved. "
        "Set `target.dpsgd_path` in the config, or pass `dpsgd_metadata_path` explicitly."
    )


class BaseMIMICHandler(AbstractInputHandler):
    UserDataset = MIMICUserDataset

    def __init__(self) -> None:
        super().__init__()

    def get_criterion(self) -> nn.Module:
        return BCEWithLogitsLoss()

    def get_optimizer(self, model: nn.Module) -> optim.Optimizer:
        return optim.Adam(model.parameters(), lr=0.01)


class LRHandler(BaseMIMICHandler):
    def train(self: Self,
              dataloader: DataLoader,
              model: nn.Module = None,
              criterion: nn.Module = None,
              optimizer: optim.Optimizer = None,
              epochs: int = None,) -> TrainingOutput:

        if epochs is None:
            raise ValueError("epochs not found in configs")

        device_name = device("cuda" if cuda.is_available() else "cpu")
        model.to(device_name)

        accuracy_history = []
        loss_history = []

        for epoch in range(epochs):
            train_loss, train_acc, total_samples = 0, 0, 0
            model.train()
            for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                labels = labels.float().unsqueeze(1)
                inputs, labels = inputs.to(device_name, non_blocking=True), labels.to(device_name, non_blocking=True)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                pred = outputs >= 0.5
                loss.backward()
                optimizer.step()

                train_acc += pred.eq(labels.view_as(pred)).sum().item()
                total_samples += labels.size(0)
                train_loss += loss.item() * labels.size(0)

            avg_train_loss = train_loss / total_samples
            train_accuracy = train_acc / total_samples

            accuracy_history.append(train_accuracy)
            loss_history.append(avg_train_loss)

        results = EvalOutput(accuracy=train_accuracy,
                             loss=avg_train_loss,
                             extra={"accuracy_history": accuracy_history, "loss_history": loss_history})
        return TrainingOutput(model=model, metrics=results)

    def eval(self: Self,
             loader: DataLoader,
             model: nn.Module,
             criterion: nn.Module) -> EvalOutput:
        device_name = device("cuda" if cuda.is_available() else "cpu")
        model.to(device_name)
        model.eval()
        loss, acc, total_samples = 0, 0, 0

        with no_grad():
            for data, target in loader:
                data, target = data.to(device_name), target.to(device_name)
                target = target.float().unsqueeze(1)
                output = model(data)
                loss += criterion(output, target).item()
                pred = (output) >= 0.5
                acc += pred.eq(target).sum().item()
                total_samples += target.size(0)
        loss /= len(loader)
        acc = float(acc) / total_samples

        return EvalOutput(accuracy=acc, loss=loss)


def _drain_loader(loader: DataLoader, n_features: int) -> tuple:
    """Materialize a dataloader into a single feature matrix and label vector.

    Preallocated rather than concatenated: the LOS design matrix is 23944 x 7488 float32
    (~717 MB), and building a list of batches before concatenating would transiently double
    the memory held for every shadow model.
    """
    n_samples = len(loader.dataset)
    x = np.empty((n_samples, n_features), dtype=np.float32)
    y = np.empty(n_samples, dtype=np.float32)

    offset = 0
    for inputs, labels in loader:
        batch_size = labels.shape[0]
        x[offset:offset + batch_size] = inputs.detach().cpu().numpy().reshape(batch_size, -1)
        y[offset:offset + batch_size] = labels.detach().cpu().numpy().reshape(-1)
        offset += batch_size

    if offset != n_samples:
        raise RuntimeError(f"Drained {offset} samples but dataset reports {n_samples}")
    return x, y


def _margin_metrics(margins: np.ndarray, y: np.ndarray) -> EvalOutput:
    """Accuracy and mean BCE-with-logits loss from raw margins."""
    margins = margins.reshape(-1)
    accuracy = float(((margins > 0).astype(np.float32) == y).mean())
    # log(1 + exp(-m)) for members of class 1, log(1 + exp(m)) for class 0, computed stably.
    signed = np.where(y > 0.5, -margins, margins)
    loss = float(np.mean(np.logaddexp(0.0, signed)))
    return EvalOutput(accuracy=accuracy, loss=loss)


class XGBHandler(BaseMIMICHandler):
    """Input handler for the XGBoost length-of-stay target.

    Trees are not trained by gradient descent, so `criterion`, `optimizer` and `epochs` are
    accepted (LeakPro's ShadowModelHandler passes them positionally) and then ignored. All
    reported metrics still use BCEWithLogitsLoss on raw margins, so they are directly
    comparable to the LR and GRU-D handlers.

    NOTE: `train` and `eval` must not call helper methods through `self`. MIAHandler rebinds
    the methods it finds on AbstractInputHandler onto itself via `types.MethodType(attr, self)`,
    so at audit time `self` is a MIAHandler, not an XGBHandler, and any attribute added by this
    subclass is invisible. Hence the module-level helpers above.
    """

    def get_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Return a stepless placeholder optimizer.

        Never used to update anything, but `MIAHandler.get_target_replica` builds one for every
        shadow model, and torch optimizers raise on an empty parameter list.
        """
        return optim.SGD(model.parameters(), lr=0.0)

    def train(self: Self,
              dataloader: DataLoader,
              model: nn.Module = None,
              criterion: nn.Module = None,  # noqa: ARG002
              optimizer: optim.Optimizer = None,  # noqa: ARG002
              epochs: int = None,  # noqa: ARG002
              ) -> TrainingOutput:

        if model is None:
            raise ValueError("XGBHandler.train requires a model instance")

        x, y = _drain_loader(dataloader, model.input_dim)
        print(f"Fitting XGBoost on {x.shape[0]} samples x {x.shape[1]} features...")
        model.fit(x, y.astype(np.int64))

        with no_grad():
            margins = model(from_numpy(x)).numpy()

        results = _margin_metrics(margins, y)
        print(f"Train accuracy={results.accuracy:.4f}, loss={results.loss:.6f}")
        return TrainingOutput(model=model, metrics=results)

    def eval(self: Self,
             loader: DataLoader,
             model: nn.Module,
             criterion: nn.Module = None) -> EvalOutput:  # noqa: ARG002

        x, y = _drain_loader(loader, model.input_dim)
        with no_grad():
            margins = model(from_numpy(x)).numpy()
        return _margin_metrics(margins, y)


class GRUHandler(BaseMIMICHandler):

    def train(self,
              dataloader: DataLoader,
              model: nn.Module,
              criterion: nn.Module,
              optimizer: optim.Optimizer,
              epochs: int,
              early_stop_loader: Optional[DataLoader] = None,
              patience_early_stopping: int = 5,
              patience_lr: float = 2,
              min_delta: float = 0.00001,
              dpsgd_metadata_path: Optional[str] = None,
              ) -> TrainingOutput:

        device_name = device("cuda" if cuda.is_available() else "cpu")

        resolved_dpsgd_path = None
        if dpsgd_metadata_path is not None:
            resolved_dpsgd_path = _resolve_dpsgd_metadata_path(getattr(self, "configs", None), dpsgd_metadata_path)
        else:
            target_cfg = getattr(getattr(self, "configs", None), "target", None)
            if target_cfg is not None and getattr(target_cfg, "dpsgd_path", None):
                resolved_dpsgd_path = _resolve_dpsgd_metadata_path(getattr(self, "configs", None))

        if resolved_dpsgd_path is not None:
            print("Training with DP-SGD...")
            with open(resolved_dpsgd_path, "rb") as f:
                config = pickle.load(f)

            sample_rate = 1 / len(dataloader)
            noise_multiplier = get_noise_multiplier(
                target_epsilon=config["target_epsilon"],
                target_delta=config["target_delta"],
                sample_rate=sample_rate,
                epochs=config["epochs"],
                epsilon_tolerance=config["epsilon_tolerance"],
                accountant="prv",
                eps_error=config["eps_error"]
            )

            privacy_engine = PrivacyEngine(accountant="prv")
            model, optimizer, dataloader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=dataloader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=config["max_grad_norm"]
            )
            print("DP-SGD training enabled.")

        model.to(device_name)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=patience_lr)

        min_val_loss = float("inf")
        patience_counter = 0
        accuracy_history, loss_history = [], []

        for epoch in tqdm(range(epochs), desc="Training Progress"):
            # TRAINING LOOP
            model.train()
            total_loss, total_samples = 0.0, 0
            all_preds, all_labels = [], []

            for inputs, labels in dataloader:
                inputs = inputs.to(device_name)
                labels = labels.to(device_name).long().float()

                prediction = model(inputs).squeeze(1)
                loss = criterion(prediction, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)

                all_preds.append(prediction.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())

            avg_loss = total_loss / total_samples
            preds = (np.concatenate(all_preds) > 0)
            labels = np.concatenate(all_labels).astype(int)
            train_acc = accuracy_score(labels, preds)

            loss_history.append(avg_loss)
            accuracy_history.append(train_acc)

            print(f"Epoch {epoch}: train_loss={avg_loss:.6f}, train_acc={train_acc:.4f}")

            # EARLY STOPPING
            if early_stop_loader is not None:
                model.eval()
                val_loss, val_acc, total_samples = 0, 0, 0
                with no_grad():
                    for val_data, val_target in early_stop_loader:
                        val_data = val_data.to(device_name)
                        val_target = val_target.to(device_name).float()
                        val_output = model(val_data)
                        val_loss += criterion(val_output.squeeze(), val_target.squeeze()).item()
                        val_pred = sigmoid(val_output).squeeze().round()
                        val_acc += val_pred.eq(val_target.squeeze()).sum().item()
                        total_samples += val_target.size(0)

                val_loss /= len(early_stop_loader)

                if val_loss < min_val_loss - min_delta:
                    print(f"Epoch {epoch}: Validation loss improved to {val_loss:.6f}")
                    min_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f"Epoch {epoch}: No improvement. Patience {patience_counter}/{patience_early_stopping}")
                    scheduler.step(val_loss)

                    if patience_counter >= patience_early_stopping:
                        print(f"Early stopping at epoch {epoch}. Best val loss: {min_val_loss:.6f}")
                        break

                    if scheduler.optimizer.param_groups[0]["lr"] < 1e-12:
                        print("Learning rate too small. Stopping training.")
                        break

        results = EvalOutput(
            accuracy=accuracy_history[-1],
            loss=loss_history[-1],
            extra={"accuracy_history": accuracy_history, "loss_history": loss_history}
        )
        return TrainingOutput(model=model, metrics=results)

    def eval(self: Self,
             loader: DataLoader,
             model: nn.Module,
             criterion: nn.Module) -> EvalOutput:
        device_name = device("cuda" if cuda.is_available() else "cpu")
        model.to(device_name)
        model.eval()
        loss, acc, total_samples = 0, 0, 0

        with no_grad():
            for data, target in loader:
                data, target = data.to(device_name), target.to(device_name)
                target = target.float()
                output = model(data)
                loss += criterion(output.squeeze(), target.squeeze()).item()
                pred = sigmoid(output).squeeze().round()
                acc += pred.eq(target.squeeze()).sum().item()
                total_samples += target.size(0)

        loss /= len(loader)
        acc = float(acc) / total_samples
        return EvalOutput(accuracy=acc, loss=loss)
