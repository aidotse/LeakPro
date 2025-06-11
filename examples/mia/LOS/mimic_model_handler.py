import os
import pickle
from typing import Optional, Self

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from torch import  cuda, device,  no_grad, nn, optim, sigmoid
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm


from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier

from leakpro import AbstractInputHandler
from leakpro.schemas import TrainingOutput, EvalOutput
from mimic_data_handler import MIMICUserDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau


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
              ) -> TrainingOutput:

        device_name = device("cuda" if cuda.is_available() else "cpu")

        if model.dpsgd_path is not None:

            print("Training with DP-SGD...")
            with open( model.dpsgd_path, "rb") as f:
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


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.is_cuda else tensor.detach().numpy()
