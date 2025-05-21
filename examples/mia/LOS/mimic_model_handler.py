import os
import pickle
from typing import Self

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from torch import Tensor, cuda, device, from_numpy, no_grad, nn, optim, sigmoid, unique
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
              model: nn.Module,
              dataloader: DataLoader,
              criterion: nn.Module ,
              optimizer: optim.Optimizer ,
              early_stop_loader:DataLoader,
              epochs: int,
              patience_early_stopping: int,
              patience_lr: float,
              min_delta: float,
              ) -> TrainingOutput:

        device_name = device("cuda" if cuda.is_available() else "cpu")
        model.to(device_name)

        # Early Stopping
        min_loss_epoch_valid = float("inf")  # Initialize to infinity for comparison
        patient_epoch = 0  # Initialize patient counter
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=patience_lr)

        accuracy_history = []
        loss_history = []


        for epoch in tqdm(range(epochs), desc="Training Progress"):
            train_loss, train_acc, total_samples = 0, 0, 0
            model.train()
            all_predictions = []
            all_labels = []

            for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                inputs = inputs.to(device_name, non_blocking=True)
                labels = labels.to(device_name, non_blocking=True)
                labels = labels.long().float()

                prediction = model(inputs).squeeze(dim =1)

                all_labels.append(to_numpy(labels))
                all_predictions.append(to_numpy(prediction))

                loss = criterion(prediction, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()* labels.size(0)
                total_samples += labels.size(0)
             
            train_loss /= total_samples
            loss_history.append(train_loss)

            # Concatenate all accumulated predictions and labels
            all_predictions = np.concatenate(all_predictions, axis=0)
            # Convert predictions to class indices
            all_predictions = (all_predictions > 0)
            all_labels = np.concatenate(all_labels, axis=0).astype(int)

            # Compute accuracy
            train_acc = accuracy_score(all_labels, all_predictions)
            accuracy_history.append(train_acc)

            # Test the model
            results = self.eval( early_stop_loader, model, criterion)
            test_loss = results.loss

            # Early stopping
            # Assume test_loss is computed for validation set
            if test_loss < min_loss_epoch_valid - min_delta:  # Improvement condition
                min_loss_epoch_valid = test_loss
                patient_epoch = 0
                print(f"Epoch {epoch}: Validation loss improved to {test_loss:.4f}")
            else:
                patient_epoch += 1
                print(f"Epoch {epoch}: No improvement. Patience counter: {patient_epoch}/{patience_early_stopping}")

                if patient_epoch >= patience_early_stopping:
                    print(f"Early stopping at epoch {epoch}. Best validation loss: {min_loss_epoch_valid:.4f}")
                    break

            # Step the scheduler
            scheduler.step(test_loss)

            # Check the learning rate
            current_lr =  optimizer.param_groups[0]["lr"]
            print(f"Learning Rate: {current_lr:.12f}")

            # Stop if learning rate becomes too small
            if current_lr < 1e-12:
                print("Learning rate too small, stopping training.")
                break

            # Print training parameters
            print("Epoch: {}, train_loss: {}, valid_loss: {}, train_acc: {}".format( \
                        epoch, \
                        np.around(train_loss, decimals=8),\
                        np.around(test_loss, decimals=8),\
                        np.around(train_acc, decimals=8) \
                          ))
            

        results = EvalOutput(accuracy = train_acc,
                            loss = train_loss,
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
                target = target.float()
                output = model(data)
                loss += criterion(output.squeeze(), target.squeeze()).item()
                pred = sigmoid(output).squeeze().round()
                acc += pred.eq(target.squeeze()).sum().item()
                total_samples += target.size(0)

        loss /= len(loader)
        acc = float(acc) / total_samples
        return EvalOutput(accuracy=acc, loss=loss)


class GRUDDPHandler(BaseMIMICHandler):
    def train(self,
              dataloader: DataLoader,
              model: nn.Module = None,
              criterion: nn.Module = None,
              optimizer: optim.Optimizer = None,
              epochs: int = None,
              early_stop_loader: DataLoader = None,
              patience: int = 5,
              min_delta: float = 0.0) -> TrainingOutput:

        print("Training shadow models with DP-SGD")
        dpsgd_path = "./target_GRUD_dpsgd/dpsgd_dic.pkl"
        sample_rate = 1 / len(dataloader)

        if not os.path.exists(dpsgd_path):
            raise FileNotFoundError(f"DP config not found at: {dpsgd_path}")

        with open(dpsgd_path, "rb") as file:
            privacy_engine_dict = pickle.load(file)

        noise_multiplier = get_noise_multiplier(
            target_epsilon=privacy_engine_dict["target_epsilon"],
            target_delta=privacy_engine_dict["target_delta"],
            sample_rate=sample_rate,
            epochs=privacy_engine_dict["epochs"],
            epsilon_tolerance=privacy_engine_dict["epsilon_tolerance"],
            accountant="prv",
            eps_error=privacy_engine_dict["eps_error"],
        )

        privacy_engine = PrivacyEngine(accountant="prv")
        model, optimizer, dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=privacy_engine_dict["max_grad_norm"],
        )

        device_name = device("cuda" if cuda.is_available() else "cpu")
        model.to(device_name)
        model.train()

        criterion = self.get_criterion()
        best_loss = float("inf")
        epochs_no_improve = 0
        best_model_state = None

        for e in tqdm(range(epochs), desc="Training Progress"):
            train_loss = 0.0
            for _, (x, labels) in enumerate(tqdm(dataloader, desc="Training Batches")):
                x = x.to(device_name)
                labels = labels.to(device_name).float()

                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output.squeeze(), labels.squeeze())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(dataloader)

            if early_stop_loader is not None:
                val_loss = 0.0
                model.eval()
                with no_grad():
                    for val_x, val_y in early_stop_loader:
                        val_x = val_x.to(device_name)
                        val_y = val_y.to(device_name).float()
                        val_output = model(val_x)
                        val_loss += criterion(val_output.squeeze(), val_y.squeeze()).item()
                val_loss /= len(early_stop_loader)
                model.train()

                if best_loss - val_loss > min_delta:
                    best_loss = val_loss
                    best_model_state = model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"Early stopping at epoch {e+1}")
                        break

        if best_model_state:
            model.load_state_dict(best_model_state)

        binary_predictions = sigmoid(output).squeeze().round().cpu().detach().numpy()
        binary_labels = labels.squeeze().cpu().numpy().astype(int)
        train_acc = accuracy_score(binary_labels, binary_predictions)

        return TrainingOutput(model=model, metrics={"accuracy": train_acc, "loss": train_loss})


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
