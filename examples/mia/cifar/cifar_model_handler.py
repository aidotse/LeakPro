"""CIFAR standard model handler — train() and eval() only.

Usage:
    from cifar_data_handler import CifarDataHandler
    from cifar_model_handler import CifarModelHandler
    leakpro = LeakPro(CifarDataHandler, config_path, model_handler=CifarModelHandler)
"""

import copy

import torch
from torch import cuda, device, no_grad, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy

from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.schemas import EvalOutput, TrainingOutput


class CifarModelHandler(AbstractInputHandler, role="model"):
    """Standard (non-DP) training handler for CIFAR. No dataset logic."""

    def train(
        self,
        dataloader: DataLoader,
        model: torch.nn.Module = None,
        criterion: torch.nn.Module = None,
        optimizer: optim.Optimizer = None,
        epochs: int = None,
    ) -> TrainingOutput:
        val_split = 0.1
        patience = 10
        dataset = dataloader.dataset
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size])

        if hasattr(dataset, "augment"):
            val_dataset = copy.copy(dataset)
            val_dataset.augment = None
            if hasattr(val_dataset, "erase_post_norm"):
                val_dataset.erase_post_norm = None
            val_subset = torch.utils.data.Subset(val_dataset, val_subset.indices)

        train_loader = DataLoader(train_subset, batch_size=dataloader.batch_size,
                                  shuffle=True, num_workers=dataloader.num_workers)
        val_loader = DataLoader(val_subset, batch_size=dataloader.batch_size,
                                shuffle=False, num_workers=dataloader.num_workers)

        if epochs is None:
            raise ValueError("epochs not found in configs")

        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        model.to(gpu_or_cpu)

        accuracy_history, loss_history = [], []
        val_loss_history, val_acc_history = [], []
        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(epochs):
            train_loss, train_acc, total_samples = 0, 0, 0
            model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for inputs, labels in pbar:
                labels = labels.long()
                inputs, labels = inputs.to(gpu_or_cpu, non_blocking=True), labels.to(gpu_or_cpu, non_blocking=True)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                pred = outputs.argmax(dim=1)
                loss.backward()
                optimizer.step()
                train_acc += pred.eq(labels.view_as(pred)).sum().item()
                total_samples += labels.size(0)
                train_loss += loss.item() * labels.size(0)
                pbar.set_postfix(loss=f"{train_loss/total_samples:.4f}", acc=f"{train_acc/total_samples:.4f}")
            scheduler.step()

            avg_train_loss = train_loss / total_samples
            train_accuracy = train_acc / total_samples
            accuracy_history.append(train_accuracy)
            loss_history.append(avg_train_loss)

            model.eval()
            val_loss, val_acc, val_samples = 0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(gpu_or_cpu), labels.to(gpu_or_cpu)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * labels.size(0)
                    val_acc += outputs.argmax(dim=1).eq(labels).sum().item()
                    val_samples += labels.size(0)
            avg_val_loss = val_loss / val_samples
            val_loss_history.append(avg_val_loss)
            val_acc_history.append(val_acc / val_samples)
            print(f"Validation loss at epoch {epoch+1}: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    model.load_state_dict(best_model_state)
                    break

        model.to("cpu")
        results = EvalOutput(
            accuracy=train_accuracy, loss=avg_train_loss,
            extra={"accuracy_history": accuracy_history, "loss_history": loss_history,
                   "val_loss_history": val_loss_history, "val_acc_history": val_acc_history},
        )
        return TrainingOutput(model=model, metrics=results)

    def eval(self, loader, model, criterion) -> EvalOutput:
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        model.to(gpu_or_cpu)
        model.eval()
        loss, acc, total_samples = 0, 0, 0
        if hasattr(loader.dataset, "augment"):
            loader.dataset.augment = None
        with no_grad():
            for data, target in loader:
                data, target = data.to(gpu_or_cpu), target.to(gpu_or_cpu)
                target = target.view(-1)
                output = model(data)
                loss += criterion(output, target).item() * target.size(0)
                acc += output.argmax(dim=1).eq(target).sum().item()
                total_samples += target.size(0)
        return EvalOutput(accuracy=float(acc) / total_samples, loss=loss / total_samples)
