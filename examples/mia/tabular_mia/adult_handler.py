#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Module containing the class to handle the user input for the CIFAR10 dataset."""

import torch
from torch import no_grad, optim
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from leakpro.schemas import EvalOutput, TrainingOutput
from leakpro.utils.device import get_device, mark_step

from leakpro import AbstractInputHandler


class AdultInputHandler(AbstractInputHandler):
    """Class to handle the user input for the Adult tabular dataset."""

    def __init__(self, configs: dict) -> None:
        super().__init__(configs = configs)

    class UserDataset(AbstractInputHandler.UserDataset):
        """Thin wrapper around tabular (data, targets) tensors."""

        def __init__(self, data, targets, **kwargs) -> None:
            self.data = data.float() if isinstance(data, torch.Tensor) else torch.tensor(data, dtype=torch.float32)
            self.targets = targets.long() if isinstance(targets, torch.Tensor) else torch.tensor(targets, dtype=torch.long)


    def get_criterion(self)->None:
        """Set the BCEWithLogitsLoss for the model."""
        return BCEWithLogitsLoss()

    def get_optimizer(self, model:torch.nn.Module) -> None:
        """Set the optimizer for the model."""
        learning_rate = 0.1
        momentum = 0.8
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    def train(
        self,
        dataloader: DataLoader,
        model: torch.nn.Module = None,
        criterion: torch.nn.Module = None,
        optimizer: optim.Optimizer = None,
        epochs: int = None,
    ) -> TrainingOutput:
        """Model training procedure."""

        dev = get_device()
        model.to(dev)
        model.train()

        criterion = self.get_criterion()
        optimizer = self.get_optimizer(model)

        for e in tqdm(range(epochs), desc="Training Progress"):
            model.train()
            train_acc, train_loss, total_samples = 0.0, 0.0, 0

            for data, target in dataloader:
                target = target.float().unsqueeze(1)
                data, target = data.to(dev, non_blocking=True), target.to(dev, non_blocking=True)
                optimizer.zero_grad()
                output = model(data)

                loss = criterion(output, target)
                pred = output >= 0.5
                train_acc += pred.eq(target).sum().item()

                loss.backward()
                optimizer.step()
                mark_step(dev)
                train_loss += loss.item()
                total_samples += target.size(0)

        train_acc = train_acc/len(dataloader.dataset)
        train_loss = train_loss/len(dataloader)
        
        
        output_dict = {"model": model, "metrics": {"accuracy": train_acc, "loss": train_loss}}
        output = TrainingOutput(**output_dict)

        return output

    def eval(
        self,
        dataloader: DataLoader,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        device: str = None,
    ) -> EvalOutput:
        """Evaluate the model on the given dataloader."""
        dev = get_device()
        model.to(dev)
        model.eval()
        loss, acc, total_samples = 0.0, 0.0, 0
        with no_grad():
            for data, target in dataloader:
                target = target.float().unsqueeze(1)
                data, target = data.to(dev, non_blocking=True), target.to(dev, non_blocking=True)
                output = model(data)
                mark_step(dev)
                loss += criterion(output, target).item() * target.size(0)
                pred = output >= 0.5
                acc += pred.eq(target).sum().item()
                total_samples += target.size(0)
        model.to("cpu")
        return EvalOutput(accuracy=float(acc) / total_samples, loss=loss / total_samples)
