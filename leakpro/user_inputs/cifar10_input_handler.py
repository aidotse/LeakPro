"""Module containing the class to handle the user input for the CIFAR10 dataset."""

import logging

import torch
from torch import cuda, device, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from leakpro.import_helper import Self
from leakpro.user_inputs.abstract_input_handler import AbstractInputHandler


class Cifar10InputHandler(AbstractInputHandler):
    """Class to handle the user input for the CIFAR10 dataset."""

    def __init__(self:Self, configs: dict, logger:logging.Logger) -> None:
        super().__init__(configs = configs, logger = logger)

        self.set_criterion()

    def set_criterion(self:Self)->None:
        """Set the CrossEntropyLoss for the model."""
        self.criterion = torch.nn.CrossEntropyLoss()

    def set_optimizer(self: Self, model:torch.nn.Module) -> None:
        """Set the optimizer for the model."""
        learning_rate = 0.1
        momentum = 0.8
        self.optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    def train(
        self: Self,
        dataloader: DataLoader,
        model: torch.nn.Module = None,
        criterion: torch.nn.Module = None,
        optimizer: optim.Optimizer = None,
    ) -> dict:
        """Model training procedure."""

        # read hyperparams for training (the parameters for the dataloader are defined in get_dataloader):
        epochs = self.configs.get(["shadow_model"]["epochs"], None)
        if epochs is None:
            raise ValueError("epochs not found in configs")

        # prepare training
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        model.to(gpu_or_cpu)

        # training loop
        for epoch in range(epochs):
            train_loss, train_acc = 0, 0
            model.train()
            for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                labels = labels.long()
                inputs, labels = inputs.to(gpu_or_cpu, non_blocking=True), labels.to(gpu_or_cpu, non_blocking=True)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                pred = outputs.data.max(1, keepdim=True)[1]
                loss.backward()
                optimizer.step()

                # Accumulate performance of shadow model
                train_acc += pred.eq(labels.data.view_as(pred)).sum()
                train_loss += loss.item()

            log_train_str = (
                f"Epoch: {epoch+1}/{epochs} | Train Loss: {train_loss/len(dataloader):.8f} | "
                f"Train Acc: {float(train_acc)/len(dataloader.dataset):.8f}")
            self.logger.info(log_train_str)
        model.to("cpu")

        return {"model": model, "metrics": {"accuracy": train_acc, "loss": train_loss}}
