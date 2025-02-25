"""Module containing the class to handle the user input for the CIFAR100 dataset."""

import torch
from torch import cuda, device, optim, sigmoid
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from leakpro import AbstractInputHandler

class CifarInputHandler(AbstractInputHandler):
    """Class to handle the user input for the CIFAR100 dataset."""

    def __init__(self, configs: dict) -> None:
        super().__init__(configs = configs)

    def get_criterion(self, criterion_name:str=None)->None:
        """Set the CrossEntropyLoss for the model."""
        if type(criterion_name) == str:
            crit = self._get_criterion_class(criterion_name)()
            return crit
        return CrossEntropyLoss()

    def get_optimizer(self, model:torch.nn.Module, optimizer_dict: dict) -> optim.Optimizer: #) -> None:
        """Set the optimizer for the model."""
        optim_dict = {k:v for k,v in optimizer_dict.items() if k not in ("name")}
        optimizer_name = optimizer_dict["name"]
        optimizer = self._get_optimizer_class(optimizer_name)
        return optimizer(model.parameters(), **optim_dict)

    def train(
        self,
        dataloader: DataLoader,
        model: torch.nn.Module = None,
        criterion: torch.nn.Module = None,
        optimizer: optim.Optimizer = None,
        epochs: int = None,
    ) -> dict:
        """Model training procedure."""

        # read hyperparams for training (the parameters for the dataloader are defined in get_dataloader):
        if epochs is None:
            raise ValueError("epochs not found in configs")

        # prepare training
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        model.to(gpu_or_cpu)

        print("optimizer", optimizer)

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
                train_acc += pred.eq(labels.data.view_as(pred)).sum() / len(dataloader.dataset)
                train_loss += loss.item() / len(dataloader)

        model.to("cpu")

        return {"model": model, "metrics": {"accuracy": train_acc, "loss": train_loss}}
