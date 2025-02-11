"""Module containing the class to handle the user input for the CIFAR100 dataset."""

import torch
from torch import cuda, device, optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from leakpro import AbstractInputHandler

class CifarInputHandler(AbstractInputHandler):
    """Class to handle the user input for the CIFAR100 dataset."""

    def __init__(self, configs: dict) -> None:
        super().__init__(configs = configs)
        print(configs)


    def get_criterion(self)->None:
        """Set the CrossEntropyLoss for the model."""
        return CrossEntropyLoss()

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
    ) -> dict:
        """Model training procedure."""

        # read hyperparams for training (the parameters for the dataloader are defined in get_dataloader):
        if epochs is None:
            raise ValueError("epochs not found in configs")

        # prepare training
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        model.to(gpu_or_cpu)

        # training loop
        for epoch in range(epochs):
            train_loss, train_acc, total_samples = 0, 0, 0
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
                total_samples += labels.size(0)
                train_loss += loss.item()

        avg_train_loss = train_loss / total_samples
        train_accuracy = train_acc / total_samples  

        model.to("cpu")

        return {"model": model, "metrics": {"accuracy": train_accuracy, "loss": avg_train_loss}}

