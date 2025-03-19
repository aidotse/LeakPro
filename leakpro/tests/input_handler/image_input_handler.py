"""Module containing the class to handle the user input for the CIFAR10 dataset."""

import torch
from torch import cuda, device, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger
from leakpro.schemas import TrainingOutput, EvalOutput


class ImageInputHandler(AbstractInputHandler):
    """Class to handle the user input for the CIFAR10 dataset."""

    def __init__(self:Self, configs: dict) -> None:
        super().__init__(configs = configs)

    def eval(self, dataloader, model, criterion, device):
        """Model evaluation procedure."""
        model.to(device)
        model.eval()
        test_loss, acc, total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, labels).item()
                pred = outputs.data.max(1, keepdim=True)[1]
                acc += pred.eq(labels.data.view_as(pred)).sum()
                total += len(labels)
        test_loss /= len(dataloader)
        test_acc = acc / total
        model.to("cpu")
        return EvalOutput(loss=test_loss, accuracy=test_acc)

    def train(
        self: Self,
        dataloader: DataLoader,
        model: torch.nn.Module = None,
        criterion: torch.nn.Module = None,
        optimizer: optim.Optimizer = None,
        epochs: int = None,
    ) -> TrainingOutput:
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
                train_loss += loss.item()
                total_samples += len(labels)
                
            train_loss /= len(dataloader)
            train_acc = train_acc / total_samples

            log_train_str = (f"Epoch: {epoch+1}/{epochs} | Train Loss: {train_loss} | "f"Train Acc: {train_acc}")
            logger.info(log_train_str)
        model.to("cpu")

        eval_output = EvalOutput(accuracy=train_acc, loss=train_loss)
        training_output = TrainingOutput(model=model, metrics=eval_output)
        return training_output

    class UserDataset(AbstractInputHandler.UserDataset):
        """Dataset with a subset method."""

        def __init__(self, data, targets, **kwargs):
            super().__init__(data, targets, **kwargs)
            self.data = data
            self.targets = targets

        def __getitem__(self, index: int) -> tuple:
            """Return a sample from the dataset."""
            return self.data[index], self.targets[index]

        def __len__(self) -> int:
            """Return the length of the dataset."""
            return len(self.targets)