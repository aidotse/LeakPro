"""Module containing the class to handle the user input for the CIFAR10 dataset."""

import torch
from torch import cuda, device, optim, sigmoid
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from leakpro import AbstractInputHandler
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.schemas import TrainingOutput

class TabularInputHandler(AbstractInputHandler):
    """Class to handle the user input for the CIFAR10 dataset."""

    def __init__(self, configs: dict) -> None:
        super().__init__(configs = configs)

    def train(
        self,
        dataloader: DataLoader,
        model: torch.nn.Module = None,
        criterion: torch.nn.Module = None,
        optimizer: optim.Optimizer = None,
        epochs: int = None,
    ) -> dict:
        """Model training procedure."""

        dev = device("cuda" if cuda.is_available() else "cpu")
        model.to(dev)
        model.train()

        criterion = self.get_criterion()
        optimizer = self.get_optimizer(model)

        for e in tqdm(range(epochs), desc="Training Progress"):
            model.train()
            train_acc, train_loss = 0.0, 0.0

            for data, target in dataloader:
                target = target.float().unsqueeze(1)
                data, target = data.to(dev, non_blocking=True), target.to(dev, non_blocking=True)
                optimizer.zero_grad()
                output = model(data)

                loss = criterion(output, target)
                pred = sigmoid(output) >= 0.5
                train_acc += pred.eq(target).sum().item()

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

        train_acc = train_acc/len(dataloader.dataset)
        train_loss = train_loss/len(dataloader)
        
        output_dict = {"model": model, "metrics": {"accuracy": train_acc, "loss": train_loss}}
        training_output = TrainingOutput(**output_dict)
        return training_output


    class UserDataset(AbstractInputHandler.UserDataset):
        
        def __init__(self, data, targets, **kwargs):
            self.data = data
            self.targets = targets

            for key, value in kwargs.items():
                setattr(self, key, value)

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]