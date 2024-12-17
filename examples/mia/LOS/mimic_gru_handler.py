
import torch.nn as nn
from torch import cuda, device, optim, from_numpy, transpose, squeeze
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# from examples.mia.LOS.utils.grud import convert_to_device
from leakpro import AbstractInputHandler

class MimicInputHandlerGRU(AbstractInputHandler):
    """Class to handle the user input for the CIFAR10 dataset."""

    def __init__(self, configs: dict) -> None:
        super().__init__(configs = configs)


    def get_criterion(self)->None:
        """Set the CrossEntropyLoss for the model."""
        return CrossEntropyLoss()

    def get_optimizer(self, model:nn.Module) -> None:
        """Set the optimizer for the model."""
        learning_rate = 0.01
        return optim.Adam(model.parameters(), lr=learning_rate)

    def convert_to_device(self, x):
        device_name = device("cuda" if cuda.is_available() else "cpu")
        return x.to(device_name)
    
    def train(
        self,
        dataloader: DataLoader,
        model: nn.Module = None,
        criterion: nn.Module = None,
        optimizer: optim.Optimizer = None,
        epochs: int = None,
    ) -> dict:
        """Model training procedure."""

        device_name = device("cuda" if cuda.is_available() else "cpu")
        model.to(device_name)
        model.train()

        criterion = self.get_criterion()
        optimizer = self.get_optimizer(model)
        
        for e in tqdm(range(epochs), desc="Training Progress"):
            model.train()
            train_loss = 0.0
            for _, (X, labels) in enumerate(tqdm(dataloader, desc="Training Batches")):

                model.zero_grad()
                X = self.convert_to_device(X)
                labels = self.convert_to_device(labels)
                labels = labels.long()
                prediction = model(X)
                loss = criterion(squeeze(prediction), squeeze(labels).long())


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(dataloader)


        return {"model": model, "metrics": { "loss": train_loss}}
    


