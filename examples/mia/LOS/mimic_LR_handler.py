
import torch
from torch import cuda, device, optim, sigmoid
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from leakpro import AbstractInputHandler


class MimicInputHandler(AbstractInputHandler):
    """Class to handle the user input for the MIMICIII dataset."""

    def __init__(self, configs: dict) -> None:
        super().__init__(configs = configs)


    def get_criterion(self)->None:
        """Set the Binary Cross Entropy Loss for the model."""
        return BCELoss()

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

        compute_device = device("cuda" if cuda.is_available() else "cpu")
        model.to(compute_device)
        model.train()

        criterion = self.get_criterion()
        optimizer = self.get_optimizer(model)

        for e in tqdm(range(epochs), desc="Training Progress"):
            model.train()
            train_acc, train_loss = 0.0, 0.0

            for data, target in dataloader:
                target = target.float().unsqueeze(1)
                data, target = data.to(compute_device, non_blocking=True), target.to(compute_device, non_blocking=True)
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

        return {"model": model, "metrics": {"accuracy": train_acc, "loss": train_loss}}
