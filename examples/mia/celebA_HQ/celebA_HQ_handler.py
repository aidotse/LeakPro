import torch
from torch import cuda, device, optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
from leakpro import AbstractInputHandler 

class CelebAHQInputHandler(AbstractInputHandler):
    """Class to handle the user input for the CelebA_HQ dataset."""

    def __init__(self, configs: dict) -> None:
        super().__init__(configs=configs)
        print("Configurations:", configs)

    def get_criterion(self) -> torch.nn.Module:
        """Set the CrossEntropyLoss for the model."""
        return CrossEntropyLoss()

    def get_optimizer(self, model: torch.nn.Module) -> optim.Optimizer:
        """Set the optimizer for the model."""
        return optim.SGD(model.parameters())

    def train(
        self,
        dataloader: DataLoader,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: optim.Optimizer,
        epochs: int,
    ) -> dict:
        """Model training procedure."""

        if not epochs:
            raise ValueError("Epochs not found in configurations")

        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        model.to(gpu_or_cpu)

        for epoch in range(epochs):
            train_loss, train_acc = 0.0, 0
            model.train()
            for inputs, labels in tqdm(dataloader, desc="Training Progress"):
                inputs, labels = inputs.to(gpu_or_cpu), labels.to(gpu_or_cpu)
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Performance metrics
                preds = outputs.argmax(dim=1)
                train_acc += (preds == labels).sum().item()
                train_loss += loss.item()

        model.to("cpu")

        return {"model": model, "metrics": {"accuracy": train_acc / len(dataloader.dataset), "loss": train_loss}}

    def evaluate(self, dataloader: DataLoader, model: torch.nn.Module, criterion: torch.nn.Module) -> dict:
        """Evaluate the model."""
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        model.to(gpu_or_cpu)
        model.eval()

        test_loss, test_acc = 0.0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Evaluating"):
                inputs, labels = inputs.to(gpu_or_cpu), labels.to(gpu_or_cpu)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                preds = outputs.argmax(dim=1)
                test_acc += (preds == labels).sum().item()
                test_loss += loss.item()

        model.to("cpu")

        return {"accuracy": test_acc / len(dataloader.dataset), "loss": test_loss}
