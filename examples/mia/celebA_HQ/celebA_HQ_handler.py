import torch
from torch import cuda, device, optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from leakpro.input_handler.minv_handler import MINVHandler
from leakpro.schemas import TrainingOutput


class CelebAHQInputHandler(MINVHandler):
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
    ) -> TrainingOutput:
        """Model training procedure."""

        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        model.to(gpu_or_cpu)

        for epoch in range(epochs):
            train_loss, train_acc , total_samples= 0.0, 0, 0
            model.train()
            for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                inputs, labels = inputs.to(gpu_or_cpu), labels.to(gpu_or_cpu)
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Performance metrics
                preds = outputs.argmax(dim=1)
                train_acc += (preds == labels).sum().item()
                total_samples += labels.size(0)
                train_loss += loss.item()

        avg_train_loss = train_loss / len(dataloader)
        train_accuracy = train_acc / total_samples  
        model.to("cpu")

        output_dict = {"model": model, "metrics": {"accuracy": train_accuracy, "loss": avg_train_loss}}
        output = TrainingOutput(**output_dict)
        
        return output

