import os
import torch
from torch import cuda, device, optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from leakpro import AbstractInputHandler
from leakpro.schemas import TrainingOutput
import kornia
import time


class CelebA_InputHandler(AbstractInputHandler):
    """Class to handle the user input for the CelebA dataset for plgmi attack."""
    
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
            train_loss, train_acc, total_samples = 0.0, 0, 0
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
                train_loss += loss.item()* labels.size(0)
                total_samples += labels.size(0)

            train_loss /= total_samples
            train_acc /= total_samples
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        return TrainingOutput(loss=train_loss, accuracy=train_acc)
    
    def eval(
        self,
        dataloader: DataLoader,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
    ) -> dict:
        """Model evaluation procedure."""
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        model.to(gpu_or_cpu)
        model.eval()

        eval_loss, eval_acc, total_samples = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Evaluating"):
                inputs, labels = inputs.to(gpu_or_cpu), labels.to(gpu_or_cpu)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Performance metrics
                preds = outputs.argmax(dim=1)
                eval_acc += (preds == labels).sum().item()
                eval_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)

        eval_loss /= total_samples
        eval_acc /= total_samples
        print(f"Evaluation - Loss: {eval_loss:.4f}, Accuracy: {eval_acc:.4f}")
        return TrainingOutput(loss=eval_loss, accuracy=eval_acc)


    # TO USE OWN METHODS UNCOMMENT
    # def pretrain(self):
    #     """Pretrain the model if necessary."""
    #     pass

    # TO USE OWN METHODS UNCOMMENT
    # def finetune(self):
    #     """Pretrain the model if necessary."""
    #     pass