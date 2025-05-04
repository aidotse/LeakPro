"""Module containing the class to handle the user input for the CIFAR100 dataset."""

import torch
from torch import cuda, device, optim, no_grad
from torch.utils.data import DataLoader
from tqdm import tqdm

from leakpro import AbstractInputHandler
from leakpro.schemas import TrainingOutput, EvalOutput

class CifarInputHandler(AbstractInputHandler):
    """Class to handle the user input for the CIFAR100 dataset."""

    def train(
        self,
        dataloader: DataLoader,
        model: torch.nn.Module = None,
        criterion: torch.nn.Module = None,
        optimizer: optim.Optimizer = None,
        epochs: int = None,
    ) -> TrainingOutput:
        """Model training procedure."""

        if epochs is None:
            raise ValueError("epochs not found in configs")

        # prepare training
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        model.to(gpu_or_cpu)

        accuracy_history = []
        loss_history = []
        
        # training loop
        for epoch in range(epochs):
            train_loss, train_acc, total_samples = 0, 0, 0
            model.train()
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for inputs, labels in pbar:
                labels = labels.long()
                inputs, labels = inputs.to(gpu_or_cpu, non_blocking=True), labels.to(gpu_or_cpu, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                pred = outputs.argmax(dim=1) 
                loss.backward()
                optimizer.step()

                # Accumulate performance of shadow model
                train_acc += pred.eq(labels.view_as(pred)).sum().item()
                total_samples += labels.size(0)
                train_loss += loss.item() * labels.size(0)

                running_acc = train_acc / total_samples
                running_loss = train_loss / total_samples
                pbar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}")

                
            avg_train_loss = train_loss / total_samples
            train_accuracy = train_acc / total_samples 
            
            accuracy_history.append(train_accuracy) 
            loss_history.append(avg_train_loss)
        
        model.to("cpu")

        results = EvalOutput(accuracy = train_accuracy,
                             loss = avg_train_loss,
                             extra = {"accuracy_history": accuracy_history, "loss_history": loss_history})
        return TrainingOutput(model = model, metrics=results)

    def eval(self, loader, model, criterion):
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        model.to(gpu_or_cpu)
        model.eval()
        loss, acc = 0, 0
        total_samples = 0
        with no_grad():
            for data, target in loader:
                data, target = data.to(gpu_or_cpu), target.to(gpu_or_cpu)
                target = target.view(-1) 
                output = model(data)
                loss += criterion(output, target).item() * target.size(0)
                pred = output.argmax(dim=1) 
                acc += pred.eq(target).sum().item()
                total_samples += target.size(0)
            loss /= total_samples
            acc = float(acc) / total_samples
            
        output_dict = {"accuracy": acc, "loss": loss}
        return EvalOutput(**output_dict)

    class UserDataset(AbstractInputHandler.UserDataset):
        def __init__(self, data, targets, **kwargs):
            """
            Args:
                data (Tensor): Image data of shape (N, H, W, C) or (N, C, H, W)
                               Expected to be in range [0,1] (normalized).
                targets (Tensor): Corresponding labels.
                mean (Tensor, optional): Precomputed mean for normalization.
                std (Tensor, optional): Precomputed std for normalization.
            """
            assert data.shape[0] == targets.shape[0], "Data and targets must have the same length"
            assert data.max() <= 1.0 and data.min() >= 0.0, "Data should be in range [0,1]"

            self.data = data.float()  # Ensure float type
            self.targets = targets

            for key, value in kwargs.items():
                setattr(self, key, value)
                
            if not hasattr(self, "mean") or not hasattr(self, "std"):
                # Reshape to (C, 1, 1) for broadcasting
                self.mean = self.data.mean(dim=(0, 2, 3)).view(-1, 1, 1)
                self.std = self.data.std(dim=(0, 2, 3)).view(-1, 1, 1)

        def transform(self, x):
            """Normalize using stored mean and std."""
            return (x - self.mean) / self.std 

        def __getitem__(self, index):
            x = self.data[index]
            y = self.targets[index]
            x = self.transform(x)
            return x, y

        def __len__(self):
            return len(self.targets)