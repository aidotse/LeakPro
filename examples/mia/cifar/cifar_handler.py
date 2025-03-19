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

        # read hyperparams for training (the parameters for the dataloader are defined in get_dataloader):
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
            for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                labels = labels.long()
                inputs, labels = inputs.to(gpu_or_cpu, non_blocking=True), labels.to(gpu_or_cpu, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                pred = outputs.argmax(dim=1) 
                loss.backward()
                optimizer.step()

                # Accumulate performance of shadow model
                train_acc += pred.eq(labels.data.view_as(pred)).sum().item()
                total_samples += labels.size(0)
                train_loss += loss.item()
                
            avg_train_loss = train_loss / len(dataloader)
            train_accuracy = train_acc / total_samples 
            
            accuracy_history.append(train_accuracy) 
            loss_history.append(avg_train_loss)
        
        model.to("cpu")

        results = EvalOutput(accuracy=train_accuracy, loss=avg_train_loss, extra={"accuracy_history": accuracy_history, "loss_history": loss_history})
        return TrainingOutput(model=model, metrics=results)

    def eval(self, loader, model, criterion, device):
        model.to(device)
        model.eval()
        loss, acc = 0, 0
        with no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                target = target.view(-1) 
                output = model(data)
                loss += criterion(output, target).item()
                pred = output.argmax(dim=1) 
                acc += pred.eq(target).sum().item()
            loss /= len(loader)
            acc = float(acc) / len(loader.dataset)
            
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
                self.mean, self.std = self._compute_mean_std()

        def _compute_mean_std(self):
            """Compute mean and std from dataset."""
            mean = self.data.mean(dim=(0, 2, 3))  # Shape (3,)
            std = self.data.std(dim=(0, 2, 3))    # Shape (3,)

            # Reshape to (C, 1, 1) for broadcasting
            mean = mean.view(-1, 1, 1)  
            std = std.view(-1, 1, 1)   
            return mean, std

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