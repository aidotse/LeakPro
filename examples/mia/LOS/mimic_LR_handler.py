from typing import Self
from torch import Tensor, cuda, device, optim, no_grad, from_numpy, unique
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch import nn

from leakpro import AbstractInputHandler
from leakpro.schemas import TrainingOutput, EvalOutput
from mimic_data_handler import MIMICUserDataset

class MIMICLRHandler(AbstractInputHandler):
    UserDataset = MIMICUserDataset

    def train(self: Self,
        dataloader: DataLoader,
        model: nn.Module = None,
        criterion: nn.Module = None,
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
            for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                labels = labels.float().unsqueeze(1)
                inputs, labels = inputs.to(gpu_or_cpu, non_blocking=True), labels.to(gpu_or_cpu, non_blocking=True)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                pred = outputs >= 0.5
                loss.backward()
                optimizer.step()

                # Accumulate performance of shadow model
                train_acc += pred.eq(labels.view_as(pred)).sum().item()
                total_samples += labels.size(0)
                train_loss += loss.item() * labels.size(0)

            avg_train_loss = train_loss / total_samples
            train_accuracy = train_acc / total_samples

            accuracy_history.append(train_accuracy)
            loss_history.append(avg_train_loss)

        results = EvalOutput(accuracy = train_accuracy,
                             loss = avg_train_loss,
                             extra = {"accuracy_history": accuracy_history, "loss_history": loss_history})
        return TrainingOutput(model = model, metrics=results)

    def eval(self: Self,
             loader: DataLoader,
             model: nn.Module,
             criterion: nn.Module) -> EvalOutput:
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        model.to(gpu_or_cpu)
        model.eval()
        loss, acc = 0, 0
        total_samples = 0

        with no_grad():
            for data, target in loader:
                data, target = data.to(gpu_or_cpu), target.to(gpu_or_cpu)
                target = target.float().unsqueeze(1)
                output = model(data)
                loss += criterion(output, target).item()
                pred = (output) >= 0.5
                acc += pred.eq(target).sum().item()
                total_samples += target.size(0)
            loss /= len(loader)
            acc = float(acc) / total_samples

        output_dict = {"accuracy": acc, "loss": loss}
        return EvalOutput(**output_dict)