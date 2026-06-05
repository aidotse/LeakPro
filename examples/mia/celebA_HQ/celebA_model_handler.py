"""CelebA-HQ standard model handler — train() and eval() only.

Usage:
    from celebA_data_handler import CelebADataHandler
    from celebA_model_handler import CelebAModelHandler
    leakpro = LeakPro(CelebADataHandler, config_path, model_handler=CelebAModelHandler)
"""

import torch
from torch import no_grad, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from leakpro import AbstractInputHandler
from leakpro.schemas import EvalOutput, TrainingOutput
from leakpro.utils.device import get_device


class CelebAModelHandler(AbstractInputHandler, role="model"):
    """Standard training handler for CelebA-HQ. No dataset logic."""

    def train(
        self,
        dataloader: DataLoader,
        model: torch.nn.Module = None,
        criterion: torch.nn.Module = None,
        optimizer: optim.Optimizer = None,
        epochs: int = None,
    ) -> TrainingOutput:
        if epochs is None:
            raise ValueError("epochs not found in configs")

        gpu_or_cpu = get_device()
        model.to(gpu_or_cpu)

        accuracy_history, loss_history = [], []

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
                train_acc += pred.eq(labels.view_as(pred)).sum().item()
                total_samples += labels.size(0)
                train_loss += loss.item() * labels.size(0)

            avg_train_loss = train_loss / total_samples
            train_accuracy = train_acc / total_samples
            accuracy_history.append(train_accuracy)
            loss_history.append(avg_train_loss)

        model.to("cpu")
        results = EvalOutput(
            accuracy=train_accuracy, loss=avg_train_loss,
            extra={"accuracy_history": accuracy_history, "loss_history": loss_history},
        )
        return TrainingOutput(model=model, metrics=results)

    def eval(self, loader, model, criterion) -> EvalOutput:
        gpu_or_cpu = get_device()
        model.to(gpu_or_cpu)
        model.eval()
        loss, acc, total_samples = 0, 0, 0
        with no_grad():
            for data, target in loader:
                data, target = data.to(gpu_or_cpu), target.to(gpu_or_cpu)
                target = target.view(-1)
                output = model(data)
                loss += criterion(output, target).item() * target.size(0)
                acc += output.argmax(dim=1).eq(target).sum().item()
                total_samples += target.size(0)
        return EvalOutput(accuracy=float(acc) / total_samples, loss=loss / total_samples)
