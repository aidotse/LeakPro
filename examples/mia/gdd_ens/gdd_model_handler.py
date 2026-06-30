#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""GDD_ENS model handler — train() and eval() only (role="model").

train() uses the criterion / optimizer / epochs passed in, so shadow models trained from the
recorded metadata recipe match the target exactly.

Usage:
    from gdd_data_handler import GddDataHandler
    from gdd_model_handler import GddModelHandler
    leakpro = LeakPro(GddDataHandler, config_path, model_handler=GddModelHandler)
"""

from torch import cuda, device, nn, no_grad, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.schemas import EvalOutput, TrainingOutput


class GddModelHandler(AbstractInputHandler, role="model"):
    """Standard training handler for the GDD_ENS MLP. No dataset logic."""

    def train(
        self,
        dataloader: DataLoader,
        model: nn.Module = None,
        criterion: nn.Module = None,
        optimizer: optim.Optimizer = None,
        epochs: int = None,
    ) -> TrainingOutput:
        if epochs is None:
            raise ValueError("epochs not found in configs")

        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        model.to(gpu_or_cpu)

        accuracy_history, loss_history = [], []
        for epoch in range(epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
            for inputs, labels in pbar:
                inputs = inputs.to(gpu_or_cpu, non_blocking=True)
                labels = labels.to(gpu_or_cpu, non_blocking=True).long()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * labels.size(0)
                correct += outputs.argmax(dim=1).eq(labels).sum().item()
                total += labels.size(0)
                pbar.set_postfix(loss=f"{running_loss / total:.4f}", acc=f"{correct / total:.4f}")

            accuracy_history.append(correct / total)
            loss_history.append(running_loss / total)

        model.to("cpu")
        results = EvalOutput(
            accuracy=accuracy_history[-1], loss=loss_history[-1],
            extra={"accuracy_history": accuracy_history, "loss_history": loss_history},
        )
        return TrainingOutput(model=model, metrics=results)

    def eval(self, loader, model, criterion) -> EvalOutput:
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        model.to(gpu_or_cpu)
        model.eval()
        loss, correct, total = 0.0, 0, 0
        with no_grad():
            for data, target in loader:
                data = data.to(gpu_or_cpu)
                target = target.to(gpu_or_cpu).long().view(-1)
                output = model(data)
                loss += criterion(output, target).item() * target.size(0)
                correct += output.argmax(dim=1).eq(target).sum().item()
                total += target.size(0)
        return EvalOutput(accuracy=float(correct) / total, loss=loss / total)
