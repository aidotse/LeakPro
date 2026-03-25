from typing import Optional

from torch import Tensor, cuda, device
import torch
from torch.nn import CrossEntropyLoss, Module
from torch.utils.data import DataLoader


def train(
    model: Module,
    data: DataLoader,
    optimizer: Module,
    criterion: Module,
    epochs: int,
    device: Optional[torch.device] = None,
) -> list:
    """Model training procedure for GIA.

    This training will create a computational graph through multiple steps, which is necessary
    for backpropagating to an input image.

    Requires a meta optimizer that performs step to a new set of parameters to keep a functioning
    graph.

    Training does not update the original model, but returns a norm of what the update would have been.
    """
    if device is None:
        device = torch.device("cuda" if cuda.is_available() else "cpu")
    model.to(device)

    for _ in range(epochs):
        for inputs, labels in data:
            inputs, labels = (
                inputs.to(device, non_blocking=True),
                (labels.to(device, non_blocking=True) if isinstance(labels, Tensor) else labels),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model
