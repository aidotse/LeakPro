#
# Copyright 2023-2026 AI Sweden
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Train function that keeps the computational graph intact."""

from collections import OrderedDict
from typing import Optional

import torch
from torch import Tensor, cuda
from torch.autograd import grad
from torch.nn import Module
from torch.utils.data import DataLoader

from leakpro.fl_utils.gia_module_to_functional import MetaModule
from leakpro.fl_utils.gia_optimizers import MetaOptimizer


def train(
    model: Module,
    data: DataLoader,
    optimizer: MetaOptimizer,
    criterion: Module,
    epochs: int,
    device: Optional[torch.device] = None,
    return_delta: bool = True,
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
    patched_model = MetaModule(model, device=device)
    outputs = None
    for _ in range(epochs):
        for inputs, labels in data:
            inputs, labels = (
                inputs.to(device, non_blocking=True),
                (labels.to(device, non_blocking=True) if isinstance(labels, Tensor) else labels),
            )
            outputs = patched_model(inputs, patched_model.parameters)
            loss = criterion(outputs, labels)  # .sum()
            patched_model.parameters = optimizer.step(loss, patched_model.parameters)
    model_delta = OrderedDict(
        (name, param - param_origin)
        for ((name, param), (name_origin, param_origin)) in zip(
            patched_model.parameters.items(), OrderedDict(model.named_parameters()).items()
        )
    )
    if return_delta:
        return list(model_delta.values())

    return list(patched_model.parameters.values())


def train_nostep(
    model: Module,
    data: DataLoader,
    optimizer: MetaOptimizer,  # noqa: ARG001
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
    outputs = None
    for _ in range(epochs):
        for inputs, labels in data:
            inputs, labels = (
                inputs.to(device, non_blocking=True),
                (labels.to(device, non_blocking=True) if isinstance(labels, Tensor) else labels),
            )
            outputs = model(inputs)
            loss = criterion(outputs, labels).sum()
            grads = grad(
                loss, list(model.parameters()), retain_graph=True, create_graph=True, only_inputs=True, allow_unused=True
            )
    return grads


def trainyolo(
    model: Module,
    data: DataLoader,
    optimizer: MetaOptimizer,
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
    patched_model = MetaModule(model)
    outputs = None
    for _ in range(epochs):
        for inputs, labels, _ in data:
            inputs, labels = (
                inputs.to(device, non_blocking=True),
                (labels.to(device, non_blocking=True) if isinstance(labels, Tensor) else labels),
            )
            outputs = patched_model(inputs, patched_model.parameters)
            loss = criterion(outputs, labels).sum()
            patched_model.parameters = optimizer.step(loss, patched_model.parameters)
    model_delta = OrderedDict(
        (name, param - param_origin)
        for ((name, param), (name_origin, param_origin)) in zip(
            patched_model.parameters.items(), OrderedDict(model.named_parameters()).items()
        )
    )
    return list(model_delta.values())
