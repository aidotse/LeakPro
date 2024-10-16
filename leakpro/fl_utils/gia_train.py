"""Train function that keeps the computational graph intact."""
from collections import OrderedDict

from torch import cuda, device
from torch.nn import Module
from torch.utils.data import DataLoader

from leakpro.fl_utils.gia_module_to_functional import MetaModule
from leakpro.fl_utils.gia_optimizers import MetaOptimizer
from leakpro.utils.logger import logger


def train(
    model: Module,
    data: DataLoader,
    optimizer: MetaOptimizer,
    criterion: Module,
    epochs: int,
) -> list:
    """Model training procedure for GIA.

    This training will create a computational graph through multiple steps, which is necessary
    for backpropagating to an input image.

    Requires a meta optimizer that performs step to a new set of parameters to keep a functioning
    graph.

    Training does not update the original model, but returns a norm of what the update would have been.
    """
    gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
    model.to(gpu_or_cpu)
    patched_model = MetaModule(model)
    outputs = None
    for _ in range(epochs):
        train_loss, train_acc = 0, 0
        for inputs, labels in data:
            labels = labels.long()
            inputs, labels = inputs.to(gpu_or_cpu, non_blocking=True), labels.to(gpu_or_cpu, non_blocking=True)
            outputs = patched_model(inputs, patched_model.parameters)
            loss = criterion(outputs, labels).sum()
            pred = outputs.data.max(1, keepdim=True)[1]
            patched_model.parameters = optimizer.step(loss, patched_model.parameters)
            train_acc += pred.eq(labels.data.view_as(pred)).sum()
            train_loss += loss.item()
    model_delta = OrderedDict((name, param - param_origin)
                                            for ((name, param), (name_origin, param_origin))
                                            in zip(patched_model.parameters.items(),
                                                    OrderedDict(model.named_parameters()).items()))
    return list(model_delta.values())
