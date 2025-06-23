"""Train function that keeps the computational graph intact."""
from collections import OrderedDict

from torch import cuda, device
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
    losses_per_epoch = []
    for _ in range(epochs):
        total_loss = 0.0
        total_samples = 0
        for inputs, labels in data:
            inputs, labels = inputs.to(gpu_or_cpu, non_blocking=True), labels.to(gpu_or_cpu, non_blocking=True)
            outputs = patched_model(inputs, patched_model.parameters)
            loss = criterion(outputs, labels).sum()
            ## --tracking loss --##
            total_loss += loss.item()
            total_samples += labels.size(0)
            patched_model.parameters = optimizer.step(loss, patched_model.parameters)
        
        mean_loss = total_loss / total_samples 
        losses_per_epoch.append(mean_loss)
    model_delta = OrderedDict((name, param - param_origin)
                                            for ((name, param), (name_origin, param_origin))
                                            in zip(patched_model.parameters.items(),
                                                    OrderedDict(model.named_parameters()).items()))
    return list(model_delta.values()), losses_per_epoch
