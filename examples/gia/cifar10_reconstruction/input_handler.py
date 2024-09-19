"""Module containing the class to handle the user input for the CIFAR10 dataset."""

from collections import OrderedDict

from leakpro.fl_utils.gia_module_to_functional import MetaModule
from leakpro.fl_utils.gia_optimizers import MetaOptimizer, MetaAdam, MetaSGD
from leakpro.user_inputs.abstract_input_handler import AbstractInputHandler
from torch import cuda, device
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import DataLoader


class Cifar10GIAInputHandler(AbstractInputHandler):
    """Class to handle the local training."""

    def __init__(self, configs: dict) -> None:
        super().__init__(configs = configs)
    
    def get_criterion(self)->None:
        """Set the CrossEntropyLoss for the model."""
        return CrossEntropyLoss()


    def train(self,
        dataloader: DataLoader,
        model: Module = None,
        criterion: Module = None,
        optimizer: MetaOptimizer = None,
        epochs: int = None,
    ) -> list:
        """Model training procedure for GIA.

        This training will create a computational graph through multiple steps, which is necessary
        for backpropagating to an input image.

        Requires a meta optimizer that performs step to a new set of parameters to keep a functioning
        graph.

        Training does not update the original model, but returns a norm of what the update would have been.
        """
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        self.target_model.to(gpu_or_cpu)
        patched_model = MetaModule(model)

        for _ in range(epochs):
            train_loss, train_acc = 0, 0
            for inputs, labels in dataloader:
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
                                                       OrderedDict(self.target_model.named_parameters()).items()))
        return list(model_delta.values())
