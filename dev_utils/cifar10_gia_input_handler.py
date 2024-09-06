"""Module containing the class to handle the user input for the CIFAR10 dataset."""

import logging
from collections import OrderedDict

from dev_utils.data_modules import CifarModule
from leakpro.fl_utils.gia_module_to_functional import MetaModule
from leakpro.fl_utils.gia_optimizers import MetaOptimizer
from leakpro.import_helper import Self
from leakpro.user_inputs.abstract_gia_input_handler import AbstractGIAInputHandler
from torch import cuda, device
from torch.nn import Module
from torch.utils.data import DataLoader


class Cifar10GIAInputHandler(AbstractGIAInputHandler):
    """Class to handle the user input for the CIFAR10 dataset."""

    def __init__(self:Self, configs: dict, logger:logging.Logger, target_model: Module) -> None:
        self.data_module = CifarModule(batch_size=configs["audit"]["gia_settings"]["client_batch_size"])
        super().__init__(configs, logger, target_model, self.data_module)
        self.criterion = self.get_criterion()

    def train(
        self: Self,
        data: DataLoader = None,
        optimizer: MetaOptimizer = None,
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
        patched_model = MetaModule(self.target_model)

        outputs = None
        epochs = self.configs["audit"]["gia_settings"]["epochs"]
        for _ in range(epochs):
            train_loss, train_acc = 0, 0
            for inputs, labels in data:
                labels = labels.long()
                inputs, labels = inputs.to(gpu_or_cpu, non_blocking=True), labels.to(gpu_or_cpu, non_blocking=True)
                outputs = patched_model(inputs, patched_model.parameters)
                loss = self.criterion(outputs, labels).sum()
                pred = outputs.data.max(1, keepdim=True)[1]
                patched_model.parameters = optimizer.step(loss, patched_model.parameters)
                train_acc += pred.eq(labels.data.view_as(pred)).sum()
                train_loss += loss.item()
        model_delta = OrderedDict((name, param - param_origin)
                                                for ((name, param), (name_origin, param_origin))
                                                in zip(patched_model.parameters.items(),
                                                       OrderedDict(self.target_model.named_parameters()).items()))
        return list(model_delta.values())
