"""Module containing the class to handle the user input for the CIFAR10 dataset."""

import logging
from collections import OrderedDict
from copy import deepcopy

import torch
from torch import cuda, device
from torch.utils.data import DataLoader
from tqdm import tqdm

from leakpro.fl_utils.gia_optimizers import MetaAdam, MetaOptimizer, MetaSGD
from leakpro.fl_utils.gia_train import MetaModule
from leakpro.import_helper import Self
from leakpro.user_inputs.abstract_gia_input_handler import AbstractGIAInputHandler


class Cifar10GIAInputHandler(AbstractGIAInputHandler):
    """Class to handle the user input for the CIFAR10 dataset."""

    def __init__(self:Self, configs: dict, logger:logging.Logger, client_data: DataLoader, target_model: torch.nn.Module) -> None:
        super().__init__(configs, logger, client_data, target_model)
        self.criterion = self.get_criterion()

    def get_criterion(self:Self)->None:
        """Set the CrossEntropyLoss for the model."""
        return torch.nn.CrossEntropyLoss()

    def get_optimizer(self: Self) -> MetaOptimizer:
        """Set the optimizer for the model."""
        optimizer = self.configs["audit"]["gia_settings"]["optimizer"]
        lr = self.configs["audit"]["gia_settings"]["learning_rate"]
        if optimizer == "SGD":
            return MetaSGD(lr=lr)
        if optimizer == "Adam":
            return MetaAdam(lr=lr)
        raise ValueError(f"Optimizer '{optimizer}' not found. Please check the optimizer settings.")

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
        patched_model = MetaModule(deepcopy(self.target_model))

        patched_model_origin = {name: param.clone() for name, param in patched_model.parameters.items()}
        outputs = None
        epochs = self.configs["audit"]["gia_settings"]["epochs"]
        for epoch in range(epochs):
            train_loss, train_acc = 0, 0
            for inputs, labels in tqdm(data, desc=f"Epoch {epoch+1}/{epochs}"):
                labels = labels.long()
                inputs, labels = inputs.to(gpu_or_cpu, non_blocking=True), labels.to(gpu_or_cpu, non_blocking=True)
                outputs = patched_model(inputs, patched_model.parameters)
                loss = self.criterion(outputs, labels).sum()
                pred = outputs.data.max(1, keepdim=True)[1]
                patched_model.parameters = optimizer.step(loss, patched_model.parameters)
                train_acc += pred.eq(labels.data.view_as(pred)).sum()
                train_loss += loss.item()
            log_train_str = (
                    f"Epoch: {epoch+1}/{epochs} | Train Loss: {train_loss/len(data):.8f} | "
                    f"Train Acc: {float(train_acc)/len(data):.8f}"
                    )
            self.logger.info(log_train_str)

        model_delta = OrderedDict((name, param - param_origin)
                                                for ((name, param), (name_origin, param_origin))
                                                in zip(patched_model.parameters.items(), patched_model_origin.items()))
        return list(model_delta.values())
