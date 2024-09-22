"""Module containing the class to handle the user input for the CIFAR10 dataset."""

from torch.nn.modules import Module
from torch.optim.optimizer import Optimizer as Optimizer

from leakpro.user_inputs.abstract_input_handler import AbstractInputHandler
from torch import cuda, device
from torch.nn import  CrossEntropyLoss


class Cifar10GIAInputHandler(AbstractInputHandler):
    """Class to handle the local training."""

    def __init__(self, configs: dict) -> None:
        super().__init__(configs = configs)
    
    def get_criterion(self)->None:
        """Set the CrossEntropyLoss for the model."""
        return CrossEntropyLoss()
    
    def get_optimizer(self, model: Module) -> Optimizer:
        return None

    def train(self,
        dataloader,
        model,
        criterion,
        optimizer,
        epochs,
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

        for _ in range(epochs):
            train_loss, train_acc = 0, 0
            for inputs, labels in dataloader:
                labels = labels.long()
                inputs, labels = inputs.to(gpu_or_cpu, non_blocking=True), labels.to(gpu_or_cpu, non_blocking=True)
                outputs = model(inputs, model.parameters)
                loss = criterion(outputs, labels).sum()
                pred = outputs.data.max(1, keepdim=True)[1]
                model.parameters = optimizer.step(loss, model.parameters)
                train_acc += pred.eq(labels.data.view_as(pred)).sum()
                train_loss += loss.item()
        return model
