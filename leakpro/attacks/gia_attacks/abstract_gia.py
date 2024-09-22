"""Module that contains the abstract class for constructing and performing a membership inference attack on a target."""

from abc import ABC, abstractmethod
from collections import OrderedDict

from torch.nn import Module
from torch.utils.data import DataLoader

from leakpro.fl_utils.gia_module_to_functional import MetaModule
from leakpro.metrics.attack_result import GIAResults
from leakpro.user_inputs.abstract_input_handler import AbstractInputHandler
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger

########################################################################################################################
# METRIC CLASS
########################################################################################################################


class AbstractGIA(ABC):
    """Interface to construct and perform a gradient inversion attack on a target model and dataset.

    This serves as a guideline for implementing a metric to be used for measuring the privacy leakage of a target model.
    """

    def __init__(
        self:Self,
        handler: AbstractInputHandler,
    )->None:
        """Initialize the AttackAbstract class.

        Args:
        ----
            handler (AbstractInputHandler): The input handler object.

        """
        # Add shared initialization between attacks here.. (model etc)
        # Add similarity tracking here..
        # Add image saving functions here..

        self.handler = handler
        self.train_indices = self.handler.train_indices
        self.criterion = self.handler.get_criterion()
        self.optimizer = self.handler.get_optimizer(model = None)
        self.batch_size = self.handler.target_model_metadata["batch_size"]
        self.epochs = self.handler.target_model_metadata["epochs"]

        # Compute the client update on the true client data
        self.client_loader = self.handler.get_dataloader(self.train_indices)
        client_gradient = self._get_pseudo_gradient(self.handler.target_model, self.client_loader)
        self.client_gradient = [p.detach() for p in client_gradient]

        frobenius_norm = sum([p.norm()**2 for p in client_gradient])**0.5
        assert frobenius_norm > 0, "Gradient norm from client update is zero."
        logger.info(f"Client gradient computed, norm is: {frobenius_norm}")


    def _get_pseudo_gradient(self:Self, init_model:Module, dataloader:DataLoader) -> list:
        """Wrapper for the train method provided by user."""
        # Train the model on the client data
        model = self.handler.train(dataloader,
                                    MetaModule(init_model),
                                    self.criterion,
                                    self.optimizer,
                                    self.epochs)

        # Compute the difference between the updated model and the initial model
        model_delta = OrderedDict((name, param - param_origin)
                                        for ((name, param), (name_origin, param_origin))
                                        in zip(model.parameters.items(),
                                                OrderedDict(init_model.named_parameters()).items()))
        return list(model_delta.values())

    @abstractmethod
    def _configure_attack(self:Self, configs:dict)->None:
        """Configure the attack.

        Args:
        ----
            configs (dict): The configurations for the attack.

        """
        pass


    @abstractmethod
    def description(self:Self) -> dict:
        """Return a description of the attack.

        Returns
        -------
        dict: A dictionary containing the reference, summary, and detailed description of the attack.

        """
        pass

    @abstractmethod
    def prepare_attack(self:Self) -> None:
        """Method that handles all computation related to the attack dataset."""
        pass

    @abstractmethod
    def run_attack(self:Self) -> GIAResults:
        """Run the metric on the target model and dataset. This method handles all the computations related to the audit dataset.

        Args:
        ----
            fpr_tolerance_rate_list (optional): List of FPR tolerance values that may be used by the threshold function
                to compute the attack threshold for the metric.

        Returns:
        -------
            Result(s) of the metric.

        """
        pass
