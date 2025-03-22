"""Geiping, Jonas, et al. "Inverting gradients-how easy is it to break privacy in federated learning?."."""
import os
from collections.abc import Generator
from copy import deepcopy

import torch
from optuna.trial import Trial
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from leakpro.attacks.gia_attacks.abstract_gia import AbstractGIA
from leakpro.attacks.gia_attacks.utils import InvertingConfig
from leakpro.fl_utils.data_utils import get_at_images
from leakpro.fl_utils.similarity_measurements import cosine_similarity_weights, total_variation
from leakpro.metrics.attack_result import GIAResults
from leakpro.utils.import_helper import Callable, Self
from leakpro.utils.logger import logger


class InvertingGradients(AbstractGIA):
    """Gradient inversion attack by Geiping et al."""

    def __init__(self: Self, model: Module, client_loader: DataLoader, train_fn: Callable,
                 data_mean: Tensor, data_std: Tensor, configs: InvertingConfig) -> None:
        super().__init__()
        self.original_model = model
        self.model = deepcopy(self.original_model)
        self.client_loader = client_loader
        self.train_fn = train_fn
        self.data_mean = data_mean
        self.data_std = data_std
        self.configs = configs
        self.best_loss = float("inf")
        self.best_reconstruction = None
        self.best_reconstruction_round = None
        # required for optuna to save the best hyperparameters
        self.attack_folder_path = "leakpro_output/attacks/inverting_grad"
        os.makedirs(self.attack_folder_path, exist_ok=True)

        logger.info("Inverting gradient initialized.")
        self.prepare_attack()

    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "Inverting gradients"
        reference_str = """Geiping, Jonas, et al. Inverting gradients-how easy is it to
            break privacy in federated learning? Neurips, 2020."""
        summary_str = ""
        detailed_str = ""
        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }

    def prepare_attack(self:Self) -> None:
        """Prepare the attack.

        Args:
        ----
            self (Self): The instance of the class.

        Returns:
        -------
            None

        """
        self.model.eval()
        self.reconstruction, self.reconstruction_loader = get_at_images(self.client_loader)
        self.reconstruction.requires_grad = True
        client_gradient = self.train_fn(self.model, self.client_loader, self.configs.optimizer,
                                        self.configs.criterion, self.configs.epochs)
        self.client_gradient = [p.detach() for p in client_gradient]

    def run_attack(self:Self) -> Generator[tuple[int, Tensor, GIAResults]]:
        """Run the attack and return the combined metric result.

        Returns
        -------
            GIAResults: Container for results on GIA attacks.

        """
        return self.generic_attack_loop(self.configs, self.gradient_closure, self.configs.at_iterations, self.reconstruction,
                                self.data_mean, self.data_std, self.configs.attack_lr, self.configs.median_pooling,
                                self.client_loader, self.reconstruction_loader)


    def gradient_closure(self: Self, optimizer: torch.optim.Optimizer) -> Callable:
        """Returns a closure function that performs a gradient descent step.

        The closure function computes the gradients, calculates the reconstruction loss,
        adds a total variation regularization term, and then performs backpropagation.
        """
        def closure() -> torch.Tensor:
            """Computes the reconstruction loss and performs backpropagation.

            This function zeroes out the gradients of the optimizer and the model,
            computes the gradient and reconstruction loss, logs the reconstruction loss,
            optionally adds a total variation term, performs backpropagation, and optionally
            modifies the gradient of the input image.

            Returns
            -------
                torch.Tensor: The reconstruction loss.

            """
            optimizer.zero_grad()
            self.model.zero_grad()

            gradient = self.train_fn(self.model, self.reconstruction_loader, self.configs.optimizer,
                                     self.configs.criterion, self.configs.epochs)
            rec_loss = cosine_similarity_weights(gradient, self.client_gradient, self.configs.top10norms)

            # Add the TV loss term to penalize large variations between pixels, encouraging smoother images.
            rec_loss += (self.configs.tv_reg * total_variation(self.reconstruction))
            rec_loss.backward()
            self.reconstruction.grad.sign_()
            return rec_loss
        return closure

    def _configure_attack(self: Self, configs: dict) -> None:
        pass

    def suggest_parameters(self: Self, trial: Trial) -> None:
        """Suggest parameters to chose and range for optimization for the Inverting Gradient attack."""
        total_variation = trial.suggest_float("total_variation", 1e-6, 1e-1, log=True)
        self.configs.tv_reg = total_variation

    def reset_attack(self: Self, new_config:dict) -> None:  # noqa: ARG002
        """Reset attack to initial state."""
        self.best_loss = float("inf")
        self.best_reconstruction = None
        self.best_reconstruction_round = None
        self.model = deepcopy(self.original_model)
        self.prepare_attack()
        logger.info("Inverting attack reset to initial state.")
