"""Module that contains the abstract class for constructing and performing a membership inference attack on a target."""

import copy
import os
from abc import abstractmethod
from collections.abc import Generator
from copy import deepcopy
from typing import Callable, Optional

import optuna
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from leakpro.attacks.attack_base import AbstractAttack
from leakpro.attacks.utils.hyperparameter_tuning.optuna import optuna_optimal_hyperparameters
from leakpro.fl_utils.model_utils import MedianPool2d
from leakpro.fl_utils.similarity_measurements import dataloaders_psnr, dataloaders_ssim_ignite
from leakpro.metrics.attack_result import GIAResults
from leakpro.schemas import OptunaConfig
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class AbstractGIA(AbstractAttack):
    """Interface to construct and perform a gradient inversion attack on a target model and dataset.

    This serves as a guideline for implementing a metric to be used for measuring the privacy leakage of a target model.
    """

    def __init__(  # noqa: B027
        self:Self,
    )->None:
        """Initialize the AttackAbstract class.

        Args:
        ----
            handler (AbstractInputHandler): The input handler object.

        """
        pass

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
    def run_attack(self:Self) -> Generator[tuple[int, Tensor, GIAResults]]:
        """Runs GIA attack.

        Returns
        -------
            Generator with intermediary results to allow for lazy evaluation during
            hyperparameter turning, and final GiaResults.

        """
        pass

    @abstractmethod
    def reset_attack(self: Self) -> None:
        """Reset attack to its initial state."""
        pass

    @abstractmethod
    def suggest_parameters(self: Self, trial: optuna.trial.Trial) -> None:
        """Apply and suggest new hyperparameters for the attack using optuna trial."""
        pass

    def run_with_optuna(self:Self, optuna_config: Optional[OptunaConfig] = None) -> optuna.study.Study:
        """Fins optimal hyperparameters using optuna."""
        if optuna_config is None:
            optuna_config = OptunaConfig()
        optuna_optimal_hyperparameters(self, optuna_config)

    def generic_attack_loop(self: Self, configs:dict, gradient_closure: Callable, at_iterations: int,
                            reconstruction: Tensor, data_mean: Tensor, data_std: Tensor, attack_lr: float,
                            median_pooling: bool, client_loader: DataLoader, reconstruction_loader: DataLoader
                            ) -> Generator[tuple[int, Tensor, GIAResults]]:
        """Generic attack loop for GIA's."""
        optimizer = torch.optim.Adam([reconstruction], lr=attack_lr)
        # reduce LR every 1/3 of total iterations
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[at_iterations // 2.667,
                                                                        at_iterations // 1.6,
                                                                        at_iterations // 1.142], gamma=0.1)
        try:
            for i in range(at_iterations):
                # loss function which does training and compares distance from reconstruction training to the real training.
                closure = gradient_closure(optimizer)
                loss = optimizer.step(closure)
                scheduler.step()
                with torch.no_grad():
                    # force pixels to be in reasonable ranges
                    reconstruction.data = torch.max(
                        torch.min(reconstruction, (1 - data_mean) / data_std), -data_mean / data_std
                        )
                    if (i +1) % 500 == 0 and median_pooling:
                        reconstruction.data = MedianPool2d(kernel_size=3, stride=1, padding=1, same=False)(reconstruction)
                # Choose image who has given least loss
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.best_reconstruction = deepcopy(reconstruction_loader)
                    self.best_reconstruction_round = i
                    logger.info(f"New best loss: {loss} on round: {i}")
                if i % 250 == 0:
                    logger.info(f"Iteration {i}, loss {loss}")
                    ssim = dataloaders_ssim_ignite(client_loader, self.best_reconstruction)
                    logger.info(f"ssim: {ssim}")
                    yield i, ssim, None
        except Exception as e:
            logger.info(f"Attack stopped due to {e}. \
                        Saving results.")
        ssim_score = dataloaders_ssim_ignite(client_loader, self.best_reconstruction)
        psnr_score = dataloaders_psnr(client_loader, self.best_reconstruction)
        gia_result = GIAResults(client_loader, self.best_reconstruction,
                          psnr_score=psnr_score, ssim_score=ssim_score,
                          data_mean=data_mean, data_std=data_std, config=configs)
        yield i, ssim_score, gia_result
