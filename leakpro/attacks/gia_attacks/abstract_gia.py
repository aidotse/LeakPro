"""Module that contains the abstract class for constructing and performing a membership inference attack on a target."""

from abc import abstractmethod
from collections.abc import Generator
from copy import deepcopy
from typing import Callable, Optional

import numpy as np
import optuna
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from leakpro.attacks.attack_base import AbstractAttack
from leakpro.attacks.utils.hyperparameter_tuning.optuna import optuna_optimal_hyperparameters
from leakpro.fl_utils.model_utils import MedianPool2d
from leakpro.fl_utils.similarity_measurements import dataloaders_psnr, dataloaders_ssim_ignite
from leakpro.reporting.attack_result import GIAResults
from leakpro.schemas import OptunaConfig
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger
from leakpro.fl_utils.save_text import validate_tokens


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
                            median_pooling: bool, client_loader: DataLoader, reconstruction_loader: DataLoader,
                            image_data: bool=True
                            ) -> Generator[tuple[int, Tensor, GIAResults]]:
        """Generic attack loop for GIA's."""
        if isinstance(self.reconstruction, list):
            optimizer = torch.optim.Adam(reconstruction, lr=attack_lr)
            #optimizer = torch.optim.SGD(reconstruction, lr=attack_lr)

        else:
            optimizer = torch.optim.Adam([reconstruction], lr=attack_lr)
            
        # reduce LR every 1/3 of total iterations
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[at_iterations // 2.667,
                                                                        at_iterations // 1.6,
                                                                        at_iterations // 1.142], gamma=0.1)
        for i in range(at_iterations):
            print("itteration: ", i)
           
                    # loss function which does training and compares distance from reconstruction training to the real training.
            closure = gradient_closure(optimizer)
            
           
            previous_reconstruction_data = deepcopy(reconstruction_loader)
            #previous_optimizer_state = deepcopy(optimizer.state_dict())
            last_loss = optimizer.step(closure)
            loss_ = self.inference_closure()
            print("loss: ", np.round(last_loss.item(),5), ", best_loss: ", np.round(self.best_loss.item(),5), 
                  ", loss_: ", np.round(loss_.numpy(),5))
            tryout = 0
            while loss_ > last_loss and tryout<6:
                
                with torch.no_grad():
                        for j in range(len(reconstruction)):
                            reconstruction[j].copy_(previous_reconstruction_data.dataset[j].embedding)

                #optimizer.load_state_dict(previous_optimizer_state)
                    
                #print("reconstruction sum abstract: ", reconstruction[0].sum())
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr']/2
                closure = gradient_closure(optimizer)
                loss = optimizer.step(closure)
                loss_ = self.inference_closure()
                
                print("loss_: ", loss_, ", loss: ", loss)
                tryout += 1
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = attack_lr


            
            
            scheduler.step()
            if image_data:
                with torch.no_grad():
                    # force pixels to be in reasonable ranges
                    reconstruction.data = torch.max(
                        torch.min(reconstruction, (1 - data_mean) / data_std), -data_mean / data_std
                        )
                    if (i +1) % 500 == 0 and median_pooling:
                        reconstruction.data = MedianPool2d(kernel_size=3, stride=1, padding=1, same=False)(reconstruction)
            # Choose image who has given least loss
            if loss_ < self.best_loss:
                self.best_loss = loss_
                self.best_reconstruction = deepcopy(reconstruction_loader)
                self.best_reconstruction_round = i
                logger.info(f"New best loss: {loss_} on round: {i}")
                validate_tokens(client_loader,self.best_reconstruction,'best.npy')
            if i % 250 == 0:
                logger.info(f"Iteration {i}, loss {loss_}")
                yield i, i, None#dataloaders_ssim_ignite(client_loader, self.best_reconstruction), None

        ssim_score = dataloaders_ssim_ignite(client_loader, self.best_reconstruction)
        psnr_score = dataloaders_psnr(client_loader, self.best_reconstruction)
        gia_result = GIAResults(client_loader, self.best_reconstruction,
                          psnr_score=psnr_score, ssim_score=ssim_score,
                          data_mean=data_mean, data_std=data_std, config=configs)
        yield i, ssim_score, gia_result
