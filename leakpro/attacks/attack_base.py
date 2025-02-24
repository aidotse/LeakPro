"""Run optuna to find best hyperparameters."""
from abc import ABC, abstractmethod
from typing import Optional, Self

import optuna

from leakpro.hyperparameter_tuning.optuna import OptunaConfig, optuna_optimal_hyperparameters


class AbstractAttack(ABC):
    """Abstract attack template for attack objects."""

    def run_with_optuna(self:Self, optuna_config: Optional[OptunaConfig] = None) -> optuna.study.Study:
        """Finds optimal hyperparameters using optuna."""
        if optuna_config is None:
            # Use default valiues for config
            optuna_config = OptunaConfig()
        optuna_optimal_hyperparameters(self, optuna_config)

    def get_configs(self: Self) -> dict:
        """Return configs used for attack."""
        return self.configs

    @abstractmethod
    def run_attack() -> None:
        """Run the attack on the target model and dataset.

        This method is implemented by subclasses (e.g., GIA and MIA attacks),
        each of which provides specific behavior and results.

        Returns
        -------
        Depends on the subclass implementation.

        """
        pass

