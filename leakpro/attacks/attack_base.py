"""Run optuna to find best hyperparameters."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Self

import optuna


class AbstractAttack(ABC):
    """Abstract attack template for attack objects."""

    @abstractmethod
    def run_with_optuna(self:Self, optuna_config: Optional[dataclass]) -> optuna.study.Study:
        """Fins optimal hyperparameters using optuna."""
        pass

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

