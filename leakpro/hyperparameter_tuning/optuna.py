"""Run optuna to find best hyperparameters."""
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Optional

import optuna
from torch import Tensor

from leakpro.attacks.attack_base import AbstractAttack
from leakpro.metrics.attack_result import MIAResult
from leakpro.utils.logger import logger
from leakpro.utils.seed import seed_everything


@dataclass
class OptunaConfig:
    """Config for configurable stuff in optuna."""

    # pruner (optuna.pruners.BasePruner): Number of steps before pruning of experiments will be available.
    pruner: optuna.pruners.BasePruner = field(default_factory=lambda: optuna.pruners.MedianPruner(n_warmup_steps=5))
    # n_trials (int): Number of trials to find the optimal hyperparameters.
    n_trials: int = 50
    # direction (str): Direction of the optimization, minimize or maximize, depending on the optuna objective.
    direction: str = "maximize"
    # seed (int): Random seed to run the attack from.
    seed:int = 1234

def optuna_optimal_hyperparameters(attack_object: AbstractAttack, optuna_config: OptunaConfig
                                   ) -> optuna.study.Study:
    """Find optimal hyperparameters for an attack object.

    Args:
    ----
            attack_object (Union[AbstractGIA, AbstractMIA]): Attack object to find optimal hyperparameters for.
            optuna_config (OptunaConfig): configureable settings for optuna

    """
    def objective(trial: optuna.trial.Trial) -> Tensor:
        attack_object.reset_attack()
        attack_object.suggest_parameters(trial)
        seed_everything(optuna_config.seed)
        result = attack_object.run_attack()
        if isinstance(result, Generator):
            for step, intermediary_results, result_object in result:
                trial.report(intermediary_results, step)

                if trial.should_prune():
                    break
                # save results if not pruned
                if result_object is not None:
                    result_object.save(name="optuna", path="./leakpro_output/results", config=attack_object.get_configs())
                    return intermediary_results
        elif isinstance(result, MIAResult):
            return result.accuracy # add something reasonable to optimize toward here
        return None

    # Define the pruner and study
    pruner = optuna_config.pruner
    study = optuna.create_study(direction=optuna_config.direction, pruner=pruner)

    # Run optimization
    study.optimize(objective, n_trials=optuna_config.n_trials)

    # Display and save the results
    logger.info(f"Best hyperparameters: {study.best_params}")
    logger.info(f"Best optimized value: {study.best_value}")

    results_file = "optuna_results.txt"
    with open(results_file, "w") as f:
        f.write("Best hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"{key}: {value}\n")

    logger.info(f"Results saved to {results_file}")
    return study
