"""Run optuna to find best hyperparameters."""
from collections.abc import Generator
from typing import Union

import optuna
from torch import Tensor

from leakpro.attacks.gia_attacks.abstract_gia import AbstractGIA
from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.metrics.attack_result import MIAResult
from leakpro.utils.logger import logger


def optuna_optimal_hyperparameters(attack_object: Union[AbstractGIA, AbstractMIA], n_warmup_steps:int= 5,
                                   n_trials:int=50, direction:str="maximize"
                                   ) -> optuna.study.Study:
    """Find optimal hyperparameters for an attack object."""
    def objective(trial: optuna.trial.Trial) -> Tensor:
        attack_object.reset_attack()
        attack_object.suggest_parameters(trial)

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
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=n_warmup_steps)
    study = optuna.create_study(direction=direction, pruner=pruner)

    # Run optimization
    study.optimize(objective, n_trials=n_trials)

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
