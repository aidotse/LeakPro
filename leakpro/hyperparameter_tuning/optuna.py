"""Run optuna to find best hyperparameters."""
from collections.abc import Generator
from typing import Literal

import optuna
from pydantic import BaseModel, Field
from torch import Tensor

from leakpro.attacks.attack_base import AbstractAttack
from leakpro.metrics.attack_result import MIAResult
from leakpro.utils.logger import logger
from leakpro.utils.seed import seed_everything


class OptunaConfig(BaseModel):
    """Configuration for the Optuna hyperparameter search."""

    seed: int = Field(default=1234, description="Random seed for reproducibility")
    n_trials: int = Field(default=50, description="Number of trials to find the optimal hyperparameters")
    direction: Literal["maximize", "minimize"] = Field("maximize", description="Direction of the optimization, minimize or maximize")
    pruner: optuna.pruners.BasePruner = Field(default=optuna.pruners.MedianPruner(n_warmup_steps=5), description="Number of steps before pruning of experiments will be available")  # noqa: E501

    class Config:
        """Configuration for OptunaConfig to enable arbitrary type handling."""

        arbitrary_types_allowed = True

def optuna_optimal_hyperparameters(attack_object: AbstractAttack, optuna_config: OptunaConfig = None) -> optuna.study.Study:
    """Find optimal hyperparameters for an attack object.

    Args:
    ----
            attack_object (Union[AbstractGIA, AbstractMIA]): Attack object to find optimal hyperparameters for.
            optuna_config (OptunaConfig): configureable settings for optuna

    """
    def objective(trial: optuna.trial.Trial) -> Tensor:
        # Suggest hyperparameters
        new_config = attack_object.suggest_parameters(trial)
        # Reset attack to apply new hyperparameters
        attack_object.reset_attack(new_config)
        seed_everything(optuna_config.seed)
        result = attack_object.run_attack()
        if isinstance(result, Generator):
            for step, intermediary_results, result_object in result:
                trial.report(intermediary_results, step)

                if trial.should_prune():
                    raise optuna.TrialPruned()
                # save results if not pruned
                if result_object is not None:
                    result_object.save(name="optuna", path="./leakpro_output/results", config=attack_object.get_configs())
                    return intermediary_results
        elif isinstance(result, MIAResult):
            # Retrieve configuration and result metric
            roc_auc = result.roc_auc

            trial.set_user_attr("config", new_config)
            trial.set_user_attr("roc_auc", roc_auc)

            # Optionally print the details for immediate feedback
            logger.info(f"Trial {trial.number} - Config: {new_config} - roc_auc: {roc_auc}")

            # MIA cannot be used with pruning as we need the final result to be computed
            return result.roc_auc  # add something reasonable to optimize toward here
        return None

    # Define the pruner and study
    pruner = optuna_config.pruner
    study = optuna.create_study(direction=optuna_config.direction, pruner=pruner)

    # Run optimization
    study.optimize(objective, n_trials=optuna_config.n_trials)

    # Display and save the results
    logger.info(f"Best hyperparameters: {study.best_params}")
    logger.info(f"Best optimized value: {study.best_value}")

    f_results_file = attack_object.attack_folder_path + "/optuna_results.txt"
    with open(f_results_file, "w") as f:
        f.write("Best hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"{key}: {value}\n")

    logger.info(f"Results saved to {f_results_file}")
    best_config = attack_object.configs.copy(update=study.best_params)
    return best_config  # noqa: RET504
