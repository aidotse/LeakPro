"""Run optuna to find best hyperparameters."""
from collections.abc import Generator

import optuna
from torch import Tensor

from leakpro.attacks.attack_base import AbstractAttack
from leakpro.reporting.mia_result import MIAResult
from leakpro.schemas import OptunaConfig
from leakpro.utils.logger import logger
from leakpro.utils.seed import seed_everything


def optuna_optimal_hyperparameters(attack_object: AbstractAttack, optuna_config: OptunaConfig = None) -> optuna.study.Study:
    """Find optimal hyperparameters for an attack object.

    Args:
    ----
            attack_object (Union[AbstractGIA, AbstractMIA]): Attack object to find optimal hyperparameters for.
            optuna_config (OptunaConfig): configureable settings for optuna

    Returns:
    -------
            optuna.study.Study: Optuna study object containing the results of the optimization.

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
                # check every 3000 results
                if step%optuna_config.check_interval==0:
                    trial.report(intermediary_results, step)

                    if trial.should_prune():
                        raise optuna.TrialPruned()
                    # save results if not pruned
                if result_object is not None:
                    result_object.save(name="optuna"+"trial"+str(trial.number), path="./leakpro_output/results",
                                    config=attack_object.get_configs())
                    return intermediary_results
        elif isinstance(result, MIAResult):
            # Retrieve configuration and result metric
            obj_val = optuna_config.objective(result)

            trial.set_user_attr("config", new_config)
            trial.set_user_attr("objective_fn val", obj_val)

            # Optionally print the details for immediate feedback
            logger.info(f"Trial {trial.number} - Config: {new_config} - objective_fn val: {obj_val}")

            # MIA cannot be used with pruning as we need the final result to be computed
            return obj_val
        return None

    # Define the pruner and study
    pruner = optuna_config.pruner
    study = optuna.create_study(direction=optuna_config.direction, pruner=pruner)
    # Run optimization
    study.optimize(objective, n_trials=optuna_config.n_trials)

    # Display and save the results
    logger.info(f"Best hyperparameters: {study.best_params}")
    logger.info(f"Best optimized value: {study.best_value}")

    f_results_file = attack_object.attack_cache_folder_path + "/optuna_results.txt"
    with open(f_results_file, "w") as f:
        f.write("Best hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"{key}: {value}\n")

    logger.info(f"Results saved to {f_results_file}")

    # Return the study
    return study
