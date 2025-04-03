"""Module that contains the abstract class for constructing and performing a model inversion attack on a target."""

from abc import ABC, abstractmethod
from typing import Optional

import optuna
from optuna import Trial
from pydantic import BaseModel

from leakpro.attacks.utils.hyperparameter_tuning.optuna import optuna_optimal_hyperparameters
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.metrics.attack_result import AttackResult
from leakpro.schemas import OptunaConfig
from leakpro.signals.signal_extractor import PytorchModel
from leakpro.utils.import_helper import List, Self, Union
from leakpro.utils.logger import logger


class AbstractMINV(ABC):
    """Interface to construct and perform a model inversion attack on a target model and dataset.

    This serves as a guideline for implementing a metric to be used for measuring the privacy leakage of a target model.
    """

    # TODO: Think about what class attributes should be used and not used

    # Class attributes for sharing between the different attacks
    public_population = None
    public_population_size = None
    target_model = None
    target_dataset = None
    handler=None
    _initialized = False

    AttackConfig: type[BaseModel]  # Subclasses must define an attack config

    def __init__(
        self:Self,
        handler: AbstractInputHandler
    )->None:
        """Initialize the AttackAbstract class.

        Args:
        ----
            handler (AbstractInputHandler): The input handler object.

        """
        # These objects are shared and should be initialized only once
        if not AbstractMINV._initialized:
            AbstractMINV.target_model = PytorchModel(handler.target_model, handler.get_criterion())
            # Ensure that public_data_path is provided
            AbstractMINV.public_data_path = handler.configs.target.public_data_path
            AbstractMINV.handler = handler
            self._validate_shared_quantities()
            AbstractMINV._initialized = True

        # TODO: Class attributes initialized checks

    @classmethod
    def get_default_attack_config(cls) -> BaseModel:
        """Get the attack configuration.

        Returns
        -------
            BaseModel: The configuration of the attack.

        Raises
        ------
            ValueError: If the attack type is unknown.

        """
        return cls.Config()
    def _validate_config(self: Self, name: str, value: float, min_val: float, max_val: float) -> None:
        if not (min_val <= value <= (max_val if max_val is not None else value)):
            raise ValueError(f"{name} must be between {min_val} and {max_val}")

    def _validate_shared_quantities(self:Self) -> None:
        """Check if the shared quantities are initialized."""
        if AbstractMINV.target_model is None:
            raise ValueError("Target model is not initialized.")
        if AbstractMINV.public_population is None:
            raise ValueError("Public population is not initialized.")
        if AbstractMINV.public_population_size is None:
            raise ValueError("Public population size is not initialized.")



    def run_with_optuna(self:Self, optuna_config: Optional[OptunaConfig] = None) -> optuna.study.Study:
        """Finds optimal hyperparameters using optuna."""
        if optuna_config is None:
            # Use default valiues for config
            optuna_config = OptunaConfig()
        return optuna_optimal_hyperparameters(self, optuna_config)

    def suggest_parameters(self:Self, trial: Trial) -> BaseModel:
        """Update the given config with suggested parameters from the trial."""
        suggestions = {}
        for field_name, field in self.configs.model_fields.items():
            extra = field.json_schema_extra
            if extra is None or "optuna" not in extra:
                continue

            opt_variable = extra["optuna"]

            # Check if the field should be suggested (e.g. only when not online)
            enabled_if = opt_variable.get("enabled_if", None)
            if enabled_if is not None and not enabled_if(self.configs):
                continue

            param_type = opt_variable.get("type")
            if param_type == "float":
                suggestions[field_name] = trial.suggest_float(
                    field_name,
                    opt_variable["low"],
                    opt_variable["high"],
                    log=opt_variable.get("log", False)
                )
            elif param_type == "int":
                suggestions[field_name] = trial.suggest_int(
                    field_name,
                    opt_variable["low"],
                    opt_variable["high"]
                )
            elif param_type == "categorical":
                suggestions[field_name] = trial.suggest_categorical(
                    field_name,
                    opt_variable["choices"]
                )
            else:
                raise ValueError(f"Unsupported parameter type: {param_type}")

        return self.configs.model_copy(update=suggestions)

    def set_effective_optuna_metadata(self:Self, user_attack_config: dict) -> int:
        """Set the parameters to be optimized by optuna."""

        # Create default config
        self.optuna_params = 0
        attack_config = self.configs.model_fields.items() # Read out default configs for attack as a dict
        for field_name, field in attack_config:
            extra = field.json_schema_extra
            if extra is None or "optuna" not in extra:
                continue

            self.optuna_params += 1 # one more parameter to be optimized

            user_provided_value = user_attack_config.get(field_name)
            # remove the optuna dict to prevent the parameter to get optimized if the user has provided a value
            if user_provided_value is not None:
                self.configs.model_fields[field_name].json_schema_extra = None
                self.optuna_params -= 1 # remove one parameter going into optuna
                logger.info(f"User provided value for {field_name}, it won't be optimized by optuna.")
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
    def run_attack(self:Self) -> Union[AttackResult, List[AttackResult]]:
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
