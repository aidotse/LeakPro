"""A module that implements the HopSkipJump attack for membership inference."""

from typing import Literal, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.metrics.attack_result import MIAResult
from leakpro.signals.signal import HopSkipJumpDistance
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class HSJConfig(BaseModel):
    """Configuration for the RMIA attack."""

    attack_data_fraction: float = Field(default=0.1, ge = 0.0, le=1.0, description="Fraction of the data to use for the attack")
    norm: Union[int, float] = Field(default=2, description="The norm to use for the attack. Must be one of [1, 2, np.inf]")
    initial_num_evals: int = Field(default=100, ge=1, le=1000, description="The initial number of evaluations")
    max_num_evals: int = Field(default=10000, ge=1, description="The maximum number of evaluations")
    num_iterations: int = Field(default=100, ge=1, description="The number of iterations")
    gamma: float = Field(default=1.0, ge=0.0, description="The gamma value")
    constraint: Literal[1,2] = Field(default=2, description="The constraint value must be 1 or 2")
    batch_size: int = Field(default=128, ge=1, description="The batch size")
    epsilon_threshold: float = Field(default=1e-6, ge=0.0, le=0.001, description="The epsilon threshold")

    @field_validator("norm", mode="before")
    @classmethod
    def validate_norm(cls, v: Union[int, float]) -> Union[int, float]:
        """Validate the norm value.

        Args:
            v (Union[int, float]): The norm value to validate.

        Returns:
            Union[int, float]: The validated norm value.

        Raises:
            ValueError: If the norm value is not one of [1, 2, np.inf].

        """
        if v not in {1, 2, np.inf}:
            raise ValueError("Norm must be one of [1, 2, np.inf]")
        return v

    @model_validator(mode="after")
    @classmethod
    def check_max_greater_than_initial(cls, values : dict) -> dict:
        """Ensure max_num_evals > initial_num_evals."""
        if values.max_num_evals <= values.initial_num_evals:
            raise ValueError("max_num_evals must be greater than initial_num_evals")
        return values

    model_config = {"arbitrary_types_allowed": True}  # Pydantic v2 config to allow `np.inf`


class AttackHopSkipJump(AbstractMIA):  # noqa: D101

    AttackConfig = HSJConfig

    def __init__(self: Self,
                 handler: AbstractInputHandler,
                 configs: dict
                ) -> None:
        """Initialize the HopSkipJump class.

        Args:
        ----
            handler (AbstractInputHandler): The input handler object.
            configs (dict): A dictionary containing the attack loss_traj configurations.

        """
        logger.info("Configuring label only attack")
        self.configs = HSJConfig(**configs)
        super().__init__(handler)

        # Assign the configuration parameters to the object
        for key, value in self.configs.model_dump().items():
            setattr(self, key, value)

        self.signal = HopSkipJumpDistance()

        self.y_target = None
        self.image_target = None
        self.verbose = configs.get("verbose", True)
        self.stepsize_search = "geometric_progression"

    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "Label-Only Membership Inference Attacks"
        reference_str = "Christopher A. Choquette-Choo, Florian Tramer, Nicholas Carlini and Nicolas Papernot\
            Label-Only Membership Inference Attacks. (2020)."
        summary_str = "This attack is one of the introduce black-box membership inference attacks in the paper."
        detailed_str = "The distance attack executed based on the estimation of the distance between the input \
                        data and the decision boundary of a shadow model of the target model. \
                        The attack aims to use this distance as a signal to infer membership."
        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }

    def prepare_attack(self:Self) -> None:
        """Prepare the attack by loading the shadow model and target model.

        Args:
        ----
            self (Self): The instance of the class.

        Returns:
        -------
            None

        """
        logger.info("Preparing the data for Hop Skip Jump attack")

        in_member_indices = self.audit_dataset["in_members"]
        out_member_indices = self.audit_dataset["out_members"]

        audit_in_member_indicies = np.random.choice(in_member_indices,
                                              int(len(in_member_indices) * self.attack_data_fraction),
                                                replace=False)

        audit_out_member_indicies = np.random.choice(out_member_indices,
                                                int(len(out_member_indices) * self.attack_data_fraction),
                                                replace=False)
        audit_indices = np.concatenate((audit_in_member_indicies, audit_out_member_indicies))

        self.attack_dataloader = self.handler.get_dataloader(audit_indices, batch_size=self.batch_size)



    def run_attack(self:Self) -> MIAResult:
        """Run the attack and return the combined metric result.

        Returns
        -------
            MIAResult: The Result containing predicted labels, true labels,
            predictions probabilities, and signal values.

        """

        logger.info("Running Hop Skip Jump distance attack")

        # compute the perturbation distances of the attack data from the target model decision boundary
        _ , perturbation_distances = self.signal(self.target_model,
                                                    self.attack_dataloader,
                                                    self.norm,
                                                    self.y_target,
                                                    self.image_target,
                                                    self.initial_num_evals,
                                                    self.max_num_evals,
                                                    self.stepsize_search,
                                                    self.num_iterations,
                                                    self.gamma,
                                                    self.constraint,
                                                    self.batch_size,
                                                    self.verbose
                                                    )

        # create thresholds
        min_signal_val = np.min(perturbation_distances)
        max_signal_val = np.max(perturbation_distances)
        thresholds = np.linspace(min_signal_val, max_signal_val, 1000)
        num_threshold = len(thresholds)

        # compute the signals for the in-members and out-members
        member_signals = (np.array(perturbation_distances).reshape(-1, 1).repeat(num_threshold, 1).T)

        member_preds = np.greater(member_signals, thresholds[:, np.newaxis])

        # set true labels for being in the training dataset
        true_labels = np.concatenate(
            [
                np.ones(int(len(self.attack_dataloader.dataset)/2)),
                np.zeros(int(len(self.attack_dataloader.dataset)/2)),
            ]
        )

        # compute ROC, TP, TN etc
        return MIAResult(
            predicted_labels=member_preds,
            true_labels=true_labels,
            predictions_proba=None,
            signal_values= perturbation_distances,
        )
