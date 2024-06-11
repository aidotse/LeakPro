# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details.
"""Privacy evaluator that measures the inference risk."""
from typing import Any, List, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict

from leakpro.import_helper import Self
from leakpro.synthetic_data_attacks.anonymeter.neighbors.mixed_types_n_neighbors import mixed_type_n_neighbors
from leakpro.synthetic_data_attacks.anonymeter.stats.confidence import EvaluationResults
from leakpro.synthetic_data_attacks.anonymeter.utils import assert_x_in_bound


class InferenceGuesses(BaseModel):
    """Utility class to store guesses, secrets and count from inference attack.

    Parameters
    ----------
    guesses : pd.Series
        Attacker guesses for each of the targets.
    secrets : pd.Series
        Array with the true values of the secret for each of the targets.
    regression : bool
        Whether or not the attacker is trying to solve a classification or
        a regression task. The first case is suitable for categorical or
        discrete secrets, the second for numerical continuous ones.
    tolerance : float, default is 0.05
        Maximum value for the relative difference between target and secret
        for the inference to be considered correct.

    """

    model_config = ConfigDict(arbitrary_types_allowed = True)
    guesses: npt.NDArray
    secrets: npt.NDArray
    regression: bool
    tolerance: float = 0.05
    matches: Optional[npt.NDArray] = None
    count: Optional[int] = None

    def __init__(self: Self, **kwargs: Any) -> None: # noqa: ANN401
        super().__init__(**kwargs)
        #Assert input values
        assert len(self.guesses.shape) == 1
        assert self.guesses.shape[0] > 0
        assert self.guesses.shape == self.secrets.shape
        assert_x_in_bound(
            x = self.tolerance,
            x_name = "tolerance",
            inclusive_flag = True
        )

    def evaluate_inference_guesses(self: Self) -> Self:
        """Evaluate the success of an inference attack.

        The attack is successful if the attacker managed to make a correct guess.

        In case of regression problems, when the secret is a continuous variable,
        the guess is correct if the relative difference between guess and target
        is smaller than a given tolerance. In the case of categorical target
        variables, the inference is correct if the secrets are guessed exactly.

        Returns
        -------
        Self : InferenceGuesses
            Object containing inference guesses and successes (count).

        """
        #Set guesses and secrets
        guesses = self.guesses
        secrets = self.secrets
        #Calculate value matches (depending on self.regression)
        if self.regression:
            rel_abs_diff = np.abs(guesses - secrets) / (guesses + 1e-12)
            value_matches = rel_abs_diff <= self.tolerance
        else:
            value_matches = guesses == secrets
        #Calculate nan matches
        nan_matches = np.logical_and(pd.isna(guesses), pd.isna(secrets))
        #Calculate matches ('or' operator on value and nan matches)
        self.matches = np.logical_or(value_matches, nan_matches)
        #Calculate sucesses
        self.count = self.matches.sum()
        return self

class InferenceEvaluator(BaseModel):
    """Measure the inference risk created by a synthetic dataset.

    The attacker's goal is to use the synthetic dataset to learn about some
    (potentially all) attributes of a target record from the original database.
    The attacker has a partial knowledge of some attributes of the target
    record (the auxiliary information AUX) and uses a similarity score to find
    the synthetic record that matches best the AUX. The success of the attack
    (main attack) is compared to the baseline scenario of the trivial attacker,
    who guesses randomly (naive attack).

    An inference risk of 1 means that every single attacked secret
    could be in infered correctly by the auxiliary information.
    An inference risk of 0 means that all record inferences were incorrect.

    Parameters
    ----------
    ori : pd.DataFrame
        Dataframe with the target records whose secrets the attacker
        will try to guess. This is the private dataframe from which
        the synthetic one has been derived.
    syn : pd.DataFrame
        Dataframe with the synthetic records. It is assumed to be
        fully available to the attacker.
    aux_cols : List[str]
        Features (columns) of data that are given to the attacker as auxiliary information.
    secret : str
        Secret attribute of the targets that is unknown to the attacker.
        This is what the attacker will try to guess.
    regression : bool, optional
        Specifies whether the target of the inference attack is quantitative
        (regression = True) or categorical (regression = False). If None
        (default), the code will try to guess this by checking the type of
        the variable.
    n_attacks : int, default is 2000
        Number of attack attempts.
    confidence_level : float, default is 0.95
        Confidence level for the error bound calculation.
    n_jobs : int, default is -2
        The number of jobs to run in parallel for finding nearest k neighbor.
    main_guesses: InferenceGuesses, optional
        InferenceGuesses object holding main attack guesses, secrets and count.
        Parameter will be set in evaluate method.
    naive_guesses: InferenceGuesses, optional
        InferenceGuesses object holding naive attack guesses, secrets and count.
        Parameter will be set in evaluate method.
    results: EvaluationResults, optional
        EvaluationResults object containing the success rates for the various attacks.
        Parameter will be set in evaluate method.

    """

    model_config = ConfigDict(arbitrary_types_allowed = True)
    ori: pd.DataFrame
    syn: pd.DataFrame
    aux_cols: List[str]
    secret: str
    regression: Optional[bool] = None
    n_attacks: int = 2_000
    confidence_level: float = 0.95
    n_jobs: int = -2
    #Following parameters are set in evaluate method
    main_guesses: Optional[InferenceGuesses] = None
    naive_guesses: Optional[InferenceGuesses] = None
    results: Optional[EvaluationResults] = None

    def __init__(self: Self, **kwargs: Any) -> None: # noqa: ANN401
        super().__init__(**kwargs)
        #Assert input values
        if self.ori.shape[0]==0 or self.syn.shape[0]==0:
            raise ValueError("ori and syn must contain rows.")
        if len(self.ori.columns) <= 1:
            raise ValueError("ori must contain at least 2 columns.")
        if list(self.ori.columns) != list(self.syn.columns):
            raise ValueError("ori and syn columns must be equal.")
        if len(self.aux_cols)==0:
            raise ValueError("aux_cols must contain at least 1 element.")
        if not set(self.aux_cols).issubset(set(self.ori.columns)):
            raise ValueError("aux_cols not subset of ori.columns.")
        if self.secret not in self.ori.columns:
            raise ValueError("secret not in ori.columns.")
        if self.secret in self.aux_cols:
            raise ValueError("secret can't be included in aux_columns.")
        assert_x_in_bound(x=self.confidence_level, x_name="confidence_level")
        self.n_attacks = min(self.n_attacks, self.ori.shape[0])
        #Set self.regression if None
        if self.regression is None:
            self.regression = pd.api.types.is_numeric_dtype(self.ori[self.secret])

    def inference_attack(self: Self, *, naive: bool) -> InferenceGuesses:
        """Inference attack function.

        Function performs a main or naive attack (depending on naive flag), and
        returns a InferenceGuesses object withholding the guesses, secrets and successes (count) of the attack.
        """
        #Set variables
        n_attacks = self.n_attacks
        targets = self.ori.sample(n_attacks, replace=False)
        syn = self.syn
        aux_cols = self.aux_cols
        secret = self.secret
        #Generate guesses
        if naive:
            guesses = syn.sample(n_attacks)[secret]
        else:
            guesses_idx = mixed_type_n_neighbors(
                queries = targets[aux_cols],
                candidates = syn[aux_cols],
                n_neighbors = 1,
                n_jobs = self.n_jobs
            )
            guesses = syn.iloc[guesses_idx.reshape(-1)][secret]
        return InferenceGuesses(
            guesses=guesses.to_numpy(),
            secrets=targets[secret].to_numpy(),
            regression=self.regression
        ).evaluate_inference_guesses()

    def evaluate(self: Self) -> EvaluationResults:
        """Run the inference attacks (main and naive) and set and return results."""
        # Run main and naive attacks
        self.main_guesses = self.inference_attack(naive=False)
        self.naive_guesses = self.inference_attack(naive=True)
        # Set results
        self.results = EvaluationResults(
            n_total = self.n_attacks,
            n_main = self.main_guesses.count,
            n_naive = self.naive_guesses.count,
            confidence_level = self.confidence_level
        )
        return self.results
