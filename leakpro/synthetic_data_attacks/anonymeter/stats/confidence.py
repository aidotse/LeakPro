# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details.
"""Functions for estimating rates and errors in privacy attacks."""
from math import sqrt
from typing import List, Optional, Tuple, Union

from leakpro.import_helper import Self
from leakpro.synthetic_data_attacks.anonymeter.utils import assert_x_in_bound
from pydantic import BaseModel
from scipy.stats import norm


def get_confidence_interval(*, rate: float, error: float) -> Tuple[float, float]:
    """Function will return lower and upper bound (confidence interval) for provided rate and error.

    Values are adjusted to be within [0,1] interval.
    """
    bound_lower = min(max(rate - error, 0.0), 1.0)
    bound_upper = min(max(rate + error, 0.0), 1.0)
    return bound_lower, bound_upper

class SuccessRate(BaseModel):
    """Estimate of the success rate of a privacy attack.

    Parameters
    ----------
    rate : float
        Estimate of the success rate of the attack.
    error : float
        Error on the estimate.
    ci : (float, float)
        Confidence interval of estimate.
        Interval will be constructed on initialization.

    """

    rate: float
    error: float
    ci: Optional[Tuple[float, float]] = None

    def __init__(self: Self, **kwargs: Union[float, Optional[Tuple[float, float]]]) -> None:
        super().__init__(**kwargs)
        #Assert input values
        assert_x_in_bound(x=self.rate, x_name="rate", inclusive_flag=True)
        if self.error<=0.0:
            raise ValueError(f"Parameter `error` must be > 0.0. Got {self.error} instead.")
        # Get confidence interval and set parameter
        self.ci = get_confidence_interval(rate=self.rate, error=self.error)

def success_rate(*,
    n_total: int,
    n_success: int,
    confidence_level: float
) -> SuccessRate:
    """Estimate success rate in a Bernoulli-distributed sample.

    Attack scores follow a Bernoulli distribution (success/failure with rates p/1-p).
    The Wilson score interval is a frequentist-type estimator for success rate and
    confidence which is robust in problematic cases (e.g., when p goes extreme or
    sample size is small). The estimated rate is a weighted average between the
    MLE result and 0.5 which, however, in the sample sizes used in privacy attacks
    does not differ visibly from the MLE outcome.

    Parameters
    ----------
    n_total : int
        Size of the sample.
    n_success : int
        Number of successful trials in the sample.
    confidence_level : float
        Confidence level for the error estimation.

    Returns
    -------
    SuccessRate estimate (rate, error and confidence interval)

    Notes
    -----
    E.B. WILSON
    Probable inference, the law of succession, and statistical inference
    Journal of the American Statistical Association 22, 209-212 (1927)
    DOI 10.1080/01621459.1927.10502953

    """
    #Assert input
    assert isinstance(n_total, int)
    assert n_total > 0
    assert isinstance(n_success, int)
    assert n_success >= 0
    assert isinstance(confidence_level, float)
    if n_success > n_total:
            raise ValueError("Parameter n_sucess can not be larger than n_total.")
    assert isinstance(confidence_level, float)
    assert_x_in_bound(x=confidence_level, x_name="confidence_level")
    # Probit value for given confidence level
    z = norm.ppf(0.5 * (1.0 + confidence_level))
    # Calculations
    z_squared = z * z
    denominator = n_total + z_squared
    rate = (n_success + z_squared / 2) / denominator
    n_success_var = n_success * (n_total - n_success) / n_total
    error = (z / denominator) * sqrt(n_success_var + z_squared / 4)
    return SuccessRate(rate=rate, error=error)

def residual_rate(*,
    main_rate: SuccessRate,
    naive_rate: SuccessRate,
) -> SuccessRate:
    """Compute residual success rate in a privacy attack.

    Residual success is defined as the excess of main attack
    success over naive attack success.

    Parameters
    ----------
    main_rate : SuccessRate
        Success rate of main attack.
    naive_rate : SuccessRate
        Success rate of naive attack.

    Returns
    -------
    SuccessRate
        Residual success rate, adjusted in case naive_rate>attack_rate,
        in which case the estimate is 0.
        The error estimate is the propagated error bound of the residual
        success rate.

    """
    # Calculate residual rate
    rate = main_rate.rate - naive_rate.rate
    # Propagate the error using
    # dF = sqrt[ (dF/dx)^2 dx^2 + (dF/dy)^2 dy^2 + ... ]
    error = sqrt(main_rate.error**2 + naive_rate.error**2)
    # Adjust for 0 as lower bound
    if rate<0:
        error = max(error + rate, 0.000001)
        rate = 0.0
    return SuccessRate(rate=rate, error=error)

class EvaluationResults(BaseModel):
    """Results of a privacy evaluator.

    Class will compute (on instantiation) the attacker's success rates
    on main and naive attacks, and estimate the corresponding
    residual success rate.

    Parameters
    ----------
    n_total : int
        Total number of guesses performed (for both main and naive attacks).
    n_main : int
        Number of successful guesses on main attack.
    n_naive : int
        Number of successful guesses on naive attack. Serves as baseline (random-guessing).
    confidence_level : float, default is 0.95
        Desired confidence level for the different rates' confidence intervals.

    """

    n_total: int
    n_main: int
    n_naive: int
    confidence_level: float = 0.95
    main_rate: Optional[SuccessRate] = None
    naive_rate: Optional[SuccessRate] = None
    residual_rate: Optional[SuccessRate] = None
    #Result columns
    res_cols: List[str] = ["n_total", "n_main", "n_naive", "confidence_level", "main_rate", "naive_rate", "residual_rate"]

    def __init__(self: Self, **kwargs: Union[int, float, Optional[SuccessRate]]) -> Union[None, ValueError]:
        super().__init__(**kwargs)
        #Assert input values
        if self.n_total <= 0:
            raise ValueError("n_total must be greater than 0.")
        if self.n_main < 0 or self.n_naive < 0:
            raise ValueError("n_main and n_naive must be greater or equal to 0.")
        if max(self.n_total, self.n_main, self.n_naive) != self.n_total:
            raise ValueError("n_total must be greater or equal than n_main and n_naive.")
        assert_x_in_bound(x=self.confidence_level, x_name="confidence_level")
        #Rates calculations
        self.main_rate = success_rate(n_total=self.n_total, n_success=self.n_main, confidence_level=self.confidence_level)
        self.naive_rate = success_rate(n_total=self.n_total, n_success=self.n_naive, confidence_level=self.confidence_level)
        self.residual_rate = residual_rate(main_rate=self.main_rate, naive_rate=self.naive_rate)

    def pack_results(self: Self) -> List[Union[int,float]]:
        """Returns a list with results."""
        #Note: if return object is changed, change corresponding res_cols attribute!
        return [
            self.n_total,
            self.n_main,
            self.n_naive,
            self.confidence_level,
            self.main_rate.rate,
            self.naive_rate.rate,
            self.residual_rate.rate
        ]

    def print_results(self: Self) -> None:
        """Prints results."""
        def pprint(rate:float) -> str:
            return f"{(rate*100):.2f}%"
        print(f"Success rate of main attack (and nr and total): {pprint(self.main_rate.rate)}, {self.n_main}, {self.n_total}") # noqa: T201
        print(f"Success rate of naive attack (and nr and total): {pprint(self.naive_rate.rate)}, {self.n_naive}, {self.n_total}") # noqa: T201
        print(f"Residual rate: {pprint(self.residual_rate.rate)}") # noqa: T201
