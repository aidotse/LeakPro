"""Tests for confidence module."""
# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details.
from typing import Union

import numpy as np
import pytest
from pydantic_core._pydantic_core import ValidationError

from leakpro.synthetic_data_attacks.anonymeter.stats.confidence import (
    EvaluationResults,
    SuccessRate,
    assert_x_in_bound,
    get_confidence_interval,
    residual_rate,
    success_rate,
)


def test_assert_x_in_bound() -> None:
    """Assert results of assert_x_in_bound function on different input."""
    #Case not correct type input
    with pytest.raises(TypeError) as e:
        assert_x_in_bound(x="not_float_or_int")
    assert e.type == TypeError
    assert str(e.value) == "'<=' not supported between instances of 'str' and 'float'"
    #Case ValueError when x not in bounds
    e_msg = "Parameter `param1` must be > 0.0 and < 10. Got 11 instead."
    with pytest.raises(ValueError, match=e_msg) as e:
        assert_x_in_bound(x=11, x_name="param1", high_bound=10)
    assert e.type == ValueError
    #Case x in bounds
    assert_x_in_bound(x=0.5)
    assert_x_in_bound(x=9, high_bound=10)
    #Case x equal to low_bound/high_bound and inclusive_flag = True
    assert_x_in_bound(x=0, inclusive_flag=True)
    assert_x_in_bound(x=1, inclusive_flag=True)
    #Case x equal to low_bound/high_bound and inclusive_flag = True
    e_msg = "Parameter must be > 0.0 and < 1.0. Got 0 instead."
    with pytest.raises(ValueError, match=e_msg) as e:
        assert_x_in_bound(x=0, inclusive_flag=False)
    assert e.type == ValueError
    e_msg = "Parameter must be > 0.0 and < 1.0. Got 1 instead."
    with pytest.raises(ValueError, match=e_msg) as e:
        assert_x_in_bound(x=1, inclusive_flag=False)
    assert e.type == ValueError

@pytest.mark.parametrize(
    ("rate", "error", "e_lower_b", "e_upper_b"),
    [
        (0.1, 0.2, 0, 0.3),
        (0.9, 0.15, 0.75, 1),
        (0.5, 0.1, 0.4, 0.6)
    ],
)
def test_get_confidence_interval(rate: float, error: float, e_lower_b: float, e_upper_b: float) -> None:
    """Assert results of get_confidence_interval function on different input."""
    lower_b, upper_b = get_confidence_interval(rate=rate, error=error)
    assert round(lower_b, 3) == e_lower_b
    assert round(upper_b, 3) == e_upper_b

def test_SuccessRate_init() -> None: # noqa: N802
    """Assert results of SuccessRate init method on different input."""
    #Case not correct type input
    with pytest.raises(ValidationError) as e:
        SuccessRate(rate="not_float", error=0.1, ci=(0,0.15))
    assert e.type == ValidationError
    e_value = "1 validation error for SuccessRate"
    assert str(e.value)[0:len(e_value)] == e_value
    #Case ValueError when error <= 0
    e_msg = "Parameter `error` must be > 0.0. Got 0.0 instead."
    with pytest.raises(ValueError, match=e_msg) as e:
        SuccessRate(rate=0.0, error=0.0)
    assert e.type == ValueError
    #Case normal input
    sr = SuccessRate(rate=0.05, error=0.1)
    assert sr.rate == 0.05
    assert sr.error == 0.1
    assert sr.ci[0] == 0.0
    assert round(sr.ci[1], 6) == 0.15

def test_success_rate_input_errors() -> None:
    """Assert success_rate different errors raised on different input values."""
    #Case n_total <= 0
    with pytest.raises(AssertionError) as e:
        success_rate(n_total=0, n_success=0, confidence_level=0.95)
    assert e.type == AssertionError
    #Case n_success < 0
    with pytest.raises(AssertionError) as e:
        success_rate(n_total=1, n_success=-1, confidence_level=0.95)
    assert e.type == AssertionError
    #Case n_success > n_total
    e_msg = "Parameter n_sucess can not be larger than n_total."
    with pytest.raises(ValueError, match=e_msg) as e:
        success_rate(n_total=1, n_success=2, confidence_level=0.95)
    assert e.type == ValueError
    #Case confidence_level not in (0,1)
    e_msg = "Parameter `confidence_level` must be > 0.0 and < 1.0. Got 1.0 instead."
    with pytest.raises(ValueError, match=e_msg) as e:
        success_rate(n_total=1, n_success=0, confidence_level=1.0)
    assert e.type == ValueError

@pytest.mark.parametrize(
    ("n_success", "e_risk", "e_error", "e_low_bound", "e_high_bound"),
    [
        (850, 0.849, 0.022, 0.827, 0.871),
        (0, 0.002, 0.002, 0.0, 0.004),
        (1000, 0.998, 0.002, 0.996, 1.0)
    ],
)
def test_success_rate(
    n_success: int,
    e_risk: float,
    e_error: float,
    e_low_bound: float,
    e_high_bound: float
) -> None:
    """Assert results of success_rate function on different input values."""
    sr = success_rate(n_total=1000, n_success=n_success, confidence_level=0.95)
    assert np.round(sr.rate, decimals=3) == e_risk
    assert np.round(sr.error, decimals=3) == e_error
    assert np.round(sr.ci[0], decimals=3) == e_low_bound
    assert np.round(sr.ci[1], decimals=3) == e_high_bound

def assert_equal_SuccessRates(x: SuccessRate, y: SuccessRate) -> Union[None, AssertionError]: # noqa: N802
    """Assert provided SuccessRates instances are equal up to certain rounding."""
    assert isinstance(x, SuccessRate)
    assert isinstance(y, SuccessRate)
    #Rate and error rounding
    rr = 6
    er = 4
    assert round(x.rate, rr) == round(y.rate, rr)
    assert round(x.error, rr) == round(y.error, rr)
    assert round(x.ci[0], er) == round(y.ci[0], er)
    assert round(x.ci[1], er) == round(y.ci[1], er)

@pytest.mark.parametrize(
    ("main_rate", "naive_rate", "e_rate"),
    [
        (SuccessRate(rate=0.05, error=0.05), SuccessRate(rate=0.06, error=0.1), SuccessRate(rate=0.0, error=0.119953)),
        (SuccessRate(rate=0.8, error=0.1), SuccessRate(rate=0.9, error=0.1), SuccessRate(rate=0.0, error=2.236068)),
        (SuccessRate(rate=0.9, error=0.1), SuccessRate(rate=0.8, error=0.1), SuccessRate(rate=0.5, error=0.559017)),
        (SuccessRate(rate=0.9, error=0.02), SuccessRate(rate=0.85, error=0.02), SuccessRate(rate=0.333333, error=0.160247)),
        (SuccessRate(rate=0.05, error=0.05), SuccessRate(rate=0.03, error=0.04), SuccessRate(rate=0.020619, error=0.065484))
    ],
)
def test_residual_rate(main_rate: SuccessRate, naive_rate: SuccessRate, e_rate: SuccessRate) -> None:
    """Assert results of residual_rate function on different input values."""
    residual = residual_rate(main_rate=main_rate, naive_rate=naive_rate)
    assert_equal_SuccessRates(residual, e_rate)

def test_residual_rate_naive_rate_1() -> None:
    """Assert residual_rate function results when naive_rate.rate == 1."""
    e_warning = "Success of naive attack is 100%. Cannot measure residual success rate."
    with pytest.warns(UserWarning, match=e_warning):
        residual = residual_rate(
            main_rate = SuccessRate(rate=0.9, error=0.02),
            naive_rate = SuccessRate(rate=1, error=0.05)
        )
    e_rate = SuccessRate(rate=1.0, error=0.01)
    assert_equal_SuccessRates(residual, e_rate)

def test_EvaluationResults_init_error() -> None: # noqa: N802
    """Assert EvaluationResults.init method for different errors raised on different input values."""
    #Case n_total <= 0
    e_msg = "n_total must be greater than 0."
    with pytest.raises(ValueError, match=e_msg) as e:
        EvaluationResults(
            n_total=0,
            n_main=0,
            n_naive=0,
            confidence_level=0,
        )
    assert e.type == ValueError
    #Case n_naive or n_naive < 0
    e_msg = "n_main and n_naive must be greater or equal to 0."
    with pytest.raises(ValueError, match=e_msg) as e:
        EvaluationResults(
            n_total=1,
            n_main=-1,
            n_naive=0
        )
    assert e.type == ValueError
    with pytest.raises(ValueError, match=e_msg) as e:
        EvaluationResults(
            n_total=1,
            n_main=0,
            n_naive=-1
        )
    assert e.type == ValueError
    #Case n_total not max of n_main and n_naive
    e_msg = "n_total must be greater or equal than n_main and n_naive."
    with pytest.raises(ValueError, match=e_msg) as e:
        EvaluationResults(
            n_total=1,
            n_main=2,
            n_naive=1
        )
    assert e.type == ValueError
    with pytest.raises(ValueError, match=e_msg) as e:
        EvaluationResults(
            n_total=1,
            n_main=1,
            n_naive=2
        )
    assert e.type == ValueError
    #Case confidence_level not in (0,1) interval
    e_msg = "Parameter `confidence_level` must be > 0.0 and < 1.0. Got 0.0 instead."
    with pytest.raises(ValueError, match=e_msg) as e:
        EvaluationResults(
            n_total=1,
            n_main=0,
            n_naive=0,
            confidence_level=0
        )
    assert e.type == ValueError
    #Case no error on input
    EvaluationResults(
        n_total=1,
        n_main=0,
        n_naive=0
    )

@pytest.mark.parametrize(
    ("n_total", "n_main", "n_naive", "conf_level"),
    [
        (100, 100, 0, 0.95),
        (100, 23, 11, 0.95),
        (111, 84, 42, 0.95),
        (100, 0, 100, 0.95),
        (100, 100, 0, 0.8),
        (100, 23, 11, 0.8),
        (111, 84, 42, 0.8),
        (100, 0, 100, 0.8),
    ],
)
def test_EvaluationResults(n_total: int, n_main: int, n_naive: int, conf_level: float) -> None: # noqa: N802
    """Assert EvaluationResults.init results for different input values."""
    e_main = success_rate(n_total=n_total, n_success=n_main, confidence_level=conf_level)
    e_naive = success_rate(n_total=n_total, n_success=n_naive, confidence_level=conf_level)
    e_residual = residual_rate(main_rate=e_main, naive_rate=e_naive)
    res = EvaluationResults(
        n_total=n_total,
        n_main=n_main,
        n_naive=n_naive,
        confidence_level=conf_level
    )
    assert_equal_SuccessRates(res.main_rate, e_main)
    assert_equal_SuccessRates(res.naive_rate, e_naive)
    assert_equal_SuccessRates(res.residual_rate, e_residual)
