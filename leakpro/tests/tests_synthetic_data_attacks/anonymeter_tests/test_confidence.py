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
    assert e.type is TypeError
    assert str(e.value) == "'<=' not supported between instances of 'str' and 'float'"
    #Case ValueError when x not in bounds
    e_msg = "Parameter `param1` must be > 0.0 and < 10. Got 11 instead."
    with pytest.raises(ValueError, match=e_msg) as e:
        assert_x_in_bound(x=11, x_name="param1", high_bound=10)
    assert e.type is ValueError
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
    assert e.type is ValueError
    e_msg = "Parameter must be > 0.0 and < 1.0. Got 1 instead."
    with pytest.raises(ValueError, match=e_msg) as e:
        assert_x_in_bound(x=1, inclusive_flag=False)
    assert e.type is ValueError

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
    assert e.type is ValueError
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
    assert e.type is AssertionError
    #Case n_success < 0
    with pytest.raises(AssertionError) as e:
        success_rate(n_total=1, n_success=-1, confidence_level=0.95)
    assert e.type is AssertionError
    #Case n_success > n_total
    e_msg = "Parameter n_sucess can not be larger than n_total."
    with pytest.raises(ValueError, match=e_msg) as e:
        success_rate(n_total=1, n_success=2, confidence_level=0.95)
    assert e.type is ValueError
    #Case confidence_level not in (0,1)
    e_msg = "Parameter `confidence_level` must be > 0.0 and < 1.0. Got 1.0 instead."
    with pytest.raises(ValueError, match=e_msg) as e:
        success_rate(n_total=1, n_success=0, confidence_level=1.0)
    assert e.type is ValueError

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
        (SuccessRate(rate=0.05, error=0.01), SuccessRate(rate=0.1, error=0.01), SuccessRate(rate=0.0, error=0.000001)),
        (SuccessRate(rate=0.05, error=0.05), SuccessRate(rate=0.06, error=0.1), SuccessRate(rate=0.0, error=0.101803)),
        (SuccessRate(rate=0.8, error=0.1), SuccessRate(rate=0.9, error=0.1), SuccessRate(rate=0.0, error=0.041421)),
        (SuccessRate(rate=0.9, error=0.1), SuccessRate(rate=0.8, error=0.1), SuccessRate(rate=0.1, error=0.141421)),
        (SuccessRate(rate=0.9, error=0.02), SuccessRate(rate=0.85, error=0.02), SuccessRate(rate=0.05, error=0.028284)),
        (SuccessRate(rate=0.05, error=0.05), SuccessRate(rate=0.03, error=0.04), SuccessRate(rate=0.02, error=0.064031))
    ],
)
def test_residual_rate(main_rate: SuccessRate, naive_rate: SuccessRate, e_rate: SuccessRate) -> None:
    """Assert results of residual_rate function on different input values."""
    residual = residual_rate(main_rate=main_rate, naive_rate=naive_rate)
    assert_equal_SuccessRates(residual, e_rate)

def test_EvaluationResults_init_error() -> None: # noqa: N802
    """Assert EvaluationResults.init method for different errors raised on different input values."""
    #Case n_main_total or n_naive_total <= 0
    e_msg = "n_main_total and n_naive_total must be greater than 0."
    with pytest.raises(ValueError, match=e_msg) as e:
        EvaluationResults(
            n_main_total = 0,
            n_main_success = 0,
            n_naive_total = 1,
            n_naive_success = 0,
            confidence_level = 0,
        )
    assert e.type is ValueError
    with pytest.raises(ValueError, match=e_msg) as e:
        EvaluationResults(
            n_main_total = 1,
            n_main_success = 0,
            n_naive_total = 0,
            n_naive_success = 0,
            confidence_level = 0,
        )
    assert e.type is ValueError
    #Case n_main_success or n_naive_success < 0
    e_msg = "n_main_success and n_naive_success must be greater or equal to 0."
    with pytest.raises(ValueError, match=e_msg) as e:
        EvaluationResults(
            n_main_total = 1,
            n_main_success = -1,
            n_naive_total = 1,
            n_naive_success = 0
        )
    assert e.type is ValueError
    with pytest.raises(ValueError, match=e_msg) as e:
        EvaluationResults(
            n_main_total = 1,
            n_main_success = 1,
            n_naive_total = 1,
            n_naive_success = -1
        )
    assert e.type is ValueError
    #Case n_main_total < n_main_success
    e_msg = "n_main_total must be greater or equal than n_main_success."
    with pytest.raises(ValueError, match=e_msg) as e:
        EvaluationResults(
            n_main_total = 1,
            n_main_success = 2,
            n_naive_total = 1,
            n_naive_success = 0
        )
    assert e.type is ValueError
    #Case n_naive_total < n_naive_success.
    e_msg = "n_naive_total must be greater or equal than n_naive_success."
    with pytest.raises(ValueError, match=e_msg) as e:
        EvaluationResults(
            n_main_total = 1,
            n_main_success = 0,
            n_naive_total = 1,
            n_naive_success = 2
        )
    assert e.type is ValueError
    #Case confidence_level not in (0,1) interval
    e_msg = "Parameter `confidence_level` must be > 0.0 and < 1.0. Got 0.0 instead."
    with pytest.raises(ValueError, match=e_msg) as e:
        EvaluationResults(
            n_main_total = 1,
            n_main_success = 0,
            n_naive_total = 1,
            n_naive_success = 0,
            confidence_level = 0
        )
    assert e.type is ValueError
    #Case no error on input
    EvaluationResults(
        n_main_total = 1,
        n_main_success = 0,
        n_naive_total = 1,
        n_naive_success = 0
    )

@pytest.mark.parametrize(
    ("n_main_total", "n_main_success", "n_naive_total", "n_naive_success", "conf_level"),
    [
        (100, 100, 100, 0, 0.95),
        (100, 23, 100, 11, 0.95),
        (111, 84, 100, 42, 0.95),
        (100, 0, 100, 100, 0.95),
        (100, 100, 100, 0, 0.8),
        (100, 23, 100, 11, 0.8),
        (111, 84, 100, 42, 0.8),
        (100, 0, 100, 100, 0.8),
    ],
)
def test_EvaluationResults( # noqa: N802
    n_main_total: int,
    n_main_success: int,
    n_naive_total: int,
    n_naive_success: int,
    conf_level: float
) -> None:
    """Assert EvaluationResults.init results for different input values.

    Test additionally tests pack_results method.
    """
    e_main = success_rate(n_total=n_main_total, n_success=n_main_success, confidence_level=conf_level)
    e_naive = success_rate(n_total=n_naive_total, n_success=n_naive_success, confidence_level=conf_level)
    e_residual = residual_rate(main_rate=e_main, naive_rate=e_naive)
    res = EvaluationResults(
        n_main_total = n_main_total,
        n_main_success = n_main_success,
        n_naive_total = n_naive_total,
        n_naive_success = n_naive_success,
        confidence_level = conf_level
    )
    e_res_cols = [
        "n_main_total", "n_main_success", "n_naive_total", "n_naive_success",
        "confidence_level", "main_rate", "naive_rate", "residual_rate"
    ]
    assert res.res_cols == e_res_cols
    assert_equal_SuccessRates(res.main_rate, e_main)
    assert_equal_SuccessRates(res.naive_rate, e_naive)
    assert_equal_SuccessRates(res.residual_rate, e_residual)
    #Test pack_results method
    packed_res = res.pack_results()
    assert len(packed_res) == len(e_res_cols)
    assert packed_res == [
        res.n_main_total,
        res.n_main_success,
        res.n_naive_total,
        res.n_naive_success,
        res.confidence_level,
        res.main_rate.rate,
        res.naive_rate.rate,
        res.residual_rate.rate
    ]
