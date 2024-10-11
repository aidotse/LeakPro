"""Tests for inference_evaluator module."""
# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details.
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

from leakpro.synthetic_data_attacks.anonymeter.evaluators.inference_evaluator import InferenceEvaluator, InferenceGuesses
from leakpro.synthetic_data_attacks.anonymeter.stats.confidence import EvaluationResults
from leakpro.tests.tests_synthetic_data_attacks.anonymeter_tests.fixtures import get_adult


def test_InferenceGuesses_init() -> None: # noqa: N802
    """Assert InferenceGuesses.init method raises different errors for different input values."""
    #Case len(guesses.shape)<1
    with pytest.raises(AssertionError) as e:
        InferenceGuesses(guesses=np.array([[]]), secrets=np.array([]), regression=False)
    assert e.type is AssertionError
    #Case guesses.shape[0]==0
    with pytest.raises(AssertionError) as e:
        InferenceGuesses(guesses=np.array([]), secrets=np.array([]), regression=False)
    assert e.type is AssertionError
    #Case guesses.shape!=secrets.shape
    input_array = np.array([0])
    with pytest.raises(AssertionError) as e:
        InferenceGuesses(guesses=input_array, secrets=np.array([]), regression=False)
    assert e.type is AssertionError
    #Case tolerance not in range
    e_msg = "Parameter `tolerance` must be >= 0.0 and <= 1.0. Got -0.1 instead."
    with pytest.raises(ValueError, match=e_msg) as e:
        InferenceGuesses(guesses=input_array, secrets=input_array, regression=False, tolerance=-0.1)
    assert e.type is ValueError
    #Case no errors on init
    ig = InferenceGuesses(guesses=input_array, secrets=input_array, regression=False)
    assert ig.guesses == input_array
    assert ig.secrets == input_array
    assert ig.regression is False
    assert ig.tolerance == 0.05
    assert ig.matches is None
    assert ig.count is None

@pytest.mark.parametrize(
    ("guesses", "secrets", "e_matches", "e_count"),
    [
        (("a", "b"), ("a", "b"), (True, True), 2),
        ((np.nan, "b"), (np.nan, "b"), (True, True), 2),
        ((np.nan, np.nan), (np.nan, np.nan), (True, True), 2),
        ((np.nan, "b"), ("a", np.nan), (False, False), 0),
        (("a", "b"), ("a", "c"), (True, False), 1),
        (("b", "b"), ("a", "c"), (False, False), 0),
        ((1, 0), (2, 0), (False, True), 1)
    ],
)
def test_evaluate_inference_guesses_classification(
    guesses: Tuple,
    secrets: Tuple,
    e_matches: Tuple,
    e_count: int
) -> None:
    """Assert InferenceGuesses.evaluate_inference_guesses results for regression=False."""
    ig = InferenceGuesses(guesses=np.array(guesses), secrets=np.array(secrets), regression=False)
    ig.evaluate_inference_guesses()
    np.testing.assert_equal(ig.matches, e_matches)
    assert ig.count == e_count

@pytest.mark.parametrize(
    ("guesses", "secrets", "e_matches", "e_count"),
    [
        ((1.0, 1.0), (1.0, 1.0), (True, True), 2),
        ((1.01, 1.0), (1.0, 1.01), (True, True), 2),
        ((1.0, 1.0), (2.0, 1.01), (False, True), 1),
        ((1.0, 2.0), (2.0, 1.01), (False, False), 0)
    ],
)
def test_evaluate_inference_guesses_regression(
    guesses: Tuple,
    secrets: Tuple,
    e_matches: Tuple,
    e_count: int
) -> None:
    """Assert InferenceGuesses.evaluate_inference_guesses results for regression=True."""
    ig = InferenceGuesses(guesses=np.array(guesses), secrets=np.array(secrets), regression=True)
    ig.evaluate_inference_guesses()
    np.testing.assert_equal(ig.matches, e_matches)
    assert ig.count == e_count

@pytest.mark.parametrize(
    ("guesses", "secrets", "tolerance", "e_matches", "e_count"),
    [
        ((1.0, 1.0), (1.05, 1.06), 0.05, (True, False), 1),
        ((1.0, 1.0), (1.05, 1.06), 0.06, (True, True), 2),
        ((1.0, np.nan), (1.05, np.nan), 0.06, (True, True), 2),
        ((np.nan, np.nan), (np.nan, np.nan), 0.06, (True, True), 2),
        ((1, np.nan), (np.nan, 1.06), 0.06, (False, False), 0),
        ((1.0, 1.0), (1.05, 1.06), 0.04, (False, False), 0),
        ((1.0, 1.0), (1.25, 1.26), 0.2, (False, False), 0),
        ((1.0, 1.0), (1.26, 1.25), 0.25, (False, True), 1)
    ],
)
def test_evaluate_inference_guesses_regression_tolerance(
    guesses: Tuple,
    secrets: Tuple,
    tolerance: float,
    e_matches: Tuple,
    e_count: int
) -> None:
    """Assert InferenceGuesses.evaluate_inference_guesses results for regression=True and different tolerance input values."""
    ig = InferenceGuesses(guesses=np.array(guesses), secrets=np.array(secrets), regression=True, tolerance=tolerance)
    ig.evaluate_inference_guesses()
    np.testing.assert_equal(ig.matches, e_matches)
    assert ig.count == e_count

def test_InferenceEvaluator_init() -> None: # noqa: N802, PLR0915
    """Assert InferenceEvaluator.init raises errors on different input and no error flow."""
    #Case ori or syn has no rows
    e_msg = "ori and syn must contain rows."
    with pytest.raises(ValueError, match=e_msg) as e:
        InferenceEvaluator(ori=pd.DataFrame(), syn=pd.DataFrame([1]), aux_cols=[], secret="")
    assert e.type is ValueError
    with pytest.raises(ValueError, match=e_msg) as e:
        InferenceEvaluator(ori=pd.DataFrame([1]), syn=pd.DataFrame(), aux_cols=[], secret="")
    assert e.type is ValueError
    #Case ori contains only 1 column
    e_msg = "ori must contain at least 2 columns."
    with pytest.raises(ValueError, match=e_msg) as e:
        InferenceEvaluator(ori=pd.DataFrame([1]), syn=pd.DataFrame([1]), aux_cols=[], secret="")
    assert e.type is ValueError
    #Case ori and syn columns differ
    df_ex = pd.DataFrame([[1,1]], columns=["a", "b"])
    e_msg = "ori and syn columns must be equal."
    with pytest.raises(ValueError, match=e_msg) as e:
        InferenceEvaluator(ori=df_ex, syn=pd.DataFrame([1]), aux_cols=[], secret="")
    assert e.type is ValueError
    #Case aux_cols contains no elements
    e_msg = "aux_cols must contain at least 1 element."
    with pytest.raises(ValueError, match=e_msg) as e:
        InferenceEvaluator(ori=df_ex, syn=df_ex, aux_cols=[], secret="")
    assert e.type is ValueError
    #Case aux_cols not subset of ori.columns
    e_msg = "aux_cols not subset of ori.columns."
    with pytest.raises(ValueError, match=e_msg) as e:
        InferenceEvaluator(ori=df_ex, syn=df_ex, aux_cols=[""], secret="")
    assert e.type is ValueError
    #Case secret not in ori.columns
    e_msg = "secret not in ori.columns."
    with pytest.raises(ValueError, match=e_msg) as e:
        InferenceEvaluator(ori=df_ex, syn=df_ex, aux_cols=["a"], secret="")
    assert e.type is ValueError
    #Case secret included in aux_columns
    e_msg = "secret can't be included in aux_columns."
    with pytest.raises(ValueError, match=e_msg) as e:
        InferenceEvaluator(ori=df_ex, syn=df_ex, aux_cols=["a"], secret="a") # noqa: S106
    assert e.type is ValueError
    #Case confidence_level not in bound
    e_msg = "Parameter `confidence_level` must be > 0.0 and < 1.0. Got 0.0 instead."
    with pytest.raises(ValueError, match=e_msg) as e:
        InferenceEvaluator(ori=df_ex, syn=df_ex, aux_cols=["a"], secret="b", confidence_level=0.0) # noqa: S106
    assert e.type is ValueError
    #Case no error raised
    ie = InferenceEvaluator(ori=df_ex, syn=df_ex, aux_cols=["a"], secret="b") # noqa: S106
    assert ie.ori.equals(df_ex)
    assert ie.syn.equals(df_ex)
    assert ie.aux_cols == ["a"]
    assert ie.secret == "b" # noqa: S105
    assert ie.n_attacks == 1
    assert ie.confidence_level == 0.95
    assert ie.n_jobs == -2
    assert ie.main_guesses is None
    assert ie.naive_guesses is None
    assert ie.results is None
    assert ie.regression
    #Case regression is False
    df_ex = pd.DataFrame([[0,"a"]], columns=["col0", "col1"])
    ie = InferenceEvaluator(ori=df_ex, syn=df_ex, aux_cols=["col0"], secret="col1") # noqa: S106
    assert ie.regression is False

@pytest.mark.parametrize(
    ("ori", "syn", "e_n_main"),
    [
        ([["a", "b"], ["c", "d"]], [["a", "b"], ["c", "d"]], 2),
        ([["a", "b"], ["c", "d"]], [["a", "b"], ["c", "e"]], 1),
        ([["a", "b"], ["c", "d"]], [["a", "h"], ["c", "g"]], 0),
    ],
)
def test_inference_evaluator(ori: List, syn: List, e_n_main: int) -> None:
    """Assert InferenceEvaluator results for different input values."""
    cols = ["c0", "c1"]
    ori = pd.DataFrame(ori, columns=cols)
    syn = pd.DataFrame(syn, columns=cols)
    evaluator = InferenceEvaluator(
        ori = ori,
        syn = syn,
        aux_cols = [cols[0]],
        secret = cols[1],
        n_attacks = 2,
        n_jobs = 1
    )
    results = evaluator.evaluate()
    assert isinstance(results, EvaluationResults)
    assert results.n_total == 2
    assert results.n_main == e_n_main

@pytest.mark.parametrize(
    "aux_cols",
    [
        ["type_employer", "capital_loss", "hr_per_week", "age"],
        ["education_num", "capital_loss", "hr_per_week"],
        ["age", "type_employer", "race"],
    ],
)
@pytest.mark.parametrize("secret", ["education", "marital", "capital_gain"])
def test_inference_evaluator_full_leak(aux_cols: List, secret: str) -> None:
    """Assert InferenceEvaluator results with a sample of adults dataset as both ori and syn."""
    ori = get_adult(return_ori=True, n_samples=10)
    evaluator = InferenceEvaluator(
        ori = ori,
        syn = ori,
        aux_cols = aux_cols,
        secret = secret,
        n_attacks = 10,
        n_jobs = 1,
    )
    results = evaluator.evaluate()
    assert isinstance(results, EvaluationResults)
    assert results.n_total == 10
    assert results.n_main == 10
