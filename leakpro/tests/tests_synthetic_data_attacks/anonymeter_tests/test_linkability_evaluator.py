"""Tests for linkability_evaluator module."""
# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details.
from typing import Callable, Generator

import numpy as np
import pandas as pd
import pytest

from leakpro.synthetic_data_attacks.anonymeter.evaluators.linkability_evaluator import LinkabilityEvaluator, LinkabilityIndexes
from leakpro.synthetic_data_attacks.anonymeter.stats.confidence import EvaluationResults


def test_LinkabilityIndexes_init() -> None: # noqa: N802
    """Assert LinkabilityIndexes.init method raises different errors for different input values."""
    #Case len(idx_0.shape)==1
    with pytest.raises(AssertionError) as e:
        LinkabilityIndexes(idx_0=np.array([]), idx_1=np.array([]))
    assert e.type == AssertionError
    #Case len(idx_0.shape)>1 but idx_0.shape[1]==0
    with pytest.raises(AssertionError) as e:
        LinkabilityIndexes(idx_0=np.array([[]]), idx_1=np.array([[]]))
    assert e.type == AssertionError
    #Case len(idx_0.shape)>1, idx_0.shape[1]>0 and idx_0.shape!=idx_1.shape
    np_ex = np.array([[1]])
    with pytest.raises(AssertionError) as e:
        LinkabilityIndexes(idx_0=np_ex, idx_1=np.array([[1,2]]))
    assert e.type == AssertionError
    #Case no errors on init
    li = LinkabilityIndexes(idx_0=np_ex, idx_1=np_ex)
    assert li.idx_0 == np_ex
    assert li.idx_1 == np_ex
    assert li.links is None
    assert li.count is None

def test_LinkabilityIndexes_find_links_errors() -> None: # noqa: N802
    """Assert LinkabilityIndexes.find_links raises errors on n_neighors input."""
    #Case n_neighbors not in bounds
    np_ex = np.array([[1]])
    e_msg = "Parameter `n_neighbors` must be >= 1 and <= 1. Got 0 instead."
    with pytest.raises(ValueError, match=e_msg) as e:
        LinkabilityIndexes(idx_0=np_ex, idx_1=np_ex).find_links(n_neighbors=0)
    assert e.type == ValueError
    e_msg = "Parameter `n_neighbors` must be >= 1 and <= 1. Got 2 instead."
    with pytest.raises(ValueError, match=e_msg) as e:
        LinkabilityIndexes(idx_0=np_ex, idx_1=np_ex).find_links(n_neighbors=2)
    assert e.type == ValueError

@pytest.mark.parametrize(
    ("n_neighbors", "idx_0", "idx_1", "e_links", "e_count"),
    [
        (1, [[0], [1], [2], [3]], [[4], [5], [6], [7]], {}, 0),
        (1, [[0], [1], [2], [3]], [[4], [1], [6], [7]], {1: np.array([1])}, 1),
        (1, [[0], [1], [6], [3]], [[4], [1], [6], [7]], {1: np.array([1]), 2: np.array([6])}, 2),
        (1, [[0, 1], [2, 3]], [[1, 0], [3, 2]], {}, 0),
        (2, [[0, 1], [2, 3]], [[1, 0], [3, 2]], {0: np.array([0, 1]), 1: np.array([2, 3])}, 2),
    ],
)
def test_find_links(n_neighbors: int, idx_0: list, idx_1: list, e_links: dict, e_count: int) -> None:
    """Assert test_LinkabilityIndexes.find_links results with different input values."""
    indexes = LinkabilityIndexes(idx_0=np.array(idx_0), idx_1=np.array(idx_1))
    indexes = indexes.find_links(n_neighbors=n_neighbors)
    assert isinstance(indexes, LinkabilityIndexes)
    assert indexes.links.keys() == e_links.keys()
    for k,v in indexes.links.items():
        assert np.equal(v, e_links[k]).all()
    assert indexes.count == e_count

def test_LinkabilityEvaluator_init() -> None: # noqa: N802
    """Assert LinkabilityEvaluator.init raises errors on different input and no error flow."""
    #Case ori or syn has no rows
    e_msg = "ori and syn must contain rows."
    with pytest.raises(ValueError, match=e_msg) as e:
        LinkabilityEvaluator(ori=pd.DataFrame(), syn=pd.DataFrame([1]), aux_cols=([],[]))
    assert e.type == ValueError
    with pytest.raises(ValueError, match=e_msg) as e:
        LinkabilityEvaluator(ori=pd.DataFrame([1]), syn=pd.DataFrame(), aux_cols=([],[]))
    assert e.type == ValueError
    #Case ori and syn columns differ
    df_ex = pd.DataFrame([1], columns=["a"])
    e_msg = "ori and syn columns must be equal."
    with pytest.raises(ValueError, match=e_msg) as e:
        LinkabilityEvaluator(ori=df_ex, syn=pd.DataFrame([1]), aux_cols=([],[]))
    assert e.type == ValueError
    #Case aux_cols contains no elements
    df_ex = pd.DataFrame([1], columns=["a"])
    e_msg = "aux_cols tuple must contain 2 list with at least 1 element."
    with pytest.raises(ValueError, match=e_msg) as e:
        LinkabilityEvaluator(ori=df_ex, syn=df_ex, aux_cols=([],["a"]))
    assert e.type == ValueError
    with pytest.raises(ValueError, match=e_msg) as e:
        LinkabilityEvaluator(ori=df_ex, syn=df_ex, aux_cols=(["a"],[]))
    assert e.type == ValueError
    #Case confidence_level not in bound
    aux_cols = (["a"],["a"])
    e_msg = "Parameter `confidence_level` must be > 0.0 and < 1.0. Got 0.0 instead."
    with pytest.raises(ValueError, match=e_msg) as e:
        LinkabilityEvaluator(ori=df_ex, syn=df_ex, aux_cols=aux_cols, confidence_level=0.0)
    assert e.type == ValueError
    #Case no error raised
    le = LinkabilityEvaluator(ori=df_ex, syn=df_ex, aux_cols=aux_cols)
    assert le.ori.equals(df_ex)
    assert le.syn.equals(df_ex)
    assert le.aux_cols == aux_cols
    assert le.n_attacks == 1
    assert le.confidence_level == 0.95
    assert le.n_neighbors == 1
    assert le.n_jobs == -2
    assert le.main_links is None
    assert le.naive_links is None
    assert le.results is None

@pytest.mark.parametrize(
    ("n_neighbors", "confidence_level", "e_main_rate", "e_ci"),
    [
        (1, 0.1, 0.5, (0.4686, 0.5314)),
        (2, 0.1, 0.5, (0.4686, 0.5314)),
        (3, 0.1, 0.5, (0.4686, 0.5314)),
        (4, 0.1, 0.5, (0.4686, 0.5314)),
        (1, 0.95, 0.5, (0.1500, 0.85)),
        (2, 0.95, 0.5, (0.1500, 0.85)),
    ],
)
def test_linkability_main_attack(
    monkeypatch: Generator,
    n_neighbors: int,
    confidence_level: float,
    e_main_rate: float,
    e_ci: tuple
) -> None:
    """Test linkability main attack results.

    Test implicitly tests functions main_linkability_attack, random_links and naive_linkability_attack.
    """
    #Note: monkeypatch for fixing random seed in calculations
    rng = np.random.default_rng(seed=42)
    callable = np.random.default_rng
    def mock_np_random_default_rng(seed: int = None) -> Callable:
        if seed is None:
            return callable(seed=int(rng.random()*1000))
        return callable(seed=seed)
    monkeypatch.setattr(np.random, "default_rng", mock_np_random_default_rng)
    #Continuing with test after monkeypatch
    ori = pd.DataFrame({"col0": [0, 1, 4, 9], "col1": [0, 1, 9, 4]})
    syn = pd.DataFrame({"col0": [0, 1, 4, 9], "col1": [0, 1, 4, 9]})
    evaluator = LinkabilityEvaluator(
        ori = ori,
        syn = syn,
        aux_cols = (["col0"], ["col1"]),
        n_attacks = 4,
        confidence_level = confidence_level,
        n_neighbors = n_neighbors,
        n_jobs = 1
    )
    results = evaluator.evaluate()
    assert isinstance(results, EvaluationResults)
    main_rate = results.main_rate
    assert main_rate.rate == e_main_rate
    assert abs(main_rate.ci[0]-e_ci[0])<0.0001
    assert abs(main_rate.ci[1]-e_ci[1])<0.0001

@pytest.mark.parametrize(
    ("n_neighbors", "e_naive_rate"),
    [(1, 0.25), (2, 0.25)]
)
def test_linkability_naive_attack(n_neighbors: int, e_naive_rate: float) -> None:
    """Test linkability naive attack results.

    Test implicitly tests functions main_linkability_attack, random_links and naive_linkability_attack.
    """
    rng = np.random.default_rng(seed=42)
    # Note that for the naive attack, it does not really matter
    # what's inside the synthetic or the original dataframe.
    ori = pd.DataFrame(rng.choice(["a", "b"], size=(400, 2)), columns=["c0", "c1"])
    syn = pd.DataFrame([["a", "a"], ["b", "b"], ["a", "a"], ["a", "a"]], columns=["c0", "c1"])
    evaluator = LinkabilityEvaluator(
        ori = ori,
        syn = syn,
        aux_cols = (["c0"],["c1"]),
        n_neighbors=n_neighbors,
        n_jobs = 1
    )
    results = evaluator.evaluate()
    assert isinstance(results, EvaluationResults)
    naive_rate = results.naive_rate
    assert naive_rate.ci[0] <= e_naive_rate
    assert e_naive_rate <= naive_rate.ci[1]
