"""Tests for mixed_types_n_neighbors module."""
# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details.
import numpy as np
import pandas as pd
import pytest

from leakpro.synthetic_data_attacks.anonymeter.neighbors.mixed_types_n_neighbors import gower_distance, mixed_type_n_neighbors, shuffled_argsorted
from leakpro.tests.tests_synthetic_data_attacks.anonymeter_tests.fixtures import get_adult

rng = np.random.default_rng()

@pytest.mark.parametrize(
    ("r0", "r1", "expected"),
    [
        ([0, 1, 0, 0], [0, 1, 0, 0], 0),
        ([1, 1, 0, 0], [0, 1, 0, 0], 1),
        ([1, 1, 1, 0], [0, 1, 0, 0], 2),
        ([1, 0, 1, 0], [1, 1, 0, 1], 3),
        ([1, 0, 1, 0], [0, 1, 0, 1], 4),
    ],
)
def test_gower_distance_categorical(r0: list, r1: list, expected: int) -> None:
    """Assert result of gower_distance function when types are categorical."""
    r0, r1 = np.array(r0), np.array(r1)
    dist = gower_distance(r0=r0, r1=r1, cat_cols_index=0)
    np.testing.assert_equal(dist, expected)

def test_gower_distance_numerical() -> None:
    """Assert result of gower_distance function when types are numerical."""
    r0, r1 = rng.random(size=10), rng.random(size=10)
    dist = gower_distance(r0=r0, r1=r1, cat_cols_index=10)
    np.testing.assert_almost_equal(dist, np.sum(np.abs(r0 - r1)))

@pytest.mark.parametrize(
    ("r0", "r1", "expected"),
    [
        ([np.nan, 2.1, 0, 1, 0, 0], [1.1, 2.1, 0, 1, 0, 0], 1),
        ([1.1, 2.1, 1, np.nan, 0, 0], [1.1, 2.1, 0, 1, 0, 0], 2),
        ([1.1, 2.1, 1, np.nan, 0, 0], [1.1, 2.1, 0, np.nan, 0, 0], 1),
        ([1.1, 2.1, 1, 1, 1, 0], [1.1 , 2.3, 0, 1, 0, 0], 2.2),
        ([0, 1.5, 1, 0, 1, 0], [1, 3.8, 1, 1, 0, 1], 6.3)
    ],
)
def test_gower_distance(r0: list, r1: list, expected: int) -> None:
    """Assert result of gower_distance function when types are mixed (NaNs included)."""
    r0, r1 = np.array(r0), np.array(r1)
    dist = gower_distance(r0=r0, r1=r1, cat_cols_index=2)
    np.testing.assert_almost_equal(dist, expected)

def test_mixed_type_n_neighbors() -> None:
    """Assert result of test_mixed_type_n_neighbors function on adults first 10 rows."""
    df = get_adult(return_ori=True, n_samples=10)
    shuffled_idx = rng.integers(10, size=10)
    ids, dist = mixed_type_n_neighbors(
        queries = df.iloc[shuffled_idx],
        candidates = df,
        return_distance = True
    )
    np.testing.assert_equal(ids.flatten(), shuffled_idx)
    np.testing.assert_equal(dist, 0)

def test_mixed_type_n_neighbors_numerical() -> None:
    """Assert result of test_mixed_type_n_neighbors function on numerical differing rows."""
    ori = pd.DataFrame([[0.0, "a", "b"], [0.2, "a", "d"], [0.15, "a", "d"], [0.1, "a", "b"]])
    # Case query does not belong to ori.
    syn = pd.DataFrame([[0.01, "a", "b"]])
    ids, dist = mixed_type_n_neighbors(
        queries = syn,
        candidates = ori,
        n_neighbors = 4,
        return_distance = True
    )
    np.testing.assert_equal(ids, [[0, 3, 2, 1]])
    np.testing.assert_almost_equal(dist, [[0.05, 0.45, 1.7 , 1.95]])
    # Case query belongs to ori.
    # The distance to the 2nd record in ori will be maximal.
    syn = pd.DataFrame([[0.0, "a", "b"]])
    # Syn includes a row from ori.
    # The distance to the 2nd record in ori will be maximal.
    ids, dist = mixed_type_n_neighbors(
        queries = syn,
        candidates = ori,
        n_neighbors = 4,
        return_distance = True
    )
    np.testing.assert_equal(ids, [[0, 3, 2, 1]])
    np.testing.assert_almost_equal(dist, [[0, 0.5, 1.75, 2]])

@pytest.mark.parametrize(("n_neighbors", "n_queries"), [(1, 10), (3, 5)])
def test_mixed_type_n_neighbors_shape(n_neighbors: int, n_queries: int) -> None:
    """Assert result of returned ids.shape from test_mixed_type_n_neighbors function."""
    df = get_adult(return_ori=True, n_samples=10)
    ids = mixed_type_n_neighbors(
        queries = df.head(n_queries),
        candidates = df,
        n_neighbors = n_neighbors
    )
    assert isinstance(ids, np.ndarray)
    assert ids.shape == (n_queries, n_neighbors)

def test_shuffled_argsorted_logic():
    """
    Tests that shuffled_argsorted returns valid indices that sort the array,
    verifying against specific valid permutations for ties.
    """
    # Array with duplicates to allow for multiple valid index permutations
    arr = np.array([10, 5, 8, 5, 2])
    
    result = shuffled_argsorted(arr)
    
    # Check that the indices actually sort the array
    sorted_arr = arr[result]
    assert np.all(np.diff(sorted_arr) >= 0), "The result indices did not sort the array correctly"
    
    valid_option_1 = np.array([4, 1, 3, 2, 0])
    valid_option_2 = np.array([4, 3, 1, 2, 0])
    
    match_1 = np.array_equal(result, valid_option_1)
    match_2 = np.array_equal(result, valid_option_2)
    
    assert match_1 or match_2, (
        f"Result {result} matches neither expected permutation "
        f"({valid_option_1} or {valid_option_2})"
    )

def test_randomness_across_runs():
    """Ensure that running the function twice on the same data produces different index orders for ties."""
    arr = np.array([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3])
    
    run1 = shuffled_argsorted(arr)
    run2 = shuffled_argsorted(arr)
    
    # The values sorted by these indices must be the same (correct sorting)
    np.testing.assert_array_equal(arr[run1], arr[run2])
    
    # But the indices themselves should differ because of the shuffle
    # (Note: There is a tiny chance they shuffle to the same order, but with 17 items (17!) 
    # the chance is 1/355687428096000. We assume they won't match for this test).
    assert not np.array_equal(run1, run2), "Two runs produced identical permutations for tied values."

def test_edge_cases():
    """Test empty arrays and single-element arrays."""
    # Empty
    arr_empty = np.array([])
    assert len(shuffled_argsorted(arr_empty)) == 0
    
    # Single element
    arr_single = np.array([42])
    res_single = shuffled_argsorted(arr_single)
    assert len(res_single) == 1
    assert res_single[0] == 0