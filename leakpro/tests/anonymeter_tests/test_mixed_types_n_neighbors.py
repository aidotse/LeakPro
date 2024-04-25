"""Tests for mixed_types_n_neighbors module."""
# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details.
import numpy as np
import pandas as pd
import pytest

from leakpro.synthetic_data_attacks.anonymeter.neighbors.mixed_types_n_neighbors import gower_distance, mixed_type_n_neighbors
from leakpro.tests.anonymeter_tests.fixtures import get_adult

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
    df = get_adult("ori", n_samples=10)
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
    df = get_adult("ori", n_samples=10)
    ids = mixed_type_n_neighbors(
        queries = df.head(n_queries),
        candidates = df,
        n_neighbors = n_neighbors
    )
    assert isinstance(ids, np.ndarray)
    assert ids.shape == (n_queries, n_neighbors)
