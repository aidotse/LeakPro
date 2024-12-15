"""Tests for linkability_utils module."""
import os
from typing import List

import numpy as np
import pytest

import leakpro.synthetic_data_attacks.linkability_utils as lu
from leakpro.synthetic_data_attacks import utils as u
from leakpro.tests.tests_synthetic_data_attacks.anonymeter_tests.fixtures import get_adult


@pytest.mark.parametrize(
    ("cols", "buck1_nr", "buck2_nr", "raises_error"),
    [
        (["a", "b"], 0, 1, True), #Case buck1_nr or buck2_nr not positive
        (["a", "b"], 1, 0, True), #Case buck1_nr or buck2_nr not positive
        (["a", "b"], 1, 2, True), #Case buck1_nr < buck2_nr
        (["a"], 1, 1, True), #Case len(cols) < 2
        (["a", "b"], 2, 1, True), #Case buck1_nr+buck2_nr>len(cols)
        (["a", "b"], 1, 1, False), #Case no AssertionError

    ],
)
def test_aux_assert_input_values_get_combs_2_buckets(cols: List[str], buck1_nr: int, buck2_nr: int, raises_error: bool) -> None:
    """Assert aux_assert_input_values_get_combs_2_buckets function raises AssertionError for different input values."""
    if raises_error:
        with pytest.raises(AssertionError) as e:
            lu.aux_assert_input_values_get_combs_2_buckets(cols=cols, buck1_nr=buck1_nr, buck2_nr=buck2_nr)
        assert e.type is AssertionError
    else:
        lu.aux_assert_input_values_get_combs_2_buckets(cols=cols, buck1_nr=buck1_nr, buck2_nr=buck2_nr)

def test_get_all_combs_2_buckets() -> None:
    """Assert results for get_all_combs_2_buckets function for different input values.

    Test implicitly tests function get_nr_all_combs_2_buckets
    """
    #Case 1
    combs = lu.get_all_combs_2_buckets(cols=["a", "b", "c", "d"], buck1_nr=2, buck2_nr=1)
    e_combs = [
        [["a", "b"], ["c"]], [["a", "b"], ["d"]], [["a", "c"], ["b"]],
        [["a", "c"], ["d"]], [["a", "d"], ["b"]], [["a", "d"], ["c"]],
        [["b", "c"], ["a"]], [["b", "c"], ["d"]], [["b", "d"], ["a"]],
        [["b", "d"], ["c"]], [["c", "d"], ["a"]], [["c", "d"], ["b"]]
    ]
    assert combs == e_combs
    #Case 2
    combs = lu.get_all_combs_2_buckets(cols=["a", "b", "c", "d"], buck1_nr=2, buck2_nr=2)
    e_combs = [[["a", "b"], ["c", "d"]], [["a", "c"], ["b", "d"]], [["a", "d"], ["b", "c"]]]
    assert combs == e_combs
    #Case buck1_nr>buck2_nr with larger values, assert only length
    #Case tests get_nr_all_combs_2_buckets
    cols = ["a", "b", "c", "d", "e"]
    buck1_nr, buck2_nr = 2, 1
    combs = lu.get_all_combs_2_buckets(cols=cols, buck1_nr=buck1_nr, buck2_nr=buck2_nr)
    assert len(combs) == lu.get_nr_all_combs_2_buckets(cols=cols, buck1_nr=buck1_nr, buck2_nr=buck2_nr)
    for comb in combs:
        assert len(comb[0]) == buck1_nr
        assert len(comb[1]) == buck2_nr
    #Case buck1_nr==buck2_nr with larger values, assert only length
    #Case tests get_nr_all_combs_2_buckets
    cols = ["a", "b", "c", "d", "e", "f"]
    buck1_nr, buck2_nr = 2, 2
    combs = lu.get_all_combs_2_buckets(cols=cols, buck1_nr=buck1_nr, buck2_nr=buck2_nr)
    assert len(combs) == lu.get_nr_all_combs_2_buckets(cols=cols, buck1_nr=buck1_nr, buck2_nr=buck2_nr)
    for comb in combs:
        assert len(comb[0]) == buck1_nr
        assert len(comb[1]) == buck2_nr

def test_get_n_sample_combinations() -> None:
    """Assert results for get_n_sample_combinations function for different input values."""
    ##Case AssertionError from wrong array input shapes
    cols = ["a", "b"]
    #Case len(buck1_nr_arr.shape)>1
    buck1_nr_arr = np.array([[1],[2]])
    buck2_nr_arr = np.array([])
    with pytest.raises(AssertionError) as e:
        lu.get_n_sample_combinations(cols=cols, buck1_nr_arr=buck1_nr_arr, buck2_nr_arr=buck2_nr_arr)
    assert e.type is AssertionError
    #Case buck1_nr_arr.shape[0]==0
    buck1_nr_arr = np.array([])
    buck2_nr_arr = np.array([])
    with pytest.raises(AssertionError) as e:
        lu.get_n_sample_combinations(cols=cols, buck1_nr_arr=buck1_nr_arr, buck2_nr_arr=buck2_nr_arr)
    assert e.type is AssertionError
    #Case buck1_nr_arr.shape!=buck2_nr_arr.shape
    buck1_nr_arr = np.array([1])
    buck2_nr_arr = np.array([])
    with pytest.raises(AssertionError) as e:
        lu.get_n_sample_combinations(cols=cols, buck1_nr_arr=buck1_nr_arr, buck2_nr_arr=buck2_nr_arr)
    assert e.type is AssertionError
    ##Case normal input results
    #Simple test
    buck1_nr_arr = np.array([1])
    buck2_nr_arr = np.array([1])
    combs_sample = lu.get_n_sample_combinations(cols=cols, buck1_nr_arr=buck1_nr_arr, buck2_nr_arr=buck2_nr_arr)
    assert (combs_sample==[[["a"], ["b"]]]) or (combs_sample==[[["b"], ["a"]]]) # noqa: PLR1714
    #More complex test
    cols = ["a", "b", "c", "d", "e"]
    buck1_nr_arr = np.array([3,1,3,4])
    buck2_nr_arr = np.array([1,1,2,1])
    combs_sample = lu.get_n_sample_combinations(cols=cols, buck1_nr_arr=buck1_nr_arr, buck2_nr_arr=buck2_nr_arr)
    assert len(combs_sample) == buck1_nr_arr.shape[0]
    for comb, buck1_nr, buck2_nr in zip(combs_sample, buck1_nr_arr, buck2_nr_arr):
        assert len(comb[0]) == buck1_nr
        assert len(comb[1]) == buck2_nr

def test_linkability_combinations_samples() -> None:
    """Assert results for linkability_combinations_samples function for different input values."""
    #Simple test
    cols = ["a", "b"]
    n_samples = 5
    combs_sample = lu.linkability_combinations_samples(cols=cols, n_samples=n_samples)
    assert (combs_sample==[[["a"], ["b"]]]) or (combs_sample==[[["b"], ["a"]]]) # noqa: PLR1714
    #More complex test
    cols = ["a", "b", "c", "d", "e"]
    n_samples = 5
    combs_sample = lu.linkability_combinations_samples(cols=cols, n_samples=n_samples)
    assert len(combs_sample) == n_samples * len(cols[1:])
    n_cols = 2
    for j,item in enumerate(combs_sample):
        assert len(item) == 2
        assert n_cols == len(item[0])+ len(item[1])
        if j>0 and (j+1)%n_samples==0:
            n_cols+=1

def test_linkability_risk_evaluation() -> None:
    """Assert results for linkability_risk_evaluation function for simple input case.

    Test also tests function load_linkability_results.
    """
    #Prepare test variables
    ori = get_adult(return_ori=True, n_samples=10)
    syn = get_adult(return_ori=False, n_samples=10)
    adults_nr_cols = len(ori.columns)
    n_samples = 3
    #Case save_results_json = False
    full_link_res = lu.linkability_risk_evaluation(
        ori = ori,
        syn = syn,
        n_samples = n_samples,
        n_attacks = 5
    )
    assert isinstance(full_link_res, lu.LinkabilityResults)
    #Calculate number of total attacks
    counter = len(list(range(2, adults_nr_cols+1)))
    nr_total_attacks = n_samples * counter
    #Output nr columns from pack_results
    pack_results_nr_cols = 7
    #Assert results
    res = np.array(full_link_res.res)
    assert res.shape == (nr_total_attacks, pack_results_nr_cols+1)
    assert len(full_link_res.aux_cols) == nr_total_attacks
    for nr_aux_cols, aux_cols in zip(res[:,-1], full_link_res.aux_cols):
        assert nr_aux_cols == len(aux_cols[0]) + len(aux_cols[1])
    assert len(full_link_res.res_cols) == pack_results_nr_cols+1
    e_res_cols = ["n_total", "n_main", "n_naive", "confidence_level", "main_rate", "naive_rate", "residual_rate", "nr_aux_cols"]
    assert full_link_res.res_cols == e_res_cols
    #Case save_results_json = True
    dataset = "test_linkability_risk_evaluation_adults"
    for path in [None, "/tmp"]: # noqa: S108
        file_path = u.aux_file_path(prefix="linkability", dataset=dataset, path=path)
        assert not os.path.exists(file_path)
        full_link_res = lu.linkability_risk_evaluation(
            dataset = dataset,
            ori = ori,
            syn = syn,
            n_samples = n_samples,
            n_attacks = 5,
            save_results_json = True,
            path = path
        )
        if path is None:
            path = u.DEFAULT_PATH_RESULTS[:-1]
        assert os.path.dirname(file_path) == path
        assert isinstance(full_link_res, lu.LinkabilityResults)
        assert os.path.exists(file_path)
        #Test load_linkability_results
        res = lu.load_linkability_results(dataset=dataset, path=path)
        assert isinstance(res, lu.LinkabilityResults)
        #Remove results file
        os.remove(file_path)
        assert not os.path.exists(file_path)
