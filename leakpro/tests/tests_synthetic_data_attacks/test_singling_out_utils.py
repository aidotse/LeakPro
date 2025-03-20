"""Tests for singling_out_utils module."""
import os

import numpy as np
import pytest

import leakpro.synthetic_data_attacks.singling_out_utils as sou
from leakpro.synthetic_data_attacks import utils as u
from leakpro.tests.tests_synthetic_data_attacks.anonymeter_tests.fixtures import get_adult


def test_check_for_int_value() -> None:
    """Assert results for check_for_int_value function."""
    #Assertion errors from input
    #Case x not int
    with pytest.raises(AssertionError):
        sou.check_for_int_value(x="not_int")
    #Case x negative integer
    with pytest.raises(AssertionError):
        sou.check_for_int_value(x=-1)
    #Case no assertion error
    sou.check_for_int_value(x=3)

def test_get_singling_out_suffix() -> None:
    """Assert results for get_singling_out_suffix function."""
    #Case n_cols=None
    prefix = sou.get_singling_out_suffix(n_cols=None)
    assert prefix == "all"
    #Case n_cols postive integer
    prefix = sou.get_singling_out_suffix(n_cols=3)
    assert prefix == "3"

def test_get_singling_out_prefix() -> None:
    """Assert results for get_singling_out_prefix function."""
    #Case n_cols=None
    prefix = sou.get_singling_out_prefix(n_cols=None)
    assert prefix == "singling_out_n_cols_all"
    #Case n_cols postive integer
    prefix = sou.get_singling_out_prefix(n_cols=3)
    assert prefix == "singling_out_n_cols_3"

def test_singling_out_risk_evaluation() -> None:
    """Assert results for singling_out_risk_evaluation function for simple input case.

    Test also tests function load_singling_out_results.
    """
    #Prepare test variables
    ori = get_adult(return_ori=True, n_samples=10)
    syn = get_adult(return_ori=False, n_samples=10)
    #Case n_cols not int
    with pytest.raises(AssertionError):
        sou.singling_out_risk_evaluation(
            ori = ori,
            syn = syn,
            n_cols = "not_int"
        )
    #Case n_cols negative int
    with pytest.raises(AssertionError):
        sou.singling_out_risk_evaluation(
            ori = ori,
            syn = syn,
            n_cols = 0
        )
    #Case n_cols==2
    with pytest.raises(ValueError, match="Parameter `n_cols` must be different than 2."):
        sou.singling_out_risk_evaluation(
            ori = ori,
            syn = syn,
            n_cols = 2
        )
    #Case n_cols positive int
    sin_out_res = sou.singling_out_risk_evaluation(
        ori = ori,
        syn = syn,
        n_cols = 1,
        n_attacks = 2
    )
    #Output nr columns from pack_results
    pack_results_nr_cols = 8
    assert isinstance(sin_out_res, sou.SinglingOutResults)
    #Assert results
    res = np.array(sin_out_res.res)
    assert res.shape == (1, pack_results_nr_cols+1)
    assert len(sin_out_res.res_cols) == pack_results_nr_cols+1
    e_res_cols = [
        "n_main_total", "n_main_success", "n_naive_total", "n_naive_success", "confidence_level",
        "main_rate", "naive_rate", "residual_rate", "n_cols"
    ]
    assert sin_out_res.res_cols == e_res_cols
    #Case n_cols None
    sin_out_res = sou.singling_out_risk_evaluation(
        ori = ori,
        syn = syn,
        n_attacks = 2
    )
    #Output nr columns from pack_results
    pack_results_nr_cols = 8
    assert isinstance(sin_out_res, sou.SinglingOutResults)
    #Assert results
    res = np.array(sin_out_res.res)
    e_rows = len(ori.columns)-1 #-1 because of n_cols==2 case
    assert res.shape == (e_rows, pack_results_nr_cols+1)
    assert len(sin_out_res.res_cols) == pack_results_nr_cols+1
    e_res_cols = [
        "n_main_total", "n_main_success", "n_naive_total", "n_naive_success", "confidence_level",
        "main_rate", "naive_rate", "residual_rate", "n_cols"
    ]
    assert sin_out_res.res_cols == e_res_cols
    #Case save_results_json = True
    dataset = "test_sor_evaluation_adults"
    prefix = sou.get_singling_out_prefix(n_cols=1)
    for path in [None, "/tmp"]: # noqa: S108
        file_path = u.aux_file_path(prefix=prefix, dataset=dataset, path=path)
        assert not os.path.exists(file_path)
        sin_out_res = sou.singling_out_risk_evaluation(
            dataset = dataset,
            ori = ori,
            syn = syn,
            n_cols = 1,
            n_attacks = 2,
            save_results_json = True,
            path = path
        )
        if path is None:
            path = u.DEFAULT_PATH_RESULTS[:-1]
        assert os.path.dirname(file_path) == path
        assert isinstance(sin_out_res, sou.SinglingOutResults)
        assert os.path.exists(file_path)
        #Test load_singling_out_results
        res = sou.load_singling_out_results(dataset=dataset, n_cols=1, path=path)
        assert isinstance(res, sou.SinglingOutResults)
        #Remove results file
        os.remove(file_path)
        assert not os.path.exists(file_path)
