"""Tests for inference_utils module."""
import math
import os
from typing import List

import numpy as np
import pytest

import leakpro.synthetic_data_attacks.inference_utils as iu
from leakpro.synthetic_data_attacks import utils as u
from leakpro.tests.tests_synthetic_data_attacks.anonymeter_tests.fixtures import get_adult


#Auxiliary function
def get_min_n_samples_nr_combs(*, cols: List[str], n_samples: int, k: int) -> int:
    """Auxiliary function used in test suite."""
    nr_combs = math.comb(len(cols), k)
    return min(nr_combs, n_samples)

def test_get_inference_prefix() -> None:
    """Assert results for get_inference_prefix function."""
    prefix = iu.get_inference_prefix(worst_case_flag=True)
    assert prefix == "inference_worst_case"
    prefix = iu.get_inference_prefix(worst_case_flag=False)
    assert prefix == "inference_base_case"

def test_inference_risk_evaluation() -> None: # noqa: PLR0915
    """Assert results for inference_risk_evaluation function for simple input case.

    Test also tests function load_inference_results.
    """
    #Prepare test variables
    ori = get_adult(return_ori=True, n_samples=10)
    syn = get_adult(return_ori=False, n_samples=10)
    n_attacks = 2
    n_samples = 2
    adults_nr_cols = len(ori.columns)
    #Output nr columns from pack_results
    pack_results_nr_cols = 8
    #Get expected len_all_samples
    e_base_case_len = 0
    for secret in ori.columns:
        aux_cols = [col for col in ori.columns if col != secret]
        for i in range(1, len(aux_cols)+1):
            e_base_case_len += get_min_n_samples_nr_combs(cols=aux_cols, n_samples=n_samples, k=i)
    #Assemple zip lists
    worst_case_flags = [True, False]
    e_lens = [adults_nr_cols, e_base_case_len]
    assert len(worst_case_flags) == len(e_lens)
    for worst_case_flag, e_len in zip(worst_case_flags, e_lens):
        #Case save_results_json = False
        inf_res = iu.inference_risk_evaluation(
            ori = ori,
            syn = syn,
            worst_case_flag = worst_case_flag,
            n_attacks = n_attacks,
            n_samples = n_samples
        )
        assert isinstance(inf_res, iu.InferenceResults)
        #Assert results
        res = np.array(inf_res.res)
        assert res.shape == (e_len, pack_results_nr_cols+1)
        assert e_len>0
        if worst_case_flag:
            assert (res[:,-1] == adults_nr_cols-1).all()
        else:
            set_secrets = np.unique(res[:,-1], return_counts=True)
            e_counts = np.ones(shape=adults_nr_cols-1)*adults_nr_cols*n_samples
            e_counts[-1] = adults_nr_cols
            assert np.equal(set_secrets[0], np.arange(1,adults_nr_cols)).all()
            assert np.equal(set_secrets[1], e_counts).all()
        assert len(inf_res.res_cols) == pack_results_nr_cols+1
        e_res_cols = [
            "n_main_total", "n_main_success", "n_naive_total", "n_naive_success", "confidence_level",
            "main_rate", "naive_rate", "residual_rate", "nr_aux_cols"
        ]
        assert inf_res.res_cols == e_res_cols
        assert len(inf_res.aux_cols) == e_len
        assert len(inf_res.secrets) == e_len
        for aux_cols, secret in zip(inf_res.aux_cols, inf_res.secrets):
            assert secret not in aux_cols
            if worst_case_flag:
                assert len(aux_cols) == e_len-1
                assert set(aux_cols+[secret]) == set(ori.columns)
            else:
                assert set(aux_cols).issubset(set(ori.columns))
        #Case save_results_json = True
        dataset = "test_ir_evaluation_adults"
        prefix = iu.get_inference_prefix(worst_case_flag=worst_case_flag)
        for path in [None, "/tmp"]: # noqa: S108
            file_path = u.aux_file_path(prefix=prefix, dataset=dataset, path=path)
            assert not os.path.exists(file_path)
            inf_res = iu.inference_risk_evaluation(
                dataset = dataset,
                ori = ori,
                syn = syn,
                worst_case_flag = worst_case_flag,
                n_attacks = n_attacks,
                n_samples = n_samples,
                save_results_json = True,
                path = path
            )
            if path is None:
                path = u.DEFAULT_PATH_RESULTS[:-1]
            assert os.path.dirname(file_path) == path
            assert isinstance(inf_res, iu.InferenceResults)
            assert os.path.exists(file_path)
            #Test load_inference_results
            res = iu.load_inference_results(dataset=dataset, worst_case_flag=worst_case_flag, path=path)
            assert isinstance(res, iu.InferenceResults)
            #Remove results file
            os.remove(file_path)
            assert not os.path.exists(file_path)

@pytest.mark.parametrize(
    ("cols", "k", "n"),
    [
        (["a", "b"], 0, 0),
        (["a", "b"], 0, 1),
        (["a", "b"], 1, 2),
        (["a", "b", "c"], 2, 3),
        (["a", "b", "c", "d", "e"], 3, 5)
    ],
)
def test_get_n_random_combinations(cols: List[str], k: int, n: int) -> None:
    """Assert results for get_n_random_combinations function for different input values."""
    sample = iu.get_n_random_combinations(cols=cols, k=k, n=n)
    assert len(sample) == n
    sample_u = []
    set_cols = set(cols)
    for sample_ in sample:
        assert len(sample_) == k
        assert set(sample_).issubset(set_cols)
        sorted_sample = sorted(sample_)
        assert sorted_sample not in sample_u
        sample_u.append(sorted_sample)

@pytest.mark.parametrize(
    ("cols", "n_samples"),
    [
        (["a"], 0),
        (["a"], 1),
        (["a"], 2),
        (["a", "b"], 0),
        (["a", "b"], 1),
        (["a", "b"], 2),
        (["a", "b", "c"], 3),
        (["a", "b", "c", "d", "e"], 10),
        (["a", "b", "c", "d", "e", "f", "g"], 20)
    ],
)
def test_get_samples_length_subsets_cols(cols: List[str], n_samples: int) -> None:
    """Assert results for get_samples_length_subsets_cols function for different input values."""
    #Get expected len_all_samples
    e_len_all_samples = 0
    for i in range(1, len(cols)+1):
        e_len_all_samples += get_min_n_samples_nr_combs(cols=cols, n_samples=n_samples, k=i)
    #Get all_samples and assert results
    all_samples = iu.get_samples_length_subsets_cols(cols=cols, n_samples=n_samples)
    assert len(all_samples) == e_len_all_samples
    sample_u = []
    set_cols = set(cols)
    k = 1
    count_k = 0
    nr_combs_k = get_min_n_samples_nr_combs(cols=cols, n_samples=n_samples, k=k)
    for sample in all_samples:
        new_k = len(sample)
        if new_k != k:
            assert nr_combs_k == count_k
            assert new_k == k+1
            count_k = 0
            nr_combs_k = get_min_n_samples_nr_combs(cols=cols, n_samples=n_samples, k=new_k)
        count_k += 1
        k = new_k
        assert set(sample).issubset(set_cols)
        sorted_sample = sorted(sample)
        assert sorted_sample not in sample_u
        sample_u.append(sorted_sample)

@pytest.mark.parametrize(
    ("div", "num", "e_progress"),
    [
        (["a", "b", "c"], 0, 0),
        (["a", "b", "c"], 1, 33.33),
        (["a", "b", "c"], 1.5, 50),
        (["a", "b", "c"], 2, 66.67),
        (["a", "b", "c"], 3, 100),
    ],
)
def test_get_progress(div: List, num:float, e_progress: float) -> None:
    """Assert results for get_progress function for different input values."""
    progress = iu.get_progress(num=num, div=div)
    assert progress == e_progress
