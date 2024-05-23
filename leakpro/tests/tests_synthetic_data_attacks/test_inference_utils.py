"""Tests for inference_utils module."""
import os

import numpy as np

import leakpro.synthetic_data_attacks.inference_utils as iu
from leakpro.synthetic_data_attacks.utils import aux_file_path
from leakpro.tests.tests_synthetic_data_attacks.anonymeter_tests.fixtures import get_adult


def test_inference_risk_evaluation_each_against_rest_columns() -> None:
    """Assert results for inference_risk_evaluation_each_against_rest_columns function for simple input case."""
    #Prepare test variables
    ori = get_adult(return_ori=True, n_samples=10)
    syn = get_adult(return_ori=False, n_samples=10)
    adults_nr_cols = len(ori.columns)
    #Case save_results_json = False
    inf_res = iu.inference_risk_evaluation_each_against_rest_columns(
        ori = ori,
        syn = syn,
        n_attacks = 5
    )
    assert isinstance(inf_res, iu.InferenceResults)
    #Output nr columns from pack_results
    pack_results_nr_cols = 7
    #Assert results
    res = np.array(inf_res.res)
    assert res.shape == (adults_nr_cols, pack_results_nr_cols+1)
    assert (res[:,-1] == adults_nr_cols-1).all()
    assert len(inf_res.res_cols) == pack_results_nr_cols+1
    e_res_cols = ["n_total", "n_main", "n_naive", "confidence_level", "main_rate", "naive_rate", "residual_rate", "nr_aux_cols"]
    assert inf_res.res_cols == e_res_cols
    assert len(inf_res.aux_cols) == adults_nr_cols
    assert len(inf_res.secrets) == adults_nr_cols
    for aux_cols, secret in zip(inf_res.aux_cols, inf_res.secrets):
        assert len(aux_cols) == adults_nr_cols-1
        assert secret not in aux_cols
        assert set(aux_cols+[secret]) == set(ori.columns)
    #Case save_results_json = True
    dataset = "test_ir_evaluation_each_against_rest_columns_adults"
    _, file_path = aux_file_path(prefix="inference_each_against_rest", dataset=dataset)
    assert not os.path.exists(file_path)
    inf_res = iu.inference_risk_evaluation_each_against_rest_columns(
        dataset = dataset,
        ori = ori,
        syn = syn,
        n_attacks = 5,
        save_results_json = True
    )
    assert isinstance(inf_res, iu.InferenceResults)
    assert os.path.exists(file_path)
    #Remove results file
    os.remove(file_path)
    assert not os.path.exists(file_path)

def test_load_inference_results() -> None:
    """Assert results for load_inference_results function for dataset used in examples."""
    inf_res = iu.load_inference_results(dataset="adults")
    assert isinstance(inf_res, iu.InferenceResults)
