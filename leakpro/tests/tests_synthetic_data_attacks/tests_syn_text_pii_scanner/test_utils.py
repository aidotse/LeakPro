"""Tests for syn_text_pii_scanner/utils module."""
import json
import os
from typing import List

import numpy as np
import pytest
import torch

import leakpro.tests.tests_synthetic_data_attacks.tests_syn_text_pii_scanner.data_tests.aux_data_test_utils as aux
from leakpro.synthetic_data_attacks.syn_text_pii_scanner import utils
from leakpro.synthetic_data_attacks.syn_text_pii_scanner.pii_token_classif_models import ner_longformer_model as lgfm

#Set TOKENIZERS_PARALLELISM to false to avoide warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def test_load_json_data() -> None:
    """Assert results for load_json_data function with mock data."""
    # Setup test variables
    test_file_path = "./test_load_json_data.json"
    test_data = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"}
    ]
    # Write the test data to a JSON file
    with open(test_file_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    # Load json data
    data = utils.load_json_data(file_path=test_file_path)
    # Assert loaded data matches the original data
    assert data == test_data
    os.remove(test_file_path)
    # Assert file does not exist
    assert not os.path.exists(test_file_path)

def test_load_data() -> None:
    """Assert results for load_data function with different input."""
    #Case data not Data type
    with pytest.raises(Exception, match="Input data must be of type Data."):
        utils.load_data(data="not Data type", tokenizer=lgfm.get_tokenizer())
    #Case data of Data type
    data = aux.data_factory()
    utils.load_data(data=data, tokenizer=lgfm.get_tokenizer())
    #Assert results
    for attr in ["ori", "syn"]:
        sd = getattr(data, attr)
        assert sd.raw_data is not None
        assert sd.dataset is not None
        assert sd.dataloader is not None
        first_batch = next(iter(sd.dataloader))
        if attr == "ori":
            assert sd.label_set is not None
            assert sd.label_key == "label"
            assert first_batch["input_ids"].shape[0] == 2
            assert first_batch["labels"].shape[0] == 2
        else:
            assert sd.label_set is None
            assert sd.label_key is None
            assert first_batch["input_ids"].shape[0] == 3
            assert first_batch.get("labels") is None

def test_load_model() -> None:
    """Assert results for load_model function with simple input."""
    model = utils.load_model(
        num_labels = 3,
        non_0_label_weight = 5
    )
    assert isinstance(model, lgfm.NERLongformerModel)
    assert model.num_labels == 3
    e_loss_fun_weight = torch.tensor([1.0, 5.0, 5.0], device=utils.device)
    assert torch.equal(model.loss_fun.weight, e_loss_fun_weight)

@pytest.mark.parametrize(
    ("verbose"), [False, True]
)
def test_forward_pass(*, verbose: bool) -> None:
    """Assert results for forward_pass function with different input."""
    #Case data not Data type
    with pytest.raises(Exception, match="Input data must be of type Data."):
        utils.forward_pass(data="not Data", num_labels=1, model="not model", verbose=verbose)
    #Case model not NERLongformerModel type
    data = aux.data_factory()
    num_labels = len(data.ori.label_set.f_labels)
    with pytest.raises(Exception, match="Model must be of type NERLongformerModel."):
        utils.forward_pass(data=data, num_labels=num_labels, model="not model", verbose=verbose)
    #Case load_data not run before forward apss
    model = utils.load_model(
        num_labels = num_labels
    )
    with pytest.raises(Exception, match="Load data must be run before forward pass."):
        utils.forward_pass(data=data, num_labels=num_labels, model=model, verbose=verbose)
    #Case no errors on input
    utils.load_data(data=data, tokenizer=lgfm.get_tokenizer())
    assert data.ori.model_input_output is None
    assert data.syn.model_input_output is None
    utils.forward_pass(data=data, num_labels=num_labels, model=model, verbose=verbose)
    assert isinstance(data.ori.model_input_output, utils.ModelInputOutput)
    assert isinstance(data.syn.model_input_output, utils.ModelInputOutput)
    assert data.ori.model_input_output.labels is not None
    assert data.syn.model_input_output.labels is None

@pytest.mark.parametrize(
    ("doc_nr", "end", "e_tokens", "raises_error"), [
        (49, 1, None, True),
        (50, 2, [1], False),
        (51, 3, [1, 2], False)
    ]
)
def test_PII(*, doc_nr: int, end: int, e_tokens: List[int], raises_error: bool) -> None: # noqa: N802
    """Test PII init method with different input."""
    #Get tokenizer
    tokenizer = lgfm.get_tokenizer()
    kwargs = {
        "doc_nr": doc_nr,
        "input": np.array([0,1,2]),
        "start": 1,
        "end": end,
        "tokenizer": tokenizer
    }
    if raises_error:
        with pytest.raises(Exception, match="End must be greater than start in PII."):
            utils.PII(**kwargs)
    else:
        pii = utils.PII(**kwargs)
        assert pii.doc_nr == doc_nr
        assert pii.start_end_tok_idx == (1, end)
        assert pii.tokens == e_tokens
        assert pii.text == tokenizer.decode(e_tokens).strip()

def test_get_PIIs_01() -> None: # noqa: N802
    """Test get_PIIs_01 function with different input."""
    #Get tokenizer
    tokenizer = lgfm.get_tokenizer()
    #Case labels and inputs different shape
    with pytest.raises(Exception, match="Labels and inputs must have the same shape."):
        utils.get_PIIs_01(labels=np.array([1]), inputs=np.array([[1]]), tokenizer=tokenizer)
    #Case labels and inputs same shape
    labels = np.array([
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 1, 1],
        [0, 1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ])
    rows, cols = labels.shape
    inputs = np.array([[j for j in range(i*cols, (i+1)*cols)] for i in range(rows)]) # noqa: C416
    piis = utils.get_PIIs_01(labels=labels, inputs=inputs, tokenizer=tokenizer)
    e_piis = [
        (0, (0,1), [0]),
        (0, (6,7), [6]),
        (1, (0,2), [7,8]),
        (1, (5,7), [12,13]),
        (2, (1,3), [15,16]),
        (2, (4,5), [18]),
        (3, (1,3), [22,23]),
        (3, (4,6), [25,26])
    ]
    assert len(piis) == len(e_piis)
    for pii, e_pii in zip(piis, e_piis):
        assert isinstance(pii, utils.PII)
        assert pii.doc_nr == e_pii[0]
        assert pii.start_end_tok_idx == e_pii[1]
        assert pii.tokens == e_pii[2]
        assert pii.text == tokenizer.decode(e_pii[2]).strip()
        assert len(pii.tokens) == pii.start_end_tok_idx[1] - pii.start_end_tok_idx[0]

def test_aux_piis_texts_and_doc_nrs_start_end() -> None:
    """Test aux_piis_texts_and_doc_nrs_start_end function with simple input."""
    #Get tokenizer
    tokenizer = lgfm.get_tokenizer()
    #Get piis
    kwargs = aux.kwargs_factory(tokenizer=tokenizer)
    piis = [utils.PII(**kwargs_) for kwargs_ in kwargs]
    #Get e_piis
    s_kwargs = sorted(kwargs, key=lambda x: x["doc_nr"])
    e_piis = [utils.PII(**kwargs_) for kwargs_ in s_kwargs]
    assert e_piis[0].doc_nr == 0
    assert e_piis[0].tokens == [10,11,12]
    #Run aux_piis_texts_and_doc_nrs_start_end
    s_piis, texts, doc_nrs_start_end = utils.aux_piis_texts_and_doc_nrs_start_end(piis=piis)
    #Assert results
    assert s_piis == e_piis
    assert len(s_piis) == len(piis)
    assert len(texts) == len(piis)
    assert len(s_kwargs) == len(piis)
    for text, s_kwarg in zip(texts, s_kwargs):
        input = s_kwarg["input"]
        s = s_kwarg["start"]
        e = s_kwarg["end"]
        assert text == tokenizer.decode(input[s:e]).strip()
    e_doc_nrs_start_end = {0: [0, 1], 1: [1, 2], 2: [2, 4], 3: [4, 6]}
    assert doc_nrs_start_end == e_doc_nrs_start_end

def test_detect_non_public_pii() -> None:
    """Test detect_non_public_pii function with simple input."""
    #Get tokenizer
    tokenizer = lgfm.get_tokenizer()
    #Get piis
    kwargs = aux.extended_kwargs_factory(tokenizer=tokenizer)
    piis = [utils.PII(**kwargs_) for kwargs_ in kwargs]
    s_kwargs = sorted(kwargs[0:6], key=lambda x: x["doc_nr"])
    e_non_public_piis = [utils.PII(**kwargs_) for kwargs_ in s_kwargs]
    non_public_piis = utils.detect_non_public_pii(piis=piis, similarity_threshold=0.95, min_nr_repetitions=1)
    assert len(piis) == len(non_public_piis) + 3
    assert non_public_piis == e_non_public_piis

def test_round_to_6() -> None:
    """Test round_to_6 function with simple input."""
    x = 0.123456789
    y = utils.round_to_6(x)
    assert abs(x-y) < 0.000001
    str_num = f"{y:.16f}".rstrip("0")  # Avoid floating-point inaccuracies
    assert len(str_num.split(".")[-1]) == 6

def test_calc_print_distribution() -> None:
    """Test calc_distribution and print_distribution functions with simple input."""
    array = np.arange(100)
    e_distr = {"mean": 49.5, "0": 0.0, "10": 10.0, "25": 25.0, "50": 50.0, "75": 75.0, "90": 90.0, "99": 99.0, "100": 99.0}
    distr = utils.calc_distribution(array=array)
    assert distr == e_distr
    utils.print_distribution(distr=distr)

def test_count_sort_similar_items() -> None:
    """Test count_sort_similar_items function with simple input."""
    #Get tokenizer
    tokenizer = lgfm.get_tokenizer()
    #Get piis
    kwargs = aux.extended_kwargs_factory(tokenizer=tokenizer)
    ori_piis = [utils.PII(**kwargs_) for kwargs_ in kwargs]
    syn_piis = [utils.PII(**kwargs_) for kwargs_ in kwargs]
    #Get sorted_sim_items
    sorted_sim_items = utils.count_sort_similar_items(
        similar_items = (np.array([0,0,0,1,1]), np.array([6,7,8,1,2])),
        ori_piis = ori_piis,
        syn_piis = syn_piis
    )
    e_sorted_sim_items = [
        {
            "ori_item": 0, "ori_doc_nr": 3, "ori_text": "<pad>", "syn_items": [6, 7, 8],
            "syn_docs": [0, 1, 2], "len_syn_items": 3, "len_syn_docs": 3
        },
        {
            "ori_item": 1, "ori_doc_nr": 2, "ori_text": "<unk>.", "syn_items": [1, 2],
            "syn_docs": [2], "len_syn_items": 2, "len_syn_docs": 1
        }
    ]
    assert sorted_sim_items == e_sorted_sim_items

@pytest.mark.parametrize(
    ("verbose"), [False, True]
)
def test_compare_piis_lists(*, verbose: bool) -> None:
    """Test compare_piis_lists function with simple input."""
    #Get tokenizer
    tokenizer = lgfm.get_tokenizer()
    #Get piis
    kwargs = aux.extended_kwargs_factory(tokenizer=tokenizer)
    ori_piis = [utils.PII(**kwargs_) for kwargs_ in kwargs]
    syn_piis = [utils.PII(**kwargs_) for kwargs_ in kwargs]
    sit, tot, sorted_sim_items, distr = utils.compare_piis_lists(
        ori_piis = ori_piis,
        syn_piis = syn_piis,
        similarity_threshold = 0.95,
        ignore_list = ["a in-"],
        verbose = verbose
    )
    assert sit == 14
    assert tot == 72
    e_items = aux.e_items_test_compare_piis_lists_fact()
    assert sorted_sim_items == e_items["e_sorted_sim_items"]
    assert distr.keys() == e_items["e_distr"].keys()
    for k in distr:
        assert abs(distr[k] - e_items["e_distr"][k])<0.00001
