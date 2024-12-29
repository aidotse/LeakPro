"""Auxiliary data for test_utils.py suite of tests."""
from typing import Any, Dict, List

from numpy import array
from transformers import PreTrainedTokenizerFast

from leakpro.synthetic_data_attacks.syn_text_pii_scanner import utils
from leakpro.synthetic_data_attacks.syn_text_pii_scanner.data_handling import LabelSet


def data_factory() -> utils.Data: # noqa: D103
    return utils.Data(
        ori = utils.SubData(
            path = "./data_tests/text_data.json",
            label_set = LabelSet(labels=["MASK"], IOB2_FORMAT=False),
            label_key = "label",
            batch_size = 2,
            shuffle = False,
            num_workers = 0
        ),
        syn = utils.SubData(
            path = "./data_tests/text_syn_data.json",
            batch_size = 3,
            shuffle = True,
            num_workers = 0
        )
    )

def kwargs_factory(*, tokenizer: PreTrainedTokenizerFast) -> List[Dict]: # noqa: D103
    return [
        {"doc_nr": 3, "input": array([0,1,2]), "start": 1, "end": 2, "tokenizer": tokenizer},
        {"doc_nr": 2, "input": array([3,4,5,6]), "start": 0, "end": 2, "tokenizer": tokenizer},
        {"doc_nr": 2, "input": array([3,4,5,6]), "start": 2, "end": 3, "tokenizer": tokenizer},
        {"doc_nr": 3, "input": array([0,1,2]), "start": 2, "end": 3, "tokenizer": tokenizer},
        {"doc_nr": 1, "input": array([7,8,9]), "start": 1, "end": 2, "tokenizer": tokenizer},
        {"doc_nr": 0, "input": array([10,11,12,13]), "start": 0, "end": 3, "tokenizer": tokenizer}
    ]

def extended_kwargs_factory(*, tokenizer: PreTrainedTokenizerFast) -> List[Dict]: # noqa: D103
    to_return = kwargs_factory(tokenizer=tokenizer)
    to_return += [
        {"doc_nr": 0, "input": array([0, 10980, 4, 610, 1259, 2]), "start": 0, "end": 6, "tokenizer": tokenizer},
        {"doc_nr": 1, "input": array([0, 10980, 4, 344, 1259, 2]), "start": 0, "end": 6, "tokenizer": tokenizer},
        {"doc_nr": 2, "input": array([0, 10980, 4, 1259, 2]), "start": 0, "end": 5, "tokenizer": tokenizer}
    ]
    return to_return

def e_items_test_compare_piis_lists_fact() -> Dict[str, Any]: # noqa: D103
    return {
        "e_sorted_sim_items": [
            {
                "ori_item": 5, "ori_doc_nr": 0, "ori_text": "<s>Mr. John Smith</s>", "syn_items": [6, 7, 8],
                "syn_docs": [0, 1, 2], "len_syn_items": 3, "len_syn_docs": 3
            },
            {
                "ori_item": 6, "ori_doc_nr": 1, "ori_text": "<s>Mr. J Smith</s>", "syn_items": [6, 7, 8],
                "syn_docs": [0, 1, 2], "len_syn_items": 3, "len_syn_docs": 3
            },
            {
                "ori_item": 7, "ori_doc_nr": 2, "ori_text": "<s>Mr. Smith</s>", "syn_items": [6, 7, 8],
                "syn_docs": [0, 1, 2], "len_syn_items": 3, "len_syn_docs": 3
            },
            {
                "ori_item": 0, "ori_doc_nr": 3, "ori_text": "<pad>", "syn_items": [0],
                "syn_docs": [3], "len_syn_items": 1, "len_syn_docs": 1
            },
            {
                "ori_item": 1, "ori_doc_nr": 2, "ori_text": "<unk>.", "syn_items": [1],
                "syn_docs": [2], "len_syn_items": 1, "len_syn_docs": 1
            },
            {
                "ori_item": 2, "ori_doc_nr": 2, "ori_text": "the", "syn_items": [2],
                "syn_docs": [2], "len_syn_items": 1, "len_syn_docs": 1
            },
            {
                "ori_item": 3, "ori_doc_nr": 3, "ori_text": "</s>", "syn_items": [3],
                "syn_docs": [3], "len_syn_items": 1, "len_syn_docs": 1
            },
            {
                "ori_item": 4, "ori_doc_nr": 1, "ori_text": "and", "syn_items": [4],
                "syn_docs": [1], "len_syn_items": 1, "len_syn_docs": 1
            }
        ],
        "e_distr": {
            "mean": 0.527316, "0": 0.156378, "10": 0.256064, "25": 0.294961, "50": 0.446959,
            "75": 0.755499, "90": 1.0, "99": 1.0, "100": 1.0
        }
    }
