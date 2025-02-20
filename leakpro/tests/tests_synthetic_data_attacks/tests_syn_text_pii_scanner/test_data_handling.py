"""Tests for syn_text_pii_scanner/data_handling module."""
from typing import Any, Dict, List, Optional

import pytest
from torch import equal
from transformers import AutoTokenizer

import leakpro.tests.tests_synthetic_data_attacks.tests_syn_text_pii_scanner.data_tests.aux_data_test_CollatorWPadding as aux_data
from leakpro.synthetic_data_attacks.syn_text_pii_scanner import data_handling as dh

#Tokenizer object used in tests
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", use_fast=True, clean_up_tokenization_spaces=True)

@pytest.mark.parametrize(
    ("labels", "IOB2_FORMAT", "labels_to_id"),
    [
        (["MASK"], True, {"O": 0, "B-MASK": 1, "I-MASK": 2}),
        (["MASK"], False, {"O": 0, "MASK": 1}),
        (["DIRECT", "QUASI", "NO_MASK"], True, {"O": 0, "B-DIRECT": 1, "I-DIRECT": 2, "B-QUASI": 3, "I-QUASI": 4, "B-NO_MASK": 5, "I-NO_MASK": 6}) # noqa: E501
    ],
)
def test_LabelSet(*, labels: List[str], IOB2_FORMAT: bool, labels_to_id: Dict[str, int]) -> None: # noqa: N802, N803
    """Assert LabelSet init method with different input."""
    label_set = dh.LabelSet(labels=labels, IOB2_FORMAT=IOB2_FORMAT)
    assert label_set.labels == labels
    assert label_set.IOB2_FORMAT == IOB2_FORMAT
    assert label_set.labels_to_id == labels_to_id
    ids_to_label = {}
    for k, v in labels_to_id.items():
        ids_to_label[v] = k
    assert label_set.ids_to_label == ids_to_label
    assert label_set.f_labels == list(label_set.labels_to_id.keys())

def test_NERDataset_init_error() -> None: # noqa: N802
    """Assert NERDataset init method raises error for label_set and label_key inputs."""
    #Case label_set None, label_key not None
    with pytest.raises(Exception, match="label_set and label_key have to either be both None or both not None."):
        dh.NERDataset(input_data=None, tokenizer=None, label_key="notNone")
    #Case label_set not None, label_key None
    with pytest.raises(Exception, match="label_set and label_key have to either be both None or both not None."):
        dh.NERDataset(input_data=None, tokenizer=None, label_set="notNone")
    #Case label_set not LabelSet
    with pytest.raises(AssertionError, match="label_set needs to be a LabelSet."):
        dh.NERDataset(input_data=None, tokenizer=None, label_set="notNone", label_key="notNone")
    #Case label_key not string
    with pytest.raises(AssertionError, match="label_key needs to be a string."):
        dh.NERDataset(input_data=None, tokenizer=None, label_set=dh.LabelSet(labels=["MASK"]), label_key=1)

def test_NERDataset_pre_suf_fun() -> None: # noqa: N802
    """Assert NERDataset pre_suf_fun method results with simple input."""
    dataset = dh.NERDataset(input_data=[], tokenizer=tokenizer)
    #Test pre_suf_fun method
    text0 = "example of string"
    assert tokenizer.bos_token == "<s>" # noqa: S105
    assert tokenizer.eos_token == "</s>" # noqa: S105
    pre_suf_text0 = tokenizer.bos_token + text0 + tokenizer.eos_token
    assert dataset.pre_suf_fun(text0) == pre_suf_text0

@pytest.mark.parametrize(
    ("input_data", "label_set", "label_key", "max_length", "labelses", "idx"),
    [
        (
            [
                {"text": "this is document 0"},
                {"text": "this is a longer document 1"}
            ], None, None, 8, [None, None], 0
        ),
        (
            [
                {
                    "text": "this is document 0",
                    "annotations": [
                        {"label": "MASK", "start_offset": 0, "end_offset": 5},
                        {"label": "MASK", "start_offset": 17, "end_offset": 18},
                    ]
                },
                {
                    "text": "this is a longer document 1",
                    "annotations": [
                        {"label": "MASK", "start_offset": 5, "end_offset": 9},
                        {"label": "MASK", "start_offset": 17, "end_offset": 25},
                    ]
                },
                {
                    "text": "this is an even longer super document 2",
                    "annotations": [
                        {"label": "MASK", "start_offset": 0, "end_offset": 7},
                        {"label": "MASK", "start_offset": 23, "end_offset": 39},
                    ]
                }
            ],
            dh.LabelSet(labels=["MASK"]), "label", 10, [
                [0, 1, 0, 0, 1, 0],
                [0, 0, 1, 1, 0, 1, 0, 0],
                [0, 1, 1, 0, 0, 0, 1, 1, 1, 0]
            ],
            1
        ),
        (
            [
                {
                    "text": "this is document 0",
                    "annotations": [
                        {"other_label": "MASK", "start_offset": 0, "end_offset": 5},
                        {"other_label": "MASK", "start_offset": 17, "end_offset": 18},
                    ]
                },
                {
                    "text": "this is a longer document 1",
                    "annotations": [
                        {"other_label": "PER", "start_offset": 5, "end_offset": 9},
                        {"other_label": "LOC", "start_offset": 17, "end_offset": 25},
                    ]
                },
                {
                    "text": "this is an even longer super document 2",
                    "annotations": [
                        {"other_label": "PER", "start_offset": 0, "end_offset": 7},
                        {"other_label": "LOC", "start_offset": 23, "end_offset": 39},
                    ]
                }
            ],
            dh.LabelSet(labels=["MASK", "PER", "LOC"], IOB2_FORMAT=True), "other_label", 10, [
                [0, 1, 0, 0, 1, 0],
                [0, 0, 3, 4, 0, 5, 0, 0],
                [0, 3, 4, 0, 0, 0, 5, 6, 6, 0]
            ],
            2
        ),
    ],
)
def test_NERDataset(*, # noqa: N802
    input_data: List[Dict[str, Any]],
    label_set: Optional[dh.LabelSet],
    label_key: Optional[str],
    max_length: int,
    labelses: List[int],
    idx: int
) -> None:
    """Assert NERDataset init method results with different input, LabelSets and annotations.

    Test implicitly tests methods process_input_data, align_labels, __len__ and __getitem__.
    """
    dataset = dh.NERDataset(
        input_data = input_data,
        tokenizer = tokenizer,
        label_set = label_set,
        label_key= label_key
    )
    #Assert attributes
    assert dataset.tokenizer == tokenizer
    assert dataset.label_set == label_set
    assert dataset.label_key == label_key
    assert dataset.max_length == max_length
    assert len(dataset.data) == len(input_data)
    assert len(dataset.data) == len(labelses)
    for input_doc, doc, labels in zip(input_data, dataset.data, labelses):
        assert isinstance(doc, dh.EncodedDoc)
        encoding = tokenizer(
            input_doc["text"],
            add_special_tokens = False,
            return_attention_mask = False,
            return_offsets_mapping = True
        )
        #assert input_ids
        assert doc.input_ids[0] == 0
        assert doc.input_ids[-1] == 2
        assert doc.input_ids[1:-1] == encoding["input_ids"]
        #Assert labels
        assert doc.labels == labels
        if labels is not None:
            assert len(doc.labels) == len(doc.input_ids)
        #Assert offsets
        assert tokenizer.bos_token == "<s>" # noqa: S105
        len_bos_token = 3
        assert len(tokenizer.bos_token) == len_bos_token
        assert doc.offsets[0] == (0, len_bos_token)
        assert len(doc.offsets[1:-1]) == len(encoding["offset_mapping"])
        for doc_oft, enc_oft in zip(doc.offsets[1:-1], encoding["offset_mapping"]):
            assert doc_oft == (enc_oft[0]+len_bos_token, enc_oft[1]+len_bos_token)
        enc_oft = encoding["offset_mapping"][-1]
        assert doc.offsets[-1] == (enc_oft[1]+len_bos_token, enc_oft[1]+len_bos_token+len(tokenizer.eos_token))
    #Test __len__
    assert len(dataset) == len(input_data)
    #Test __getitem__
    assert dataset[idx] == dataset.data[idx]

@pytest.mark.parametrize(
    ("crossentropy_ignore_index", "padding_side", "e_batch"),
    [
        (-50, "left", aux_data.e_batch_0),
        (-50, "left", aux_data.e_batch_1),
        (-100, "right", aux_data.e_batch_2),
        (-100, "right", aux_data.e_batch_3),
    ],
)
def test_CollatorWPadding(*, # noqa: N802
    crossentropy_ignore_index: int,
    padding_side: str,
    e_batch: dict
) -> None:
    """Assert CollatorWPadding callable method result with different input and tokenizer padding side configuration."""
    features = aux_data.features_factory()
    if e_batch.get("labels") is None:
        for f in features:
            f.labels = None
    tokenizer.padding_side = padding_side
    collator = dh.CollatorWPadding(tokenizer=tokenizer, crossentropy_ignore_index=crossentropy_ignore_index)
    batch = collator(features)
    assert batch.keys() == e_batch.keys()
    for k,v in batch.items():
        assert equal(v, e_batch[k])
