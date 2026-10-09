##
## Copyright 2023-2026 Lindholmen Science Park AB
## SPDX-License-Identifier: Apache-2.0
##
"""Shared helpers for loading local, pre-split JSON member/non-member files.

Used by both ``prepare_target.py`` (``data.source: local``, fine-tune from scratch on your own
files) and ``import_external_target.py`` (an already-trained checkpoint + these same files), so the
tokenisation convention stays identical between the two.
"""

import json

import numpy as np


def load_json_texts(path: str, text_field: str = "text") -> list:
    """A list of strings, or a list of objects keyed by ``text_field`` -- adjust for your own files."""
    with open(path) as f:
        data = json.load(f)
    if not data:
        return []
    if isinstance(data[0], str):
        return [t for t in data if t.strip()]
    return [obj[text_field] for obj in data if obj.get(text_field, "").strip()]


def tokenise_one_per_text(texts: list, tokenizer, max_length: int) -> list:
    """Each text becomes exactly one (truncated) sequence -- never concatenated with another.

    Matches the reference WBC codebase's own local-JSON path (github.com/Stry233/WBC,
    ``trainer/misc/data.py:preprocess_dataset``: ``tokenizer(examples["text"], truncation=True,
    padding="max_length", max_length=max_length)``, one example in, one (possibly truncated) example
    out. This is *not* the EZ-MIA HF-dataset path's "concatenate everything, cut into max_length
    chunks" convention (``prepare_target.py:_tokenise``) -- that would silently merge unrelated JSON
    entries into single training rows, which the paper's own pipeline never does. Padding itself is
    left to the caller (``CausalLMCollate`` pads per-batch); only truncation happens here, so rows can
    come out shorter than `max_length` and must be handled as a ragged/variable-length population
    (``LLMDataHandler.as_object_array``), not stacked into one dense tensor.
    """
    chunks = []
    for t in texts:
        ids = tokenizer(t, truncation=True, max_length=max_length)["input_ids"]
        if len(ids) >= 2:
            chunks.append(np.asarray(ids, dtype=np.int64))
    return chunks
