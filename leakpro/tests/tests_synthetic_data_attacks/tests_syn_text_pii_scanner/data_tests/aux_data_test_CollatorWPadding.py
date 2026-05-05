#
# Copyright 2023-2026 AI Sweden
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Auxiliary data for test test_CollatorWPadding."""
from typing import List

from torch import tensor

from leakpro.synthetic_data_attacks.syn_text_pii_scanner.data_handling import EncodedDoc


def features_factory() -> List[EncodedDoc]: # noqa: D103
    return [
        EncodedDoc(
            input_ids=[0, 9226, 16, 3780, 321, 2],
            offsets=[(0, 3), (3, 7), (8, 10), (11, 19), (20, 21), (21, 25)],
            labels=[0, 1, 0, 0, 1, 0]
        ),
        EncodedDoc(
            input_ids=[0, 9226, 16, 10, 1181, 3780, 112, 2],
            offsets=[(0, 3), (3, 7), (8, 10), (11, 12), (13, 19), (20, 28), (29, 30), (30, 34)],
            labels=[0, 0, 3, 4, 0, 5, 0, 0]
        ),
        EncodedDoc(
            input_ids=[0, 9226, 16, 41, 190, 1181, 2422, 3780, 132, 2],
            offsets=[(0, 3), (3, 7), (8, 10), (11, 13), (14, 18), (19, 25), (26, 31), (32, 40), (41, 42), (42, 46)],
            labels=[0, 3, 4, 0, 0, 0, 5, 6, 6, 0]
        )
    ]

e_batch_0 = {
    "input_ids": tensor([
        [1, 1, 1, 1, 0, 9226, 16, 3780, 321, 2],
        [1, 1, 0, 9226, 16, 10, 1181, 3780, 112, 2],
        [0, 9226, 16, 41, 190, 1181, 2422, 3780, 132, 2]
    ]),
    "attention_mask": tensor([
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ])
}
e_batch_1 = e_batch_0.copy()
e_batch_1["labels"] = tensor([
    [-50, -50, -50, -50, 0, 1, 0, 0, 1, 0],
    [-50, -50, 0, 0, 3, 4, 0, 5, 0, 0],
    [0, 3, 4, 0, 0, 0, 5, 6, 6, 0]
])

e_batch_2 = {
    "input_ids": tensor([
        [0, 9226, 16, 3780, 321, 2, 1, 1, 1, 1],
        [0, 9226, 16, 10, 1181, 3780, 112, 2, 1, 1],
        [0, 9226, 16, 41, 190, 1181, 2422, 3780, 132, 2]
    ]),
    "attention_mask": tensor([
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ])
}
e_batch_3 = e_batch_2.copy()
e_batch_3["labels"] = tensor([
    [0, 1, 0, 0, 1, 0, -100, -100, -100, -100],
    [0, 0, 3, 4, 0, 5, 0, 0, -100, -100],
    [0, 3, 4, 0, 0, 0, 5, 6, 6, 0]
])
