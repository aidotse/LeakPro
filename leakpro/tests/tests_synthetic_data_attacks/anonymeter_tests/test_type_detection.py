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
"""Tests for type_detection module."""
# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details.
import numpy as np
import pandas as pd
import pytest

from leakpro.synthetic_data_attacks.anonymeter.preprocessing.type_detection import detect_col_types, detect_consistent_col_types

rng = np.random.default_rng()

@pytest.mark.parametrize(
    ("df", "expected"),
    [
        (pd.DataFrame({"num": rng.random(5), "cat": list("abcde")}), {"cat": ["cat"], "num": ["num"]}),
        (pd.DataFrame({"num1": rng.random(5), "num2": [1, 2, 3, 4, 5]}), {"cat": [], "num": ["num1", "num2"]}),
        (
            pd.DataFrame({"num1": rng.random(5), "num2": [1, 2, 3, 4, 5]}).astype("object"),
            {"cat": ["num1", "num2"], "num": []},
        ),
        (
            pd.DataFrame({"cat1": list("abcde"), "cat2": ["1", "2", "3", "4", "5"]}),
            {"cat": ["cat1", "cat2"], "num": []},
        ),
    ],
)
def test_detect_col_types(df: pd.DataFrame, expected: dict) -> None:
    """Assert results of detect_col_types function."""
    ctypes = detect_col_types(df)
    assert ctypes == expected

def test_detect_col_types_consistent() -> None:
    """Assert result of detect_consistent_col_types function when types are the same."""
    df1 = pd.DataFrame({"num": rng.random(5), "cat": list("abcde")})
    df2 = pd.DataFrame({"num": rng.random(5), "cat": list("fghil")})
    assert detect_consistent_col_types(df1, df2) == {"cat": ["cat"], "num": ["num"]}

def test_detect_col_types_consistent_raises() -> None:
    """Assert detect_consistent_col_types returns RunTimeError when types are different."""
    df1 = pd.DataFrame({"num": rng.random(5), "cat": list("abcde")})
    df2 = pd.DataFrame({"num": [str(_) for _ in rng.random(5)], "cat": list("fghil")})
    with pytest.raises(RuntimeError) as e:
        detect_consistent_col_types(df1, df2)
    assert str(e.value) == "Input dataframes have different column names/types."
