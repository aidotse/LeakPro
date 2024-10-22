"""Tests for transformations module."""
# This file is part of Anonymeter and is released under BSD 3-Clause Clear License.
# Copyright (c) 2022 Anonos IP LLC.
# See https://github.com/statice/anonymeter/blob/main/LICENSE.md for details.
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform

from leakpro.synthetic_data_attacks.anonymeter.preprocessing.transformations import mixed_types_transform

rng = np.random.default_rng()

def test_scaling_numerical() -> None:
    """Assert result of mixed_types_transform function when types are numerical."""
    df_ori = pd.DataFrame({"c": rng.random(5)})
    df_syn = pd.DataFrame({"c": rng.random(5)})
    tdf_ori, tdf_syn = mixed_types_transform(df_ori, df_syn, num_cols=["c"], cat_cols=[])
    # values are scaled so that abs(difference) is between 0 and 1.
    # since this is a square distance matrix, there will be two elements with d=1
    vals = pd.concat([tdf_ori, tdf_syn])["c"].values
    dm = squareform(pdist(vals.reshape(-1,1), "cityblock"))
    expected_sum = 2
    assert np.sum(np.isclose(dm, 1)) == expected_sum
    assert np.min(dm) == 0

@pytest.mark.parametrize(
    ("df1", "df2", "exp1", "exp2"),
    [
        (
            pd.DataFrame({"c": ["a", "b", "c", "d"]}),
            pd.DataFrame({"c": ["a", "b", "c", "c"]}),
            pd.DataFrame({"c": [0, 1, 2, 3]}),
            pd.DataFrame({"c": [0, 1, 2, 2]}),
        ),
        (
            pd.DataFrame({"c": ["a", "b", "c", None]}),
            pd.DataFrame({"c": ["a", "b", "c", "c"]}),
            pd.DataFrame({"c": [0, 1, 2, 3]}),
            pd.DataFrame({"c": [0, 1, 2, 2]}),
        ),
        (
            pd.DataFrame({"c": ["a", "b", "c", "d"]}),
            pd.DataFrame({"c": ["a", "b", None, "c"]}),
            pd.DataFrame({"c": [0, 1, 2, 3]}),
            pd.DataFrame({"c": [0, 1, 4, 2]}),
        ),
    ],
)
def test_encoding_categorical(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    exp1: dict,
    exp2: dict
) -> None:
    """Assert result of mixed_types_transform function when types are categorical."""
    enc1, enc2 = mixed_types_transform(df1=df1, df2=df2, cat_cols=["c"], num_cols=[])
    pd.testing.assert_frame_equal(enc1, exp1)
    pd.testing.assert_frame_equal(enc2, exp2)
