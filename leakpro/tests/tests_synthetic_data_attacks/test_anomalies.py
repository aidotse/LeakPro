"""Tests for anomalies module."""
import numpy as np
import pandas as pd

import leakpro.synthetic_data_attacks.anomalies as anom

#Test df
df = pd.DataFrame(
    [[i, i<20, str(i<20), str(i)] for i in range(30)],
    columns=["non_cat", "cat_bool", "cat_less_threshold", "cat_more_threshold"]
)

def test_encode_categorical_columns() -> None:
    """Assert results for encode_categorical_columns function with simple input."""
    # Check that string columns are detected as string-like (object or StringDtype in newer pandas)
    assert df.dtypes[0] == np.dtype("int64")
    assert df.dtypes[1] == np.dtype("bool")
    assert pd.api.types.is_string_dtype(df.dtypes[2])
    assert pd.api.types.is_string_dtype(df.dtypes[3])
    df_encoded = anom.encode_categorical_columns(df=df)
    assert list(df_encoded.columns) == ["non_cat", "cat_bool", "cat_more_threshold", "cat_less_threshold_True"]
    assert df_encoded[["non_cat", "cat_bool"]].equals(df[["non_cat", "cat_bool"]])
    assert df_encoded["cat_more_threshold"].dtype == np.dtype("int64")
    assert sorted(df_encoded["cat_more_threshold"].values.tolist()) == list(range(30))
    assert df_encoded["cat_less_threshold_True"].dtype == np.dtype("float64")
    assert sorted(df_encoded["cat_less_threshold_True"].values.tolist()) == sorted(
        [1.0 if df["cat_less_threshold"][i]=="True" else 0.0 for i in range(30)]
    )

def test_return_anomalies_idx() -> None:
    """Assert no errors raised for return_anomalies function with simple input."""
    df_anomalies = anom.return_anomalies(df=df, n_estimators=100, n_jobs=2)
    assert df_anomalies.shape[1]==df.shape[1]
    assert df_anomalies.shape[0]<30
    assert df.shape[0]==30
