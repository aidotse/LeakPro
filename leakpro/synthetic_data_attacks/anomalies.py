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
"""Module with functions for estimating anomalies."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def encode_categorical_columns(*, df: pd.DataFrame, threshold_one_hot_encoding: int = 20) -> pd.DataFrame:
    """Auxiliary function that encodes categorical columns in a DataFrame with a OneHotEncoder alternatively LabelEncoder."""
    # Make a copy of the DataFrame to avoid modifying the original
    df_encoded = df.copy()
    # Iterate through each column in the DataFrame
    for column in df_encoded.columns:
        if pd.api.types.is_string_dtype(df_encoded[column]):
            unique_values = df_encoded[column].nunique()
            if unique_values <= threshold_one_hot_encoding:
                # Perform one-hot encoding
                ohe = OneHotEncoder(sparse_output=False, drop="first")  # drop="first" to avoid dummy variable trap
                transformed_data = ohe.fit_transform(df_encoded[[column]])
                # Create a DataFrame with the one-hot encoded columns
                ohe_df = pd.DataFrame(transformed_data, columns=ohe.get_feature_names_out([column]))
                # Concatenate the new one-hot encoded columns to the DataFrame
                df_encoded = pd.concat([df_encoded.drop(columns=[column]), ohe_df], axis=1)
            else:
                # Perform label encoding
                le = LabelEncoder()
                df_encoded[column] = le.fit_transform(df_encoded[column])
    return df_encoded

def return_anomalies(*, df: pd.DataFrame, anomaly_threshold: float = -0.7, **kwargs: dict) -> pd.DataFrame:
    """Auxiliary function that returns anomalies entries for given DF, using IsolationForest as anomaly detector algorithm."""
    #Set kwargs
    if "n_jobs" not in kwargs:
        kwargs["n_jobs"] = -1
    if "n_estimators" not in kwargs:
        kwargs["n_estimators"] = 10_000
    #Encode df
    df_encoded = encode_categorical_columns(df=df)
    #Fit Isolation forest and predict
    clf = IsolationForest(**kwargs).fit(df_encoded)
    prediction = clf.predict(df_encoded)
    if kwargs.get("verbose", False):
        unique_pred = np.unique(prediction, return_counts=True)
        print("Unique predictions", unique_pred) # noqa: T201
    #Get anomalies index
    anom_idx = np.where(prediction<=anomaly_threshold)[0]
    return df.iloc[anom_idx, :]
