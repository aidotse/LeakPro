import pandas as pd


path = "data/physionet.org/files/mimiciv/3.1/"

# Load required tables
#admissions = pd.read_csv(path + "hosp/admissions.csv", nrows=100)

#print(admissions.head())
#print(admissions.columns)

lab_events = pd.read_csv(
    path + "hosp/labevents.csv",
    #nrows=10000000,
    usecols=["hadm_id", "itemid", "flag", "priority"],
    dtype={"hadm_id": str}  # Ensure 'hadm_id' is treated as a string
).dropna(subset=["hadm_id"])  # Drop rows where 'hadm_id' is NaN

# Filter rows where 'flag' is 'abnormal'
lab_events_abnormal = lab_events[lab_events["flag"] == "abnormal"]

# One-hot encode the 'itemid' column
lab_events_one_hot = pd.get_dummies(lab_events_abnormal, columns=["itemid"], prefix="", prefix_sep="")

# Select only numeric columns for aggregation
numeric_columns = lab_events_one_hot.drop(columns=["hadm_id", "flag", "priority"])

# Add 'hadm_id' back to the numeric columns for grouping
lab_events_one_hot = pd.concat([lab_events_one_hot["hadm_id"], numeric_columns], axis=1)

# Group by 'hadm_id' and aggregate using logical OR (max for binary data)
lab_events_grouped = lab_events_one_hot.groupby("hadm_id").max().reset_index()

print(lab_events_grouped)

# Pickle lab_events_grouped
lab_events_grouped.to_pickle("data/lab_events_grouped.pkl")

print("DataFrame has been pickled to 'data/lab_events_grouped.pkl'")

# read data/private_df.pkl

#private_df = pd.read_pickle("data/private_df.pkl")

# print counts of identity column
#print(private_df["identity"].value_counts())

# remove rows where identity value count is 1
#private_df = private_df.groupby("identity").filter(lambda x: len(x) > 49)

# print counts of identity column
#print(private_df["identity"].value_counts())

"""
# print all column names in the dataframe individually
#for col in private_df.columns:
#    print(col)

from sklearn.ensemble import RandomForestClassifier

X = private_df.drop(columns=["identity"])
from sklearn.preprocessing import OrdinalEncoder

# Apply ordinal encoding
encoder = OrdinalEncoder(dtype=int)
X[['gender']] = encoder.fit_transform(X[['gender']])
X[['insurance']] = encoder.fit_transform(X[['insurance']])
X[['race']] = encoder.fit_transform(X[['race']])

print(X.head())

y = private_df["identity"]

# drop systolic and diastolic columns
X = X.drop(columns=["systolic", "diastolic"])

rf = RandomForestClassifier(n_estimators=3, random_state=42, n_jobs=-1, verbose=2)
rf.fit(X, y)

print("Fitted model")

import numpy as np
# print feature importances in descending order
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X.shape[1]):
    print(f"{X.columns[indices[f]]}: {importances[indices[f]]}")
"""