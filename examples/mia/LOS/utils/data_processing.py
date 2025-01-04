"""
This file is inspired by https://github.com/MLforHealth/MIMIC_Extract 
MIT License
Copyright (c) 2019 MIT Laboratory for Computational Physiology
"""
#TODO: Do I need to include the license for this file.?
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch import Tensor, from_numpy
from torch.utils.data import DataLoader, Dataset, Subset
from utils.model_grud import to_3D_tensor
from tqdm import tqdm


class MimicDataset(Dataset):
    def __init__(self, x, y):
        # Check if x is already a tensor
        if not isinstance(x, Tensor):
            self.x = from_numpy(x).float()  # Convert features to torch tensors if needed
        else:
            self.x = x.float()  # Ensure it is of type float

        # Check if y is already a tensor
        if not isinstance(y, Tensor):
            self.y = from_numpy(y).float()  # Convert labels to torch tensors if needed
        else:
            self.y = y.float()  # Ensure it is of type float

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx].squeeze(0)

    def subset(self, indices):
        return MimicDataset(self.x[indices], self.y[indices])


def get_mimic_dataset(data_path,
                      train_frac,
                      validation_frac,
                      test_frac,
                      early_stop_frac,
                      use_LR = True):
    """Get the dataset, download it if necessary, and store it."""

    # Assert that the sum of all fractions is between 0 and 1
    total_frac = train_frac + validation_frac + test_frac + early_stop_frac
    assert 0 < total_frac <= 1, "The sum of dataset fractions must be between 0 and 1."

    if use_LR:
        path = data_path + "flattened/"
    else:
        path = data_path + "unflattened/"
    dataset_path = os.path.join(path, "dataset.pkl")
    indices_path = os.path.join(path, "indices.pkl")

    if os.path.exists(dataset_path) and os.path.exists(indices_path):
        print("Loading dataset...")
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)  # Load the dataset
        with open(indices_path, "rb") as f:
            indices_dict = pickle.load(f)  # Load the dictionary containing indices
            train_indices = indices_dict["train_indices"]  # Get the actual train indices
            validation_indices = indices_dict["validation_indices"]  # Get the actual validation indices
            test_indices = indices_dict["test_indices"]    # Get the actual test indices
            early_stop_indices = indices_dict["early_stop_indices"]  # Get the actual early stop indices
        print(f"Loaded dataset from {dataset_path}")
        return dataset, train_indices, validation_indices ,test_indices, early_stop_indices

    data_file_path = os.path.join(data_path, "all_hourly_data.h5")
    if os.path.exists(data_file_path):
        print("Loading data...")
        data = pd.read_hdf(data_file_path, "vitals_labs")
        statics = pd.read_hdf(data_file_path, "patients")

        ID_COLS = ["subject_id", "hadm_id", "icustay_id"]

        print("Splitting data...")
        train_data, holdout_data, y_train, y_holdout_data = data_splitter(statics,
                                                                 data,
                                                                 train_frac)
        
        print("Normalizing data...")
        train_data , holdout_data = data_normalization(train_data, holdout_data)

        print("Imputing missing values...")
        train_data, holdout_data = [
        simple_imputer(df, ID_COLS) for df in tqdm((train_data, holdout_data), desc="Imputation")]

        if use_LR:
            # Apply pivot_table to flatten the data
            print("Flattening data for LR...")
            flat_train, flat_holdout = [
                df.pivot_table(index=ID_COLS, columns=["hours_in"])
                for df in tqdm((train_data, holdout_data), desc="Flattening")
            ]
            print("Flattening data...")
            train, holdout, label_train, label_holdout = [
                flatten_multi_index(df)
                for df in tqdm((flat_train, flat_holdout, y_train, y_holdout_data), desc="Flattening Index")
            ]
        else:
            # Skip pivot_table if flatten is False
            train, holdout, label_train, label_holdout = train_data, holdout_data, y_train, y_holdout_data

        assert_no_missing_values(train_data, holdout_data, train, holdout)

        train_df, holdout_df = standard_scaler(train, holdout)

        # Creating the dataset
        data_x = pd.concat((train_df, holdout_df), axis=0)
        data_y = pd.concat((label_train, label_holdout), axis=0)

        assert np.issubdtype(data_x.values.dtype, np.number), "Non-numeric data found in features."
        assert np.issubdtype(data_y.values.dtype, np.number), "Non-numeric data found in labels."

        print("Creating dataset...")
        if use_LR:
            dataset = MimicDataset(data_x.values, data_y.values)
        else:
            data_x = to_3D_tensor(data_x)
            dataset = MimicDataset(data_x, data_y.values)

        # Generate indices for training, validation, test, and early stopping
        train_indices, validation_indices, test_indices, early_stop_indices = data_indices(data_x,
                                                                       train_frac,
                                                                       validation_frac,
                                                                       test_frac,
                                                                       early_stop_frac)

        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        # Save the dataset to dataset.pkl
        print("Saving dataset and indices...")
        with open(dataset_path, "wb") as file:
            pickle.dump(dataset, file)
            print(f"Saved dataset to {dataset_path}")

        # Save train and test indices to indices.pkl
        indices_to_save = {
            "train_indices": train_indices,
            "validation_indices": validation_indices,
            "test_indices": test_indices,
            "early_stop_indices": early_stop_indices,
        }
        with open(indices_path, "wb") as file:
            pickle.dump(indices_to_save, file)
            print(f"Saved train and test indices to {indices_path}")
    else:
        msg = "Please download the MIMIC-III dataset from https://physionet.org/content/mimiciii/1.4/ and save it in the specified path."
        raise FileNotFoundError(msg)
    return dataset, train_indices, validation_indices, test_indices, early_stop_indices


def data_splitter(statics,
                  data,
                  train_frac):
    GAP_TIME = 6  # In hours
    WINDOW_SIZE = 24 # In hours
    SEED = 1

    Ys = statics[statics.max_hours > WINDOW_SIZE + GAP_TIME][["los_icu"]]
    Ys["los_3"] = Ys["los_icu"] > 3
    Ys.drop(columns=["los_icu"], inplace=True)
    Ys["los_3"] = Ys["los_3"].astype(float)

    lvl2 = data[
        (data.index.get_level_values("icustay_id").isin(set(Ys.index.get_level_values("icustay_id")))) &
        (data.index.get_level_values("hours_in") < WINDOW_SIZE)
    ]

    data_subj_idx, y_subj_idx = [df.index.get_level_values("subject_id") for df in (lvl2, Ys)]
    data_subjects = set(data_subj_idx)
    assert data_subjects == set(y_subj_idx), "Subject ID pools differ!"

    # Randomly shuffle subjects and compute the sizes of the splits
    np.random.seed(SEED)
    subjects = np.random.permutation(list(data_subjects))
    N = len(subjects)
    N_train = int(train_frac * N)

    # Ensure no overlap between train and test sets
    train_subj = subjects[:N_train]
    test_subj = subjects[N_train::]

    # Split the data according to the subjects
    (train_data, holdout_data), (y_train, y_holdout) = [
        [df[df.index.get_level_values("subject_id").isin(s)] for s in (train_subj, test_subj)]
        for df in (lvl2, Ys)
    ]
    return train_data, holdout_data, y_train, y_holdout

# def simple_imputer(dataframe,
#                    ID_COLS):
#     idx = pd.IndexSlice
#     df = dataframe.copy()
#     if len(df.columns.names) > 2: df.columns = df.columns.droplevel(("label", "LEVEL1", "LEVEL2"))

#     df_out = df.loc[:, idx[:, ["mean", "count"]]]
#     icustay_means = df_out.loc[:, idx[:, "mean"]].groupby(ID_COLS).mean()

#     df_out.loc[:, idx[:, "mean"]] = (
#             df_out.loc[:, idx[:, "mean"]]
#             .groupby(ID_COLS)
#             .ffill()  # Replace forward fill method
#             .groupby(ID_COLS)
#             .fillna(icustay_means)  # Fill remaining NaNs with icustay_means
#             .fillna(0)  # Fill any remaining NaNs with 0
#         ) 

#     # df_out.loc[:,idx[:,"mean"]] = df_out.loc[:,idx[:,"mean"]].groupby(ID_COLS).fillna(
#     #     method="ffill"
#     # ).groupby(ID_COLS).fillna(icustay_means).fillna(0)

#     df_out.loc[:, idx[:, "count"]] = (df.loc[:, idx[:, "count"]] > 0).astype(float)
#     df_out.rename(columns={"count": "mask"}, level="Aggregation Function", inplace=True)

#     is_absent = (1 - df_out.loc[:, idx[:, "mask"]])
#     hours_of_absence = is_absent.cumsum()
#     time_since_measured = hours_of_absence - hours_of_absence[is_absent==0].fillna(method="ffill")
#     time_since_measured.rename(columns={"mask": "time_since_measured"}, level="Aggregation Function", inplace=True)

#     df_out = pd.concat((df_out, time_since_measured), axis=1)
#     df_out.loc[:, idx[:, "time_since_measured"]] = df_out.loc[:, idx[:, "time_since_measured"]].fillna(100)

#     df_out.sort_index(axis=1, inplace=True)
#     return df_out

def simple_imputer(dataframe, ID_COLS):
    idx = pd.IndexSlice
    df = dataframe.copy()

    # Adjust column levels if necessary
    if len(df.columns.names) > 2:
        df.columns = df.columns.droplevel(("label", "LEVEL1", "LEVEL2"))

    # Select mean and count columns
    df_out = df.loc[:, idx[:, ["mean", "count"]]].copy()  # Explicit deep copy

    # Compute group-level means
    icustay_means = df_out.loc[:, idx[:, "mean"]].groupby(ID_COLS).transform("mean")

    # Forward fill and fill NaNs with icustay_means
    df_out.loc[:, idx[:, "mean"]] = (
        df_out.loc[:, idx[:, "mean"]]
        .groupby(ID_COLS)
        .ffill()  # Forward fill within groups
    )
    df_out.loc[:, idx[:, "mean"]] = df_out.loc[:, idx[:, "mean"]].fillna(icustay_means)

    # Fill remaining NaNs with 0
    df_out.loc[:, idx[:, "mean"]] = df_out.loc[:, idx[:, "mean"]].fillna(0)

    # Binary mask for count columns
    df_out.loc[:, idx[:, "count"]] = (df.loc[:, idx[:, "count"]] > 0).astype(float)
    df_out = df_out.rename(columns={"count": "mask"}, level="Aggregation Function")  # Avoid inplace=True

    # Calculate time since last measurement
    is_absent = (1 - df_out.loc[:, idx[:, "mask"]])
    hours_of_absence = is_absent.cumsum()
    time_since_measured = hours_of_absence - hours_of_absence[is_absent == 0].ffill()
    time_since_measured.rename(columns={"mask": "time_since_measured"}, level="Aggregation Function", inplace=True)

    # Add time_since_measured to the output
    df_out = pd.concat((df_out, time_since_measured), axis=1)
    df_out.loc[:, idx[:, "time_since_measured"]] = df_out.loc[:, idx[:, "time_since_measured"]].fillna(100)

    # Sort columns by index
    df_out.sort_index(axis=1, inplace=True)
    
    return df_out





def data_indices(dataset,
                 train_frac,
                 valid_frac,
                 test_frac,
                 early_stop_frac):
    N = len(dataset)
    N_train = int(train_frac * N)
    N_validation = int(valid_frac * N)
    N_test = int(test_frac * N)
    N_early_stop = int(early_stop_frac * N)

    # Generate sequential indices for training and testing
    # Indices from 0 to N_train-1
    train_indices = list(range(N_train))
    # Indices from N_train to N_train + N_validation-1
    validation_indices = list(range(N_train, N_train + N_validation))
    # Indices for test set
    test_indices = list(range(N_train + N_validation, N_train + N_validation + N_test))
    # Indices for early stopping
    early_stop_indices = list(range(N_train + N_validation + N_test, N_train + N_validation + N_test + N_early_stop))
    return train_indices, validation_indices, test_indices, early_stop_indices


def get_mimic_dataloaders(dataset,
                          train_indices,
                          validation_indices,
                          test_indices,
                          early_stop_indices,
                          batch_size=128):
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    validation_subset = Subset(dataset, validation_indices)
    early_stop_subset = Subset(dataset, early_stop_indices)

    train_loader = DataLoader(train_subset, batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size, shuffle=False)
    validation_loader = DataLoader(validation_subset, batch_size, shuffle=False)
    early_stop_loader = DataLoader(early_stop_subset, batch_size, shuffle=False)

    return train_loader, validation_loader, test_loader, early_stop_loader


def data_normalization(lvl2_train,
                       lvl2_test):
    idx = pd.IndexSlice
    lvl2_means, lvl2_stds = lvl2_train.loc[:, idx[:,"mean"]].mean(axis=0), lvl2_train.loc[:, idx[:,"mean"]].std(axis=0)

    lvl2_train.loc[:, idx[:,"mean"]] = (lvl2_train.loc[:, idx[:,"mean"]] - lvl2_means)/lvl2_stds
    lvl2_test.loc[:, idx[:,"mean"]] = (lvl2_test.loc[:, idx[:,"mean"]] - lvl2_means)/lvl2_stds
    return lvl2_train, lvl2_test


def standard_scaler(flat_train,
                    flat_test):
    # Initialize the scaler
    scaler = StandardScaler()

    # Identify continuous columns (float64 and int64 types)
    continuous_columns = flat_train.select_dtypes(include=["float64", "int64"]).columns

    # Fit the scaler on training data and transform both training and test sets
    train_flat_continuous = scaler.fit_transform(flat_train[continuous_columns])
    test_flat_continuous = scaler.transform(flat_test[continuous_columns])

    # Create copies of the original DataFrames
    train_scaled = flat_train.copy()
    test_scaled = flat_test.copy()

    # Replace continuous columns with the scaled versions
    train_scaled[continuous_columns] = train_flat_continuous
    test_scaled[continuous_columns] = test_flat_continuous

    # Return the scaled DataFrames
    return train_scaled, test_scaled


def flatten_multi_index(df):
    """Flattens the multi-index DataFrame by resetting the index."""
    return df.reset_index(drop=True)


def assert_no_missing_values(*dfs):
    """Asserts that no DataFrame in the input list contains any missing values."""
    for df in dfs:
        assert not df.isnull().any().any(), "DataFrame contains missing values!"
