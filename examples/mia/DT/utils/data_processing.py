import os
import pickle
# import urllib.request

# import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
# from torch import float32, tensor, Tensor, from_numpy
from torch.utils.data import DataLoader, Dataset, Subset


class FinanceDataset(Dataset):
    def __init__(self, x, y):
        self.x = x.astype(np.float32)  # Ensure it is of type float
        self.y = y.astype(np.float32)  # Ensure it is of type float

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx].squeeze(0)

    def subset(self, indices):
        return FinanceDataset(self.x[indices], self.y[indices])


def get_finance_dataset(data_path,
                      train_frac,
                      validation_frac,
                      test_frac):
    """Get the dataset, download it if necessary, and store it."""

    # Assert that the sum of all fractions is between 0 and 1
    total_frac = train_frac + validation_frac + test_frac 
    assert 0 < total_frac <= 1, "The sum of dataset fractions must be between 0 and 1."

    dataset_path = os.path.join(data_path, "dataset.pkl")
    indices_path = os.path.join(data_path, "indices.pkl")

    if os.path.exists(dataset_path) and os.path.exists(indices_path):
        print("Loading dataset...")
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)  # Load the dataset
        with open(indices_path, "rb") as f:
            indices_dict = pickle.load(f)  # Load the dictionary containing indices
            train_indices = indices_dict["train_indices"]  # Get the actual train indices
            validation_indices = indices_dict["validation_indices"]  # Get the actual validation indices
            test_indices = indices_dict["test_indices"]    # Get the actual test indices
        print(f"Loaded dataset from {dataset_path}")
        return dataset, train_indices, validation_indices ,test_indices

    data_file_path = os.path.join(data_path, "nodes_train.csv")
    if os.path.exists(data_file_path):
        print("Loading data...")
        df = pd.read_csv(data_file_path)
        
        # Drop the two first columns 
        df_no_id = df.iloc[:, 2:]

        # Separate features (X) and labels (y)
        X = df_no_id.iloc[:, :-1]  # All columns except the last one
        y = df_no_id.iloc[:, -1]   # Last column as the label


        # print("Splitting data...")
        # train_data, holdout_data, y_train, y_holdout_data = data_splitter(y,
        #                                                          X,
        #                                                          train_frac)
        
        # check_missing_values(train_data, holdout_data, y_train, y_holdout_data)
        #TODO: Normalize/standardize data if not using XGBoost
        # print("Normalizing data...")


        # # Creating the dataset
        # data_x = pd.concat((train_data, holdout_data), axis=0)
        # data_y = pd.concat((y_train, y_holdout_data), axis=0)

        assert np.issubdtype(X.values.dtype, np.number), "Non-numeric data found in features."
        assert np.issubdtype(y.values.dtype, np.number), "Non-numeric data found in labels."

        dataset = FinanceDataset(X.values, y.values)

        # Generate indices for training, validation, test, and early stopping
        train_indices, validation_indices, test_indices = data_indices(dataset,
                                                                       train_frac,
                                                                       validation_frac,
                                                                       test_frac)

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
        }
        with open(indices_path, "wb") as file:
            pickle.dump(indices_to_save, file)
            print(f"Saved train and test indices to {indices_path}")
    else:
        msg = "The data file does not exist."
        raise FileNotFoundError(msg)
    return dataset, train_indices, validation_indices, test_indices


def split_dataset(dataset, train_indices, validation_indices, test_indices):
    """Splits dataset into training, validation, and test sets based on provided indices."""
    
    X_train, Y_train = dataset.x[train_indices], dataset.y[train_indices]
    X_val, Y_val = dataset.x[validation_indices], dataset.y[validation_indices]
    X_test, Y_test = dataset.x[test_indices], dataset.y[test_indices]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def data_indices(dataset,
                 train_frac,
                 valid_frac,
                 test_frac
                 ):
    N = len(dataset)
    N_train = int(train_frac * N)
    N_validation = int(valid_frac * N)
    N_test = int(test_frac * N)


    # Generate sequential indices for training and testing
    # Indices from 0 to N_train-1
    train_indices = list(range(N_train))
    # Indices from N_train to N_train + N_validation-1
    validation_indices = list(range(N_train, N_train + N_validation))
    # Indices for test set
    test_indices = list(range(N_train + N_validation, N_train + N_validation + N_test))
    return train_indices, validation_indices, test_indices


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


def check_missing_values(train_data, holdout_data, y_train, y_holdout_data):
    datasets = {
        "train_data": train_data,
        "holdout_data": holdout_data,
        "y_train": y_train,
        "y_holdout_data": y_holdout_data
    }
    
    for name, df in datasets.items():
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Warning: {name} contains {missing_count} missing values!")
        else:
            print(f"{name} has no missing values.")
    return