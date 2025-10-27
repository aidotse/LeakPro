import os
import pickle
import urllib.request

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch import float32, tensor
from torch.utils.data import DataLoader, Dataset, Subset


class AdultDataset(Dataset):
    def __init__(self, x:tensor, y:tensor, dec_to_onehot:dict, one_hot_encoded:bool=True):
        self.x = x
        self.y = y
        self.data = x  # align with LeakPro expectations
        self.targets = y

        # create dictionary to map between indices in categorical representation and one-hot encoded representation
        # For example: cols 1,2 continuous and col 3 categorical with 3 categories will be mapped to {1:1,2:2,3:[3,4,5]}
        self.dec_to_onehot = dec_to_onehot
        self.one_hot_encoded = one_hot_encoded

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def subset(self, indices):
        return AdultDataset(self.x[indices], self.y[indices], self.dec_to_onehot, self.one_hot_encoded)


def download_adult_dataset(data_dir):
    """Download the Adult Dataset if it's not present."""
    # URLs for the dataset
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
    data_file = os.path.join(data_dir, "adult.data")
    test_file = os.path.join(data_dir, "adult.test")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("Created directory:", data_dir)
    else:
        print("Directory already exists:", data_dir)

    # Download the dataset if not present
    if not os.path.exists(data_file):
        print("Downloading adult.data...")
        urllib.request.urlretrieve(base_url + "adult.data", data_file)

    if not os.path.exists(test_file):
        print("Downloading adult.test...")
        urllib.request.urlretrieve(base_url + "adult.test", test_file)

def preprocess_adult_dataset(path):
    """Get the dataset, download it if necessary, and store it."""

    if os.path.exists(os.path.join(path, "adult_data.pkl")):
        with open(os.path.join(path, "adult_data.pkl"), "rb") as f:
            dataset = joblib.load(f)
    else:
        column_names = [
            "age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
        ]

        # Load and clean data
        df_train = pd.read_csv(os.path.join(path, "adult.data"), names=column_names)
        df_test = pd.read_csv(os.path.join(path, "adult.test"), names=column_names, header=0)
        df_test["income"] = df_test["income"].str.replace(".", "", regex=False)

        df_concatenated = pd.concat([df_train, df_test], axis=0)
        df_clean = df_concatenated.replace(" ?", np.nan).dropna()

        # Split features and labels
        x, y = df_clean.iloc[:, :-1], df_clean.iloc[:, -1]

        # Categorical and numerical columns
        categorical_features = [col for col in x.columns if x[col].dtype == "object"]
        numerical_features = [col for col in x.columns if x[col].dtype in ["int64", "float64"]]

        # Scaling numerical features
        scaler = StandardScaler()
        x_numerical = pd.DataFrame(scaler.fit_transform(x[numerical_features]), columns=numerical_features, index=x.index)

        # Label encode the categories
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        x_categorical_one_hot = one_hot_encoder.fit_transform(x[categorical_features])
        one_hot_feature_names = one_hot_encoder.get_feature_names_out(categorical_features)
        x_categorical_one_hot_df = pd.DataFrame(x_categorical_one_hot, columns=one_hot_feature_names, index=x.index)

        # Concatenate the numerical and one-hot encoded categorical features
        x_final = pd.concat([x_numerical, x_categorical_one_hot_df], axis=1)

        # Label encode the target variable
        y = pd.Series(LabelEncoder().fit_transform(y))

        # Add numerical features to the dictionary
        dec_to_onehot_mapping = {}
        for i, feature in enumerate(numerical_features):
            dec_to_onehot_mapping[i] = [x_final.columns.get_loc(feature)]  # Mapping to column index

        # Add one-hot encoded features to the dictionary
        for i, categorical_feature in enumerate(categorical_features):
            j = i + len(numerical_features)
            one_hot_columns = [col for col in one_hot_feature_names if col.startswith(categorical_feature)]
            dec_to_onehot_mapping[j] = [x_final.columns.get_loc(col) for col in one_hot_columns]

        #--------------------
        # Create tensor dataset to be stored
        x_tensor = tensor(x_final.values, dtype=float32)
        y_tensor = tensor(y.values, dtype=float32)
        dataset = AdultDataset(x_tensor, y_tensor, dec_to_onehot_mapping, one_hot_encoded=True)
        with open(f"{path}/adult_data.pkl", "wb") as file:
            pickle.dump(dataset, file)
            print(f"Save data to {path}.pkl")

    return dataset

def get_adult_dataloaders(dataset, train_fraction=0.3, test_fraction=0.3):

    dataset_size = len(dataset)
    train_size = int(train_fraction * dataset_size)
    test_size = int(test_fraction * dataset_size)

    # Use sklearn's train_test_split to split into train and test indices
    selected_index = np.random.choice(np.arange(dataset_size), train_size + test_size, replace=False)
    train_indices, test_indices = train_test_split(selected_index, test_size=test_size)

    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=128, shuffle=False)

    return train_loader, test_loader
