import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import urllib.request
from torch import Tensor, float32, long
from torch.utils.data import Dataset, Subset, DataLoader


class AdultDataset(Dataset):
    def __init__(self, x, y):
        self.x = Tensor(x, dtype=float32)  # Convert features to torch tensors
        self.y = Tensor(y, dtype=long)     # Convert labels to torch tensors

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
def download_adult_dataset(data_dir):
    """Download the Adult Dataset if it's not present."""
    # URLs for the dataset
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
    data_file = os.path.join(data_dir, "adult.data")
    test_file = os.path.join(data_dir, "adult.test")

    # Download the dataset if not present
    if not os.path.exists(data_file):
        print("Downloading adult.data...")
        urllib.request.urlretrieve(base_url + "adult.data", data_file)

    if not os.path.exists(test_file):
        print("Downloading adult.test...")
        urllib.request.urlretrieve(base_url + "adult.test", test_file)

def get_adult_dataset():
    """Get the dataset, download it if necessary, and store it."""
    
    # Download the dataset if not present
    path = "data"
    download_adult_dataset(path)
    
    dataset_path = os.path.join(path, "adult.pkl")
    
    if os.path.exists(dataset_path):
        with open(dataset_path, "rb") as file:
            x, y = joblib.load(file)  # Load x and y directly
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

        # OneHotEncoding categorical features
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        x_categorical = onehot_encoder.fit_transform(x[categorical_features])

        # Scaling numerical features
        scaler = StandardScaler()
        x_numerical = scaler.fit_transform(x[numerical_features])

        # Concatenate numerical and categorical features
        x = np.hstack([x_numerical, x_categorical])

        # Label encode the target variable
        y = LabelEncoder().fit_transform(y)

        # Save the processed features (x) and labels (y) using joblib
        with open(dataset_path, "wb") as file:
            joblib.dump((x, y), file)  # Save x and y as a tuple
    
    return AdultDataset(x, y)
    
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

    return train_loader, test_loader, {"train_indices": train_indices, "test_indices": test_indices}
