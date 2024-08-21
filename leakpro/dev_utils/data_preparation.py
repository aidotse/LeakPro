"""Module with functions for preparing the dataset for training the target models."""

import logging
import os
import pickle
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader
from torchvision import transforms

from leakpro.dataset import GeneralDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_adult_dataset(dataset_name: str, data_dir: str, logger:logging.Logger) -> GeneralDataset:
    """Get the dataset."""
    path = f"{data_dir}/{dataset_name}"
    if os.path.exists(f"{path}.pkl"):
        with open(f"{path}.pkl", "rb") as file:
            all_data = joblib.load(file)
        logger.info(f"Load data from {path}.pkl")
    else:
        column_names = [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
            "income",
        ]
        df_train = pd.read_csv(f"{path}/{dataset_name}.data", names=column_names)
        df_test = pd.read_csv(
            f"{path}/{dataset_name}.test", names=column_names, header=0
        )
        df_test["income"] = df_test["income"].str.replace(".", "", regex=False)
        df_concatenated = pd.concat([df_train, df_test], axis=0)
        df_replaced = df_concatenated.replace(" ?", np.nan)
        df_clean = df_replaced.dropna()
        x, y = df_clean.iloc[:, :-1], df_clean.iloc[:, -1]

        categorical_features = [col for col in x.columns if x[col].dtype == "object"]
        numerical_features = [
            col for col in x.columns if x[col].dtype in ["int64", "float64"]
        ]

        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        x_categorical = onehot_encoder.fit_transform(x[categorical_features])

        scaler = StandardScaler()
        x_numerical = scaler.fit_transform(x[numerical_features])

        x = np.hstack([x_numerical, x_categorical])

        # label encode the target variable to have the classes 0 and 1
        y = LabelEncoder().fit_transform(y)

        all_data = GeneralDataset(x,y)
        Path(path).mkdir(parents=True, exist_ok=True)
        save_dataset(all_data, path, logger)
    return all_data

def get_cifar10_dataset(dataset_name: str, data_dir: str, logger:logging.Logger) -> GeneralDataset:
    """Get the dataset."""
    path = f"{data_dir}/{dataset_name}"

    if os.path.exists(f"{path}.pkl"):
        with open(f"{path}.pkl", "rb") as file:
            all_data = joblib.load(file)
        logger.info(f"Load data from {path}.pkl")
    else:
        logger.info("Downloading CIFAR-10 dataset")
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root="./data/cifar10", train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root="./data/cifar10", train=False,download=True, transform=transform)
        x = np.vstack([trainset.data, testset.data])
        y = np.hstack([trainset.targets, testset.targets])

        all_data = GeneralDataset(x, y, transform)
        Path(path).mkdir(parents=True, exist_ok=True)
        save_dataset(all_data, path, logger)
    return all_data

def get_ciar10_dataset_fl(model):
    module = CifarModule()
    return module.get_subset(1), module.data_mean, module.data_std, module.init_at_image(1, model)

def download_file(url: str, download_path: str) -> None:
    """Download a file from a given URL."""
    try:
        urlretrieve(url, download_path)  # noqa: S310
    except Exception as e:
        error_msg = f"Failed to download file from {url}: {e}"
        raise RuntimeError(error_msg) from e

def extract_tar(tar_path: str, extract_path: str) -> None:
    """Extract a tar file to a given path."""
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_path)  # noqa: S202

def get_cinic10_dataset(dataset_name: str, data_dir: str, logger:logging.Logger) -> GeneralDataset:
    """Get the dataset."""
    path = f"{data_dir}/{dataset_name}"
    if os.path.exists(f"{path}.pkl"):
        with open(f"{path}.pkl", "rb") as file:
            all_data = joblib.load(file)
        logger.info(f"Load data from {path}.pkl")
    else:
        if not os.path.exists("./data/cinic10"):
            logger.info("Downloading CINIC-10 dataset")
            os.makedirs("./data/cinic10")
            url = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz"
            download_path = "./data/CINIC-10.tar.gz"
            download_file(url, download_path)
            extract_tar(download_path, "./data/cinic10")
            os.remove(download_path)

        transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5),
                                                              (0.5, 0.5, 0.5))])

        trainset =  torchvision.datasets.ImageFolder(root="./data/cinic10/train", transform=transform)
        testset =  torchvision.datasets.ImageFolder(root="./data/cinic10/test", transform=transform)
        validset =  torchvision.datasets.ImageFolder(root="./data/cinic10/valid", transform=transform)

        train_data, train_targets = zip(*[(image.numpy(), target) for image, target in trainset])
        test_data, test_targets = zip(*[(image.numpy(), target) for image, target in testset])
        valid_data, valid_targets = zip(*[(image.numpy(), target) for image, target in validset])

        x = np.vstack([train_data, test_data, valid_data])
        x = np.transpose(x, (0, 2, 3, 1))
        y = np.hstack([train_targets, test_targets, valid_targets])

        all_data = GeneralDataset(x, y, transform)
        Path(path).mkdir(parents=True, exist_ok=True)
        save_dataset(all_data, path, logger)
    return all_data

def save_dataset(all_data: GeneralDataset, path: str, logger:logging.Logger) -> GeneralDataset:
    """Save the dataset."""
    with open(f"{path}.pkl", "wb") as file:
        pickle.dump(all_data, file)
        logger.info(f"Save data to {path}.pkl")

def prepare_train_test_datasets(dataset_size: int, configs: dict) -> dict:
    """Prepare the dataset for training the target models when the training data are sampled uniformly from the population.

    Args:
    ----
        dataset_size (int): Size of the whole dataset
        num_datasets (int): Number of datasets we should generate
        configs (dict): Data split configuration

    Returns:
    -------
        dict: Data split information which saves the information of training points index and test points index.

    """
    # The index_list will save all the information about the train, test and auit for each target model.
    all_index = np.arange(dataset_size)
    train_size = int(configs["f_train"] * dataset_size)
    test_size = int(configs["f_test"] * dataset_size)

    selected_index = np.random.choice(all_index, train_size + test_size, replace=False)
    train_index, test_index = train_test_split(selected_index, test_size=test_size)
    return {"train_indices": train_index, "test_indices": test_index}

class CifarModule:

    def __init__(self, ds = 'CIFAR10', root: str = "./data", batch_size: int = 32, num_workers = 2) -> None:
        # Dynamically construct the class name
        dataset_name = ds
        # Use getattr to access the dataset class from the datasets module
        DatasetClass = getattr(torchvision.datasets, dataset_name)
        trainset = DatasetClass(root=root, train=True, download=True, transform=transforms.ToTensor())
        valset = DatasetClass(root=root, train=False, download=True, transform=transforms.ToTensor())
        
        data_mean, data_std = self._get_meanstd(trainset)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std)])
        
        trainset.transform = transform
        valset.transform = transform

        self.trainloader = DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, drop_last=True, num_workers=num_workers)
        self.valloader = DataLoader(valset, batch_size=batch_size,
                                                shuffle=True, drop_last=False, num_workers=num_workers)
        self.data_mean = torch.as_tensor(data_mean)[:, None, None].to(DEVICE)
        self.data_std = torch.as_tensor(data_std)[:, None, None].to(DEVICE)
        
    def get_train_val_loaders(self) -> tuple[DataLoader, DataLoader]:
        return self.trainloader, self.valloader

    def _get_meanstd(self, trainset):
        cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
        data_mean = torch.mean(cc, dim=1).tolist()
        data_std = torch.std(cc, dim=1).tolist()
        return data_mean, data_std
    
    def get_subset(self, num_examples):
        labels = []
        ground_truths = []
        target_ids = np.random.choice(len(self.valloader.dataset), size=num_examples, replace=False)
        for target_id in target_ids:
            img, label = self.valloader.dataset[target_id]
            labels.append(torch.as_tensor((label,), device=DEVICE))
            ground_truths.append(img.to(DEVICE))
        return [(torch.stack(ground_truths), torch.cat(labels))]
    
    def get_subset_idx(self, indexes):
        labels = []
        ground_truths = []
        target_ids = np.array(indexes)
        for target_id in target_ids:
            img, label = self.valloader.dataset[target_id]
            labels.append(torch.as_tensor((label,), device=DEVICE))
            ground_truths.append(img.to(DEVICE))
        return torch.cat(labels), torch.stack(ground_truths)
    
    def init_at_image(self, num_images, model):
        img_shape = self.trainloader.dataset[0][0].shape
        return torch.randn((num_images, *img_shape), **dict(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype))
    
    def init_at_label(self, num_images):
        """Inits the fake labels

        Args:
            batch_size: the batch size

        Returns:
            randomly initialized or estimated labels
        """
        label_shape = self.trainloader.dataset[0][1].shape
        return torch.randn((num_images, label_shape), requires_grad=True, device=DEVICE)