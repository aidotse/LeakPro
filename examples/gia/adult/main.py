"""Script demonstrates how to perform gradient inversion attack on adult."""

from torch.utils.data import Subset, DataLoader
from utils.adult_data_preparation import download_adult_dataset, get_adult_dataloaders, preprocess_adult_dataset
from utils.adult_model_preparation import AdultNet, create_trained_model_and_metadata
from adultGIAHandler import adultGiaHandler
import numpy as np
from leakpro import LeakPro

def train_global_model(dataset):
    n_features = dataset.x.shape[1]
    n_classes = 1
    train_loader, test_loader, train_indices, _ = get_adult_dataloaders(dataset, train_fraction=0.3, test_fraction=0.3)

    global_model = AdultNet(input_size=n_features, hidden_size=64, num_classes=n_classes)
    global_model, meta_data = create_trained_model_and_metadata(global_model, train_loader,test_loader, epochs=10)
    return global_model, meta_data

if __name__ == "__main__":

    path = "./data"
    download_adult_dataset(path)
    dataset = preprocess_adult_dataset(path)
    global_model, meta_data = train_global_model(dataset)
    
    # Train the client model
    n_client_data = 8
    client_data_indices = np.random.choice(len(dataset), 8, replace=False)
    client_loader = Subset(dataset, client_data_indices)
    train_loader = DataLoader(client_loader, batch_size=8, shuffle=True)
    
    # Set up Leakpro
    config_path = "audit.yaml"

    # Prepare leakpro object
    leakpro = LeakPro(adultGiaHandler, config_path)