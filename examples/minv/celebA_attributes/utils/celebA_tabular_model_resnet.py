import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import cuda, device
import os
from torch.serialization import save
import pickle
from leakpro.schemas import MIAMetaDataSchema, OptimizerConfig, LossConfig




class ResNetTabular(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1, num_blocks=3, dropout=0.1):
        super(ResNetTabular, self).__init__()
        self.init_params = {"num_classes": output_dim, "input_dim": input_dim}

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        self.blocks = nn.Sequential(*[
            ResNetBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Sigmoid()  # Use softmax for multi-class

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = self.blocks(x)
        return self.activation(self.output_layer(x))

class ResNetBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(ResNetBlock, self).__init__()

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x  # Store the input for residual connection
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.bn2(self.fc2(x))
        return torch.relu(x + residual)  # Residual connection

def create_trained_model_and_metadata(model,
                                      train_loader,
                                      test_loader,
                                      train_config):
    lr = train_config["train"]["learning_rate"]
    momentum = train_config["train"]["momentum"]
    epochs = train_config["train"]["epochs"]
    weight_decay = train_config["train"]["weight_decay"]

    device_name = device("cuda" if cuda.is_available() else "cpu")
    model.to(device_name)
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.1, 0.9))
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    for e in tqdm(range(epochs), desc="Training Progress"):
        train_acc, train_loss = 0.0, 0.0
        model.train()
        
        for data, target in train_loader:
            data, target = data.to(device_name), target.to(device_name)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            _, preds = torch.max(output, 1)
            train_acc += torch.sum(preds == target.data)

        train_loss /= len(train_loader.dataset)
        train_acc = train_acc.double() / len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc.item())

        model.eval()
        test_acc, test_loss = 0.0, 0.0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device_name, non_blocking=True), target.to(device_name, non_blocking=True)

                output = model(data)
                loss = criterion(output, target)

                test_loss += loss.item() * data.size(0)
                _, preds = torch.max(output, 1)
                test_acc += torch.sum(preds == target.data)

        test_loss /= len(test_loader.dataset)
        test_acc = test_acc.double() / len(test_loader.dataset)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc.item())
        print(f"Epoch {e+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
    # Move the model back to the CPU
    model.to("cpu")

    os.makedirs(train_config["run"]["log_dir"], exist_ok=True)
    with open( train_config["run"]["log_dir"]+"/target_model.pkl", "wb") as f:
        save(model.state_dict(), f)

    # Create metadata and store it
    init_params = {}
    for key, value in model.init_params.items():
        init_params[key] = value
    
    optimizer_data = {
        "name": optimizer.__class__.__name__.lower(),
        "lr": optimizer.param_groups[0].get("lr", 0),
        "weight_decay": optimizer.param_groups[0].get("weight_decay", 0),
        "momentum": optimizer.param_groups[0].get("momentum", 0),
        "dampening": optimizer.param_groups[0].get("dampening", 0),
        "nesterov": optimizer.param_groups[0].get("nesterov", False)
    }
    
    loss_data = {"name": criterion.__class__.__name__.lower()}
    
    meta_data = MIAMetaDataSchema(
            train_indices=train_loader.dataset.indices,
            test_indices=test_loader.dataset.indices,
            num_train=len(train_loader.dataset.indices),
            init_params=init_params,
            optimizer=OptimizerConfig(**optimizer_data),
            loss=LossConfig(**loss_data),
            batch_size=train_loader.batch_size,
            epochs=epochs,
            train_acc=train_acc,
            test_acc=test_acc,
            train_loss=train_loss,
            test_loss=test_loss,
            dataset="celeba"
        )

    with open("target/model_metadata.pkl", "wb") as f:
        pickle.dump(meta_data, f)


    return train_losses, train_accuracies, test_losses, test_accuracies