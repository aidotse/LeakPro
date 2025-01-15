# This code is inspired by the following repository:
# https://github.com/ndb796/CelebA-HQ-Face-Identity-and-Attributes-Recognition-PyTorch/blob/main/Facial_Identity_Classification_Test_with_CelebA_HQ.ipynb

import os
import pickle

from torch import cuda, device, nn, no_grad, optim, save
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.init_params = {"num_classes": num_classes}

    def forward(self, x):
        return self.model(x)

def evaluate(model, loader, criterion, device):
    model.eval()
    loss, acc = 0, 0
    with no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            target = target.view(-1) 
            output = model(data)
            loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            acc += pred.eq(target).sum().item()
        loss /= len(loader)
        acc = float(acc) / len(loader.dataset)
    return loss, acc

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
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    for e in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        train_acc, train_loss = 0.0, 0.0

        for data, target in train_loader:
            data, target = data.to(device_name, non_blocking=True), target.to(device_name, non_blocking=True)
            target = target.view(-1)
            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, target)
            pred = output.argmax(dim=1)  # for multi-class classification
            train_acc += pred.eq(target).sum().item()

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        test_loss, test_acc = evaluate(model, test_loader, criterion, device_name)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

    # Move the model back to the CPU
    model.to("cpu")

    os.makedirs(train_config["run"]["log_dir"], exist_ok=True)
    with open( train_config["run"]["log_dir"]+"/target_model.pkl", "wb") as f:
        save(model.state_dict(), f)

    # Create metadata and store it
    meta_data = {}
    meta_data["train_indices"] = train_loader.dataset.indices
    meta_data["test_indices"] = test_loader.dataset.indices
    meta_data["num_train"] = len(meta_data["train_indices"])

    # Write init params
    meta_data["init_params"] = {}
    for key, value in model.init_params.items():
        meta_data["init_params"][key] = value

    # read out optimizer parameters
    meta_data["optimizer"] = {}
    meta_data["optimizer"]["name"] = optimizer.__class__.__name__.lower()
    meta_data["optimizer"]["lr"] = optimizer.param_groups[0].get("lr", 0)
    meta_data["optimizer"]["weight_decay"] = optimizer.param_groups[0].get("weight_decay", 0)
    meta_data["optimizer"]["momentum"] = optimizer.param_groups[0].get("momentum", 0)
    meta_data["optimizer"]["dampening"] = optimizer.param_groups[0].get("dampening", 0)
    meta_data["optimizer"]["nesterov"] = optimizer.param_groups[0].get("nesterov", False)

    # read out criterion parameters
    meta_data["loss"] = {}
    meta_data["loss"]["name"] = criterion.__class__.__name__.lower()

    meta_data["batch_size"] = train_loader.batch_size
    meta_data["epochs"] = epochs
    meta_data["train_acc"] = train_acc
    meta_data["test_acc"] = test_acc
    meta_data["train_loss"] = train_loss
    meta_data["test_loss"] = test_loss
    meta_data["dataset"] = train_config["data"]["dataset"]

    with open("target/model_metadata.pkl", "wb") as f:
        pickle.dump(meta_data, f)

    return train_accuracies, train_losses, test_accuracies, test_losses
