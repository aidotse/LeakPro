"""
This file contains the implementation of ResNet18, ResNet152, and VGG16 models 
for image classification.
"""

import os
import pickle

from torch import cuda, device, nn, no_grad, optim, save
from torchvision.models import resnet18, resnet50, resnet152
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet152_Weights
from tqdm import tqdm

from leakpro.schemas import MIAMetaDataSchema, EvalOutput
from leakpro.utils.conversion import _loss_to_config, _optimizer_to_config, _dataloader_to_config


class BaseCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # Keep track of initialization parameters for metadata generation
        self.init_params = {"num_classes": num_classes}

    def forward(self, x):
        return self.model(x)


class ResNet18(BaseCNN):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Replace the final fully-connected layer to match the desired number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)


class ResNet50(BaseCNN):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Replace the final fully-connected layer to match the desired number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

class ResNet152(BaseCNN):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
        # Replace the final fully-connected layer to match the desired number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)


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


def create_trained_model_and_metadata(model, train_loader, test_loader, train_config):
    lr = train_config["train"]["learning_rate"]
    epochs = train_config["train"]["epochs"]
    weight_decay = train_config["train"]["weight_decay"]

    device_name = device("cuda" if cuda.is_available() else "cpu")
    model.to(device_name)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
            pred = output.argmax(dim=1)
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
    with open(os.path.join(train_config["run"]["log_dir"], "target_model.pkl"), "wb") as f:
        save(model.state_dict(), f)

    # Create metadata and store it
    init_params = dict(model.init_params)
    
    train_result = EvalOutput(
        accuracy=train_accuracies[-1],
        loss=train_losses[-1],
        extra={"accuracy_history": train_accuracies, "loss_history": train_losses}
    )
    
    test_result = EvalOutput(
        accuracy=test_accuracies[-1],
        loss=test_losses[-1],
        extra={"accuracy_history": test_accuracies, "loss_history": test_losses}
    )
    
    meta_data = MIAMetaDataSchema(
        train_indices=train_loader.dataset.indices,
        test_indices=test_loader.dataset.indices,
        num_train=len(train_loader.dataset.indices),
        init_params=init_params,
        optimizer=_optimizer_to_config(optimizer),
        criterion=_loss_to_config(criterion),
        data_loader=_dataloader_to_config(train_loader),
        epochs=epochs,
        train_result=train_result,
        test_result=test_result,
        dataset="mimiciii"
    )

    with open(os.path.join(train_config["run"]["log_dir"], "model_metadata.pkl"), "wb") as f:
        pickle.dump(meta_data, f)

    return train_accuracies, train_losses, test_accuracies, test_losses
