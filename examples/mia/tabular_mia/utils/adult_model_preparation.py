#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
import pickle

from torch import nn, no_grad, optim, save
from tqdm import tqdm
from leakpro.schemas import EvalOutput, MIAMetaDataSchema
from leakpro.utils.conversion import dataloader_to_config, loss_to_config, optimizer_to_config
from leakpro.utils.device import get_device, mark_step


class AdultNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(AdultNet, self).__init__()
        self.init_params = {"input_size": input_size,
                            "hidden_size": hidden_size,
                            "num_classes": num_classes}
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

def evaluate(model, loader, criterion, device):
    model.eval()
    loss, acc = 0, 0
    with no_grad():
        for data, target in loader:
            data, target = data.to(device), target.long().to(device)
            output = model(data)
            mark_step(device)
            loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            acc += pred.eq(target).sum()
        loss /= len(loader)
        acc = float(acc) / len(loader.dataset)
    return loss, acc

def create_trained_model_and_metadata(model, train_loader, test_loader, epochs = 10, metadata = None):

    device_name = get_device()
    model.to(device_name)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.8)
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    for e in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        train_acc, train_loss = 0.0, 0.0

        for data, target in train_loader:
            target = target.long()
            data, target = data.to(device_name, non_blocking=True), target.to(device_name, non_blocking=True)
            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, target)
            pred = output.argmax(dim=1)
            train_acc += pred.eq(target).sum().item()

            loss.backward()
            optimizer.step()
            mark_step(device_name)
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
    with open("target/target_model.pkl", "wb") as f:
        save(model.state_dict(), f)

    # Write init params
    init_params = {}
    for key, value in model.init_params.items():
        init_params[key] = value
    
    meta_data = MIAMetaDataSchema(
            train_indices=train_loader.dataset.indices,
            test_indices=test_loader.dataset.indices,
            num_train=len(train_loader.dataset.indices),
            init_params=init_params,
            optimizer=optimizer_to_config(optimizer),
            criterion=loss_to_config(criterion),
            data_loader=dataloader_to_config(train_loader),
            epochs=epochs,
            train_result=EvalOutput(accuracy=train_acc, loss=train_loss),
            test_result=EvalOutput(accuracy=test_acc, loss=test_loss),
            dataset="adult"
        )
    
    with open("target/model_metadata.pkl", "wb") as f:
        pickle.dump(meta_data, f)
    
    return train_accuracies, train_losses, test_accuracies, test_losses
