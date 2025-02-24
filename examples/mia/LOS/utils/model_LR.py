import os
import pickle

import tqdm as tqdm
from torch import cuda, device, nn, no_grad, optim, save, sigmoid
from leakpro.schemas import MIAMetaDataSchema, OptimizerConfig, LossConfig

class LR(nn.Module):
    def __init__(self, input_dim: int):
        """Initialize the logistic regression model with a single linear layer.

        Args:
        ----
            input_dim (int): The size of the input feature vector.

        """
        super(LR, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Binary classification (1 output)
        # Metadata initialization
        self.init_params = {"input_dim": input_dim}

    def forward(self, x):
        """Forward pass through the model."""
        return sigmoid(self.linear(x))  # Sigmoid to produce probabilities for binary classification

def evaluate(model, loader, criterion, device):
    model.eval()
    loss, acc = 0, 0
    with no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            target = target.float().unsqueeze(1)
            output = model(data)
            loss += criterion(output, target).item()
            pred = (output) >= 0.5
            acc += pred.eq(target).float().mean().item()
        loss /= len(loader)
        acc = float(acc) / len(loader)
    return loss, acc


def create_trained_model_and_metadata(model,
                                      train_loader,
                                      test_loader,
                                      epochs ,
                                      lr ,
                                      weight_decay ,
                                      metadata = None):

    device_name = device("cuda" if cuda.is_available() else "cpu")
    model.to(device_name)
    model.train()

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(),
                          lr = lr,
                          weight_decay = weight_decay)
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    for e in tqdm.tqdm(range(epochs), desc="Training Progress"):
        model.train()
        train_acc, train_loss = 0.0, 0.0

        for data, target in train_loader:
            target = target.float().unsqueeze(1)
            data, target = data.to(device_name, non_blocking=True), target.to(device_name, non_blocking=True)
            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, target)
            pred = (output) >= 0.5
            train_acc += pred.eq(target).float().mean().item()

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        test_loss, test_acc = evaluate(model, test_loader, criterion, device_name)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

    # Move the model back to the CPU
    model.to("cpu")


    if not os.path.exists("target_LR"):
        os.makedirs("target_LR")
    with open("target_LR/target_model.pkl", "wb") as f:
        save(model.state_dict(), f)

    # Create metadata and store it
    # Write init params
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
            dataset="mimiciii"
        )


    with open("target_LR/model_metadata.pkl", "wb") as f:
        pickle.dump(meta_data, f)

    return train_accuracies, train_losses, test_accuracies, test_losses
