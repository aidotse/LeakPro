
import tqdm as tqdm
import pickle
import torch.nn as nn
from torch import device, optim, no_grad, save, sigmoid, cuda


class MimicLR(nn.Module):
    def __init__(self, input_dim: int):
        """Initialize the logistic regression model with a single linear layer.

        Args:
        ----
            input_dim (int): The size of the input feature vector.
        """
        super(MimicLR, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Binary classification (1 output)
        # Metadata initialization
        self.init_params = {"input_dim": input_dim}

    def forward(self, x):
        """Forward pass through the model."""
        return sigmoid(self.linear(x))  # Sigmoid to produce probabilities for binary classification

def evaluate(model, loader, criterion, device):
    model.eval()
    loss, correct = 0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        target = target.float().unsqueeze(1)
        
        with no_grad():
            output = model(data)
            loss += criterion(output, target).item()
            pred = output >= 0.5
            correct += (pred == target).float().sum()
    acc = float(correct)/ len(loader.dataset)
    loss /= len(loader)
    
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

    criterion = nn.BCELoss(reduction="mean")
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
            pred =output >= 0.5
            train_acc += pred.eq(target).sum().item()
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = train_acc / len(train_loader.dataset)
            
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        
        test_loss, test_acc = evaluate(model, test_loader, criterion, device_name)
        # _ , train_loss = evaluate(model, train_loader, criterion, device_name)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
    
    # Move the model back to the CPU
    model.to("cpu")
    with open("target/target_model.pkl", "wb") as f:
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
    meta_data["dataset"] = "mimiciii"
    
    with open("target/model_metadata.pkl", "wb") as f:
        pickle.dump(meta_data, f)
    
    return train_accuracies, train_losses, test_accuracies, test_losses