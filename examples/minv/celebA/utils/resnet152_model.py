import os
import pickle

from torch import cuda, device, nn, no_grad, optim, save
from torchvision.models import ResNet152_Weights, resnet152
from tqdm import tqdm

from leakpro.schemas import MIAMetaDataSchema, OptimizerConfig, LossConfig, EvalOutput, DataLoaderConfig


class ResNet152(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
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
    with open(train_config["run"]["log_dir"]+"/target_model.pkl", "wb") as f:
        save(model.state_dict(), f)

    # Create metadata and store it
    init_params = {}
    for key, value in model.init_params.items():
        init_params[key] = value
    
    optimizer_data = {
        "name": optimizer.__class__.__name__.lower(),
        "params": optimizer.param_groups[0],
    }
    
    train_result = EvalOutput(accuracy=train_accuracies[-1],
                             loss=train_losses[-1],
                             extra={"accuracy_history": train_accuracies, "loss_history": train_losses})
    
    test_result = EvalOutput(accuracy=test_accuracies[-1],
                             loss=test_losses[-1],
                             extra={"accuracy_history": test_accuracies, "loss_history": test_losses})
    
    data_loader_config = DataLoaderConfig(
        params={
            "batch_size": train_loader.batch_size,
        }
    )
    
    loss_data = {"name": criterion.__class__.__name__.lower()}
    
    meta_data = MIAMetaDataSchema(
            train_indices=train_loader.dataset.indices,
            test_indices=test_loader.dataset.indices,
            num_train=len(train_loader.dataset.indices),
            init_params=init_params,
            optimizer=OptimizerConfig(**optimizer_data),
            criterion=LossConfig(**loss_data),
            data_loader=data_loader_config,
            epochs=epochs,
            train_result=train_result,
            test_result=test_result,
            dataset="mimiciii"
        )

    with open("target/model_metadata.pkl", "wb") as f:
        pickle.dump(meta_data, f)

    return train_accuracies, train_losses, test_accuracies, test_losses
