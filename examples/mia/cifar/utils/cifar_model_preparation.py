import torch.nn as nn
from torch import device, optim, cuda, no_grad, save, sigmoid
import torchvision.models as models
import pickle
from tqdm import tqdm

from leakpro.schemas import MIAMetaDataSchema, OptimizerConfig, LossConfig

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False)
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
    
    device_name = device("cuda" if cuda.is_available() else "cpu")
    model.to(device_name)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
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
    with open( train_config["run"]["log_dir"]+"/target_model.pkl", "wb") as f:
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
            dataset=train_config["data"]["dataset"]
        )
    
    with open("target/model_metadata.pkl", "wb") as f:
        pickle.dump(meta_data, f)
    
    return train_accuracies, train_losses, test_accuracies, test_losses
