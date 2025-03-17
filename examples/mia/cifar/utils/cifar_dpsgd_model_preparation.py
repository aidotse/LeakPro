import os

from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier

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
        loss /= len(loader.dataset)
        acc = float(acc) / len(loader.dataset)
    return loss, acc

def create_trained_model_and_metadata(model,
                                      train_dataloader,
                                      test_loader,
                                      train_config,
                                      privacy_engine_dict,
                                      target_model_dir):
    lr = train_config["train"]["learning_rate"]
    momentum = train_config["train"]["momentum"]
    epochs = train_config["train"]["epochs"]
    
    device_name = device("cuda" if cuda.is_available() else "cpu")
    model.to(device_name)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    sample_rate = 1 / len(train_dataloader)
    try:
        noise_multiplier = get_noise_multiplier(target_epsilon = privacy_engine_dict["target_epsilon"],
                                        target_delta = privacy_engine_dict["target_delta"],
                                        sample_rate = sample_rate ,
                                        epochs = privacy_engine_dict["epochs"],
                                        epsilon_tolerance = privacy_engine_dict["epsilon_tolerance"],
                                        accountant = "prv",
                                        eps_error = privacy_engine_dict["eps_error"],)
    except Exception as e:
        raise ValueError(
            f"Failed to compute noise multiplier using the 'prv' accountant. "
            f"This may be due to a large target_epsilon ({privacy_engine_dict['target_epsilon']}). "
            f"Consider reducing epsilon or switching to a different accountant (e.g., 'rdp'). "
            f"Original error: {e}")

    # make the model private
    privacy_engine = PrivacyEngine(accountant = "prv")
    model, optimizer, train_dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_dataloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm= privacy_engine_dict["max_grad_norm"],
    )

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    
    for e in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        train_acc, train_loss = 0.0, 0.0
        
        for data, target in train_dataloader:
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
        
        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader.dataset)
            
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        test_loss, test_acc = evaluate(model, test_loader, criterion, device_name)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

    # Move the model back to the CPU
    model.to("cpu")

    os.makedirs(target_model_dir, exist_ok=True)

    state_dict = model.state_dict()
    cleaned_state_dict = {key.replace("_module.", "").replace("module.", ""): value
                      for key, value in state_dict.items()}

    with open(f"{target_model_dir}/target_dpsgd_model.pkl", "wb") as f:
        save(cleaned_state_dict, f)
    
    # Create metadata for privacy engine
    with open(f"{target_model_dir}/dpsgd_dic.pkl", "wb") as f:
        pickle.dump(privacy_engine_dict, f)    

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
            train_indices=train_dataloader.dataset.indices,
            test_indices=test_loader.dataset.indices,
            num_train=len(train_dataloader.dataset.indices),
            init_params=init_params,
            optimizer=OptimizerConfig(**optimizer_data),
            loss=LossConfig(**loss_data),
            batch_size=train_dataloader.batch_size,
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
