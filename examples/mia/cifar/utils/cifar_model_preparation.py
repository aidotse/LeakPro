import torch.nn as nn
from torch import optim, no_grad, save
import torchvision.models as models
import pickle
from cifar_handler import CifarInputHandler

from leakpro.schemas import MIAMetaDataSchema
from leakpro.utils.conversion import optimizer_to_config, loss_to_config

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
                                      train_config,
                                      train_indices,
                                      test_indices):
    lr = train_config["train"]["learning_rate"]
    momentum = train_config["train"]["momentum"]
    epochs = train_config["train"]["epochs"]
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    output = CifarInputHandler().train(dataloader=train_loader,
                              model=model,
                              criterion=criterion,
                              optimizer=optimizer,
                              epochs=epochs)
    model = output.model
    model.to("cpu")
    train_acc = output.metrics["accuracy"]
    train_loss = output.metrics["loss"]
    acc_history = output.metrics["accuracy_history"]
    loss_history = output.metrics["loss_history"]
    
    test_loss, test_acc = evaluate(model, test_loader, criterion, "cpu")
    
    with open( train_config["run"]["log_dir"]+"/target_model.pkl", "wb") as f:
        save(model.state_dict(), f)

    # Create metadata and store it
    # Write init params
    init_params = {}
    for key, value in model.init_params.items():
        init_params[key] = value
        
    meta_data = MIAMetaDataSchema(
            train_indices=train_indices,
            test_indices=test_indices,
            num_train=len(train_indices),
            init_params=init_params,
            optimizer=optimizer_to_config(optimizer=optimizer),
            loss=loss_to_config(loss_fn=criterion),
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
    
    return acc_history, loss_history, test_acc, test_loss
