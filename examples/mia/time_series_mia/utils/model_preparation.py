import os
import torch
import pickle
import numpy as np

from tqdm import tqdm
from torch import nn, optim, cuda, no_grad, save
from leakpro.schemas import MIAMetaDataSchema, OptimizerConfig, LossConfig, DataLoaderConfig, EvalOutput

def predict(model, loader, device, scaler=None, original_scale=False):
    model.eval()
    model.to(device)
    all_targets = []
    all_preds = []
    with no_grad():
        for data, target in loader:                
            data = data.to(device)
            pred = model(data).detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            if original_scale:
                pred_2D = pred.reshape(-1, pred.shape[-1])
                target_2D = target.reshape(-1, target.shape[-1])

                pred_descaled = scaler.inverse_transform(pred_2D)
                target_descaled = scaler.inverse_transform(target_2D)

                pred = pred_descaled.reshape(pred.shape)
                target = target_descaled.reshape(target.shape)
            all_preds.append(pred)
            all_targets.append(target)
    return np.concatenate(all_targets), np.concatenate(all_preds)

def evaluate(model, loader, criterion, device, original_scale=False):
    model.eval()
    model.to(device)
    loss = 0
    with no_grad():
        for data, target in loader:                
            data, target = data.to(device), target.to(device)
            pred = model(data)
            if original_scale:
                pred_2D = pred.detach().cpu().numpy().reshape(-1, pred.shape[-1])
                target_2D = target.detach().cpu().numpy().reshape(-1, target.shape[-1])

                pred_descaled = loader.dataset.dataset.scaler.inverse_transform(pred_2D)
                target_descaled = loader.dataset.dataset.scaler.inverse_transform(target_2D)

                pred = torch.tensor(pred_descaled, device=device).reshape(pred.shape)
                target = torch.tensor(target_descaled, device=device).reshape(target.shape)

            loss += criterion(pred, target).item()
        loss /= len(loader)
    return loss

def create_trained_model_and_metadata(model, train_loader, test_loader, epochs, optimizer_name, loss_fn, dataset_name, val_loader, early_stopping, patience):
    device = torch.device("cuda" if cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    if loss_fn.lower() == "mse":
        criterion = nn.MSELoss()
    elif loss_fn.lower() == "mae":
        criterion = nn.L1Loss()
    else:
        raise NotImplementedError(f"Loss function not found: {loss_fn}")

    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters())
    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(model.parameters())
    else:
        raise NotImplementedError(f"Optimizer not found: {optimizer_name}")
    

    best_val_loss = (-1, np.inf)  # (epoch, validation loss)
    best_state_dict = model.state_dict()

    train_losses, test_losses = [], []
    for i in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        train_loss = 0.0
        
        for data, target in train_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(data)
            
            loss = criterion(pred, target)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        test_loss = evaluate(model, test_loader, criterion, device)
        test_losses.append(test_loss)

        if not early_stopping:
                continue

        # Handle early stopping
        val_loss = evaluate(model, val_loader, criterion, device)
        if val_loss < best_val_loss[1]:
            best_val_loss = (i, val_loss)
            best_state_dict = model.state_dict()
        elif i - best_val_loss[0] > patience:
            print(f"Training stopped early at epoch {i+1}.")
            break
    
    # Restore best weights if using early stopping
    if early_stopping:
        model.load_state_dict(best_state_dict)
        print("Best weights restored.")

    # Move the model back to the CPU
    model.to("cpu")
    if not os.path.exists("target"):
        os.makedirs("target")
    with open("target/target_model.pkl", "wb") as f:
        save(model.state_dict(), f)
    
    # Write init params
    init_params = {}
    for key, value in model.init_params.items():
        init_params[key] = value

    optimizer_config = OptimizerConfig(
        name=optimizer.__class__.__name__.lower(),
        params={
            "lr": optimizer.param_groups[0].get("lr", 0),
        }
    )

    loss_config = LossConfig(name=criterion.__class__.__name__.lower())

    data_loader_config = DataLoaderConfig(params={"batch_size":train_loader.batch_size, "shuffle": True})

    train_result = EvalOutput(loss=train_loss, accuracy=0.0)
    test_result = EvalOutput(loss=test_loss, accuracy=0.0)
    
    meta_data = MIAMetaDataSchema(
        train_indices=train_loader.dataset.indices,
        test_indices=test_loader.dataset.indices,
        num_train=len(train_loader.dataset.indices),
        init_params=init_params,
        optimizer=optimizer_config,
        criterion=loss_config,
        data_loader=data_loader_config,
        epochs=epochs,
        train_result=train_result,
        test_result=test_result,
        dataset=dataset_name,
    )
    
    with open("target/model_metadata.pkl", "wb") as f:
        pickle.dump(meta_data, f)
    
    return train_losses, test_losses
