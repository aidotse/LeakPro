import xgboost as xgb
import numpy as np
import pickle
from tqdm import tqdm
from torch import device, cuda, no_grad

class XGBoostModel:
    def __init__(self, num_classes):
        """
        Initialize an XGBoost classifier.
        """
        self.num_classes = num_classes
        self.model = xgb.XGBClassifier(
            objective="multi:softmax" if num_classes > 2 else "binary:logistic",
            eval_metric="mlogloss" if num_classes > 2 else "logloss",
            use_label_encoder=False,
            num_class=num_classes if num_classes > 2 else 1
        )
        self.init_params = {"num_classes": num_classes}

    def fit(self, X_train, y_train):
        """Train the model."""
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)


def evaluate(model, loader, device):
    model.model.eval()
    loss, acc = 0, 0
    all_preds, all_targets = [], []
    
    with no_grad():
        for data, target in loader:
            data, target = data.numpy(), target.numpy()  # Convert tensors to NumPy arrays
            preds = model.predict(data)
            
            all_preds.extend(preds)
            all_targets.extend(target)
    
    acc = np.mean(np.array(all_preds) == np.array(all_targets))  # Compute accuracy
    return loss, acc  # No loss function equivalent in XGBoost like CrossEntropyLoss


def create_trained_model_and_metadata(model, train_loader, test_loader, train_config):
    lr = train_config["train"]["learning_rate"]
    epochs = train_config["train"]["epochs"]
    
    device_name = device("cuda" if cuda.is_available() else "cpu")

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    
    X_train, y_train = [], []
    for data, target in train_loader:
        X_train.append(data.numpy())
        y_train.append(target.numpy())
    
    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)

    for e in tqdm(range(epochs), desc="Training Progress"):
        model.fit(X_train, y_train)

        # Evaluate training and test performance
        train_loss, train_acc = evaluate(model, train_loader, device_name)
        test_loss, test_acc = evaluate(model, test_loader, device_name)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

    # Save the trained model
    with open(train_config["run"]["log_dir"] + "/target_model.pkl", "wb") as f:
        pickle.dump(model.model, f)

    # Create metadata
    meta_data = {
        "train_indices": train_loader.dataset.indices,
        "test_indices": test_loader.dataset.indices,
        "num_train": len(train_loader.dataset.indices),
        "init_params": model.init_params,
        "batch_size": train_loader.batch_size,
        "epochs": epochs,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "dataset": train_config["data"]["dataset"],
    }
    
    with open("target/model_metadata.pkl", "wb") as f:
        pickle.dump(meta_data, f)

    return train_accuracies, train_losses, test_accuracies, test_losses
