import os
import xgboost as xgb
import numpy as np
import pickle
from utils.data_processing import split_dataset

class XGBoostModel:
    def __init__(self, num_classes, n_estimators=100):
        """
        Initialize an XGBoost classifier.
        """
        self.num_classes = num_classes
        self.model = xgb.XGBClassifier(
            objective="multi:softmax" if num_classes > 2 else "binary:logistic",
            eval_metric="mlogloss" if num_classes > 2 else "logloss",
            use_label_encoder=False,
            num_class=num_classes if num_classes > 2 else 1,
            n_estimators=n_estimators  # Set epochs as n_estimators
        )
        self.init_params = {"num_classes": num_classes, "n_estimators": n_estimators}

    def fit(self, X_train, y_train):
        """Train the model."""
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)


def evaluate(model, X, y):
    """
    Evaluate accuracy of the model.
    """
    preds = model.predict(X)
    acc = np.mean(preds == y)  # Compute accuracy
    return acc  # Loss is not needed for XGBoost


def create_trained_model_and_metadata(model,
                                    dataset,
                                    train_indices,
                                    validation_indices,
                                    test_indices,
                                    train_config):
    """
    Train the XGBoost model and save metadata.
    """

    X_train,y_train, X_val, y_val, X_test, y_test = split_dataset(dataset,
                                                                   train_indices,
                                                                   validation_indices,
                                                                   test_indices)

    # Train the model
    print("Training XGBoost model...")
    model.fit(X_train, y_train)

    # Evaluate
    train_acc = evaluate(model, X_train, y_train)
    test_acc = evaluate(model, X_test, y_test)

    print(f"Training Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

    # Save the trained model
    os.makedirs(train_config["run"]["log_dir"], exist_ok=True)
    model_path = os.path.join(train_config["run"]["log_dir"], "target_model.pkl")

    
    with open(model_path, "wb") as f:
        pickle.dump(model.model, f)

    # Create metadata
    meta_data = {
        "train_size": len(y_train),
        "test_size": len(y_test),
        "train_indices": train_indices,
        "test_indices": test_indices,
        "num_train": len(y_train),
        "init_params": model.init_params,
        "batch_size": 0,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_loss": 0,
        "test_loss": 0,
        "dataset": train_config["data"]["dataset"],
    }

    os.makedirs("target", exist_ok=True)  # Ensure the directory exists
    with open("target/model_metadata.pkl", "wb") as f:
        pickle.dump(meta_data, f)

    return train_acc, test_acc

