import xgboost as xgb
import pickle
import os
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
import torch
from leakpro.schemas import MIAMetaDataSchema, OptimizerConfig, LossConfig

class xgboost_model(xgb.XGBClassifier):
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        params = {
        "objective": "multi:softmax",  # Change to "multi:softmax" for multi-class
        "eval_metric": "mlogloss",
        "learning_rate": 0.1,
        "max_depth": 6,
        "n_estimators": 1000,
        "subsample": 0.5,
        "colsample_bytree": 1.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "random_state": 42,
        "tree_method": "hist",
        "device": device
        }

        super().__init__(**params)

    def eval(self):
        pass

    def __call__(self, entry):
        """Make the model callable with PyTorch tensors."""
        if isinstance(entry, torch.Tensor):
            entry = entry.detach().cpu().numpy()  # Convert to NumPy array
        
        output = self.predict_proba(entry)  # Get class probabilities
        return torch.tensor(output, dtype=torch.float32)  # Convert back to Tensor
    
    def to(self, device):
        """Override the to method to make it compatible with PyTorch models."""
        return self


def train_xgboost_model(train_data, train_labels, test_data, test_labels, log_dir="logs"):    

    model = xgboost_model()
    model.fit(train_data, train_labels, eval_set=[(train_data, train_labels)],verbose=True)

    # Predictions
    train_preds = model.predict(train_data)
    test_preds = model.predict(test_data)
    train_probs = model.predict_proba(train_data)
    #test_probs = model.predict_proba(test_data)
    
    # Metrics
    train_acc = accuracy_score(train_labels, train_preds)
    test_acc = accuracy_score(test_labels, test_preds)
    train_loss = log_loss(train_labels, train_probs, labels=np.unique(train_labels))
    #test_loss = log_loss(test_labels, test_probs, labels=np.unique(test_labels))
    
    # Save model
    model.device = "cpu"
    os.makedirs(log_dir, exist_ok=True)
    model_save_path = os.path.join(log_dir, "target_model.pkl")
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
    
    optimizer_data = {
        "name": "0",
        "lr": 0,
        "weight_decay": 0,
        "momentum": 0,
        "dampening": 0,
        "nesterov": False
    }
    loss_data = {
        "name": "0"
    }

    # Metadata
    meta_data = MIAMetaDataSchema(
            train_indices=[0],
            test_indices=[0],
            num_train=0,
            init_params={},
            optimizer=OptimizerConfig(**optimizer_data),
            loss=LossConfig(**loss_data),
            batch_size=1,
            epochs=1,
            train_acc=train_acc,
            test_acc=test_acc,
            train_loss=train_loss,
            test_loss=0,
            dataset="mimic-iv"
        )

    with open("target/model_metadata.pkl", "wb") as f:
        pickle.dump(meta_data, f)
    
    return train_acc, test_acc, train_loss #, test_loss