"""Module containing the class to handle the user input for the CIFAR10 dataset."""

import xgboost as xgb
from leakpro import AbstractInputHandler


class DTInputHandler(AbstractInputHandler):
    """Class to handle the user input for the CIFAR10 dataset."""

    def __init__(self, configs: dict) -> None:
        super().__init__(configs = configs)

    def train(
        self,
        data,
        model: xgb
    ) -> dict:
        """Model training procedure."""



        for e in tqdm(range(epochs), desc="Training Progress"):
            model.train()
            train_acc, train_loss = 0.0, 0.0

            for data, target in dataloader:
                target = target.float().unsqueeze(1)
                data, target = data.to(dev, non_blocking=True), target.to(dev, non_blocking=True)
                optimizer.zero_grad()
                output = model(data)

                loss = criterion(output, target)
                pred = sigmoid(output) >= 0.5
                train_acc += pred.eq(target).sum().item()

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

        train_acc = train_acc/len(dataloader.dataset)
        train_loss = train_loss/len(dataloader)

        return {"model": model, "metrics": {"accuracy": train_acc, "loss": train_loss}}




import os
import numpy as np
import pickle
import xgboost as xgb
from tqdm import tqdm
from leakpro import AbstractInputHandler


class MimicInputHandler(AbstractInputHandler):
    """Class to handle the user input for the MIMIC-III dataset using XGBoost."""

    def __init__(self, configs: dict) -> None:
        super().__init__(configs=configs)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model: xgb.XGBClassifier = None,
    ) -> dict:
        """Train the XGBoost model."""

        # Train the XGBoost model
        print("Training XGBoost model...")
        model.fit(X_train, y_train)

        # Compute training accuracy
        train_preds = model.predict(X_train)
        train_acc = np.mean(train_preds == y_train)

        # Save the trained model
        os.makedirs("target", exist_ok=True)
        with open("target/xgboost_model.pkl", "wb") as f:
            pickle.dump(model, f)

        return {"model": model, "metrics": {"accuracy": train_acc}}
