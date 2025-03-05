"""Module containing the class to handle the user input for the CIFAR10 dataset."""

from typing import Self
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
import xgboost as xgb
import numpy as np
from leakpro import AbstractInputHandler


class DTInputHandler(AbstractInputHandler):
    """Class to handle the user input for the CIFAR10 dataset."""

    def __init__(self, configs: dict) -> None:
        super().__init__(configs = configs)

    def get_criterion(self: Self, criterion: _Loss) -> None:
        pass

    def get_optimizer(self: Self, optimizer:Optimizer) -> None:
        pass
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model: xgb.XGBClassifier = None,
    ) -> dict:
        """Train an XGBoost model."""
    
        # Train the XGBoost model
        print("Training XGBoost model...")
        model.fit(X_train, y_train)

        # Compute training accuracy
        train_preds = model.predict(X_train)
        train_acc = np.mean(train_preds == y_train)
        return {"model": model, "metrics": {"accuracy": train_acc}}
