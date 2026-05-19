"""
LR_arch.py — Logistic Regression architecture for LOS/MIMIC-III.

Upload this as arch.py in Step 2 of the LeakPro webapp.
num_features is detected automatically from the dataset and passed by the webapp.

Outputs num_classes logits (2 for binary LOS prediction).
Uses CrossEntropyLoss — compatible with RMIA and other MIA attacks.
"""
import torch.nn as nn


class LR(nn.Module):
    def __init__(self, num_features: int = 7488, num_classes: int = 2):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.linear = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.linear(x)  # raw logits
