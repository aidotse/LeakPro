"""
LR_arch.py — Logistic Regression architecture for LOS/MIMIC-III.

Upload this as arch.py in Step 2 of the LeakPro webapp.
num_features is detected automatically from the dataset and passed by the webapp.

Note: No sigmoid in forward — raw logits are returned.
      The handler uses BCEWithLogitsLoss, which is numerically stable
      and compatible with DP-SGD.
"""
import torch.nn as nn


class LR(nn.Module):
    def __init__(self, num_features: int = 7488, num_classes: int = 2):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.linear(x)  # raw logits — no sigmoid
