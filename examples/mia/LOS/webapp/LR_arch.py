"""
LR_arch.py — Logistic Regression architecture for LOS/MIMIC-III.

Upload this as arch.py in Step 2 of the LeakPro webapp.
num_features and num_classes are detected automatically from the dataset and
passed by the webapp (binary tasks get num_classes=1: a single-logit head
trained with BCEWithLogitsLoss, which LeakPro's attacks handle via sigmoid).
"""
import torch.nn as nn


class LR(nn.Module):
    def __init__(self, num_features: int = 7488, num_classes: int = 1):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.linear = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.linear(x)  # raw logits
