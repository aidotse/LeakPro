#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""GDD_ENS target model — a single MLP tumor-type classifier.

Referenced from audit.yaml (``target.module_path`` + ``target.model_class``). Each __init__
argument is stored as an instance attribute because LeakPro's ``get_model_init_params``
introspects the constructor signature and reads matching attributes to record the recipe that
shadow models are rebuilt from. Drop an attribute here and shadows rebuild with wrong shapes.
"""

from torch import Tensor, nn


class GddMLP(nn.Module):
    """Single hidden-layer MLP over the GDD_ENS genomic features (38 tumor-type classes).

    ``input_size`` / ``num_classes`` defaults are nominal; the notebook constructs the model
    with the actual feature count and class count taken from the prepared population (the
    feature count depends on how many constant columns were dropped).
    """

    def __init__(self, input_size: int = 4599, hidden_size: int = 128,
                 num_classes: int = 38, dropout: float = 0.1) -> None:
        super().__init__()
        # Stored verbatim so get_model_init_params can recover the shadow-model recipe.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
