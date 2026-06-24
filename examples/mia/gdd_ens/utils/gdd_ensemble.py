#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""GDD-ENS ensemble target model (faithful 10-MLP reimplementation).

This is the model the *ensemble* (real-model-style) audit trains and attacks, as opposed to the
single ``GddMLP`` proxy in ``gdd_model.py``. It reproduces the architecture from the GDD_ENS repo
(``scripts/train_gdd_nn.py``) and the per-member hyperparameters from the paper's Supplementary
Table S6 (Darmofal et al., Cancer Discovery 2024).

Two faithfulness details worth knowing:

* ``MLP`` has **no activation after the first Linear** (``n_features -> num_fc_units``). This is the
  GDD_ENS architecture verbatim; member 8 (``num_fc_layers=0``) is therefore a purely linear model.
* ``EnsembleClassifier`` in the original code returns a *list* of member logits and averages
  **softmaxes** downstream. We fold that into ``GddEnsemble.forward``, which returns
  ``log(mean_k softmax(member_logits_k))``. Returning log-mean-softmax means LeakPro's internal
  softmax in the LiRA/RMIA signal recovers the true ensemble probability (``softmax(log p) == p``),
  i.e. softmax-averaging rather than the wrong logit-averaging.

``GddEnsemble.__init__`` takes only ``(n_features, n_types)`` and stores both as attributes, because
LeakPro's ``get_model_init_params`` introspects the constructor signature and reads matching
attributes to record the recipe shadow models are rebuilt from. The 10 member architectures are
fixed by the module-level ``TABLE_S6`` constant (not constructor args), so every shadow ensemble is
rebuilt with the same architectures as the target.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

# Per-member hyperparameters from GDD-ENS Supplementary Table S6 (all 10 MLPs). Architecture fields
# (num_fc_layers, num_fc_units, dropout_rate) define each member; learning_rate / weight_decay are
# read by the model handler to build per-member optimizers during training.
TABLE_S6 = (
    {"num_fc_layers": 1, "num_fc_units": 1376, "dropout_rate": 0.5,      "learning_rate": 1.46e-4, "weight_decay": 7.87e-4},
    {"num_fc_layers": 2, "num_fc_units": 1051, "dropout_rate": 0.5,      "learning_rate": 2.22e-4, "weight_decay": 4.8e-5},
    {"num_fc_layers": 2, "num_fc_units": 1265, "dropout_rate": 0.5,      "learning_rate": 1.45e-4, "weight_decay": 5.1e-5},
    {"num_fc_layers": 1, "num_fc_units": 1842, "dropout_rate": 1.3185e-2, "learning_rate": 1.57e-4, "weight_decay": 2.9e-4},
    {"num_fc_layers": 3, "num_fc_units": 2048, "dropout_rate": 2.7e-5,   "learning_rate": 2.56e-4, "weight_decay": 1.0e-5},
    {"num_fc_layers": 3, "num_fc_units": 1569, "dropout_rate": 1.0714e-2, "learning_rate": 2.69e-4, "weight_decay": 3.76e-4},
    {"num_fc_layers": 3, "num_fc_units": 1711, "dropout_rate": 0.5,      "learning_rate": 8.4e-5,  "weight_decay": 1.0e-5},
    {"num_fc_layers": 0, "num_fc_units": 1171, "dropout_rate": 0.5,      "learning_rate": 1.0e-5,  "weight_decay": 1.38e-4},
    {"num_fc_layers": 3, "num_fc_units": 1817, "dropout_rate": 0.5,      "learning_rate": 1.23e-4, "weight_decay": 9.3e-5},
    {"num_fc_layers": 1, "num_fc_units": 2035, "dropout_rate": 0.5,      "learning_rate": 1.82e-4, "weight_decay": 1.3e-5},
)


class MLP(nn.Module):
    """Single GDD-ENS member MLP (verbatim architecture from scripts/train_gdd_nn.py).

    Note the first ``Linear`` has no following activation, and ``num_fc_layers`` counts the middle
    ReLU/Dropout blocks (so ``num_fc_layers=0`` is a purely linear ``n_features -> n_types`` model).
    """

    def __init__(self, num_fc_layers: int, num_fc_units: int, dropout_rate: float,
                 n_features: int, n_types: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_features, num_fc_units))
        for _ in range(num_fc_layers):
            self.layers.append(nn.Linear(num_fc_units, num_fc_units))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.Dropout(p=dropout_rate))
        self.layers.append(nn.Linear(num_fc_units, n_types))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class GddEnsemble(nn.Module):
    """10-MLP GDD-ENS ensemble. ``forward`` returns log of the mean member softmax.

    Args:
        n_features: Number of input features (set from the prepared population).
        n_types: Number of tumor-type classes.

    The member architectures are fixed by ``TABLE_S6``; only ``n_features`` / ``n_types`` are
    constructor args (and stored as attributes) so LeakPro reconstructs shadow ensembles identically.
    """

    def __init__(self, n_features: int, n_types: int) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_types = n_types
        self.members = nn.ModuleList([
            MLP(c["num_fc_layers"], c["num_fc_units"], c["dropout_rate"], n_features, n_types)
            for c in TABLE_S6
        ])

    def member_logits(self, x: Tensor) -> list[Tensor]:
        """Raw per-member logits (used by the handler to train each member with cross-entropy)."""
        return [member(x) for member in self.members]

    def forward(self, x: Tensor) -> Tensor:
        # Mean of member softmaxes (the paper's ensemble rule), returned in log space so a
        # downstream softmax recovers the ensemble probability exactly.
        probs = torch.stack([F.softmax(member(x), dim=1) for member in self.members], dim=0).mean(dim=0)
        return torch.log(probs.clamp_min(1e-12))
