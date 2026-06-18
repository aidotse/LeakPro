#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""DP-SGD variant of the GDD_ENS target model.

Identical architecture to ``GddMLP`` (utils/gdd_model.py), but exposes a ``dpsgd`` flag so the
DP-SGD model handler knows to wrap training with Opacus. The flag is stored as an instance
attribute (like every other constructor argument) so that LeakPro's ``get_model_init_params``
records it in the target metadata — shadow models are rebuilt from that metadata and therefore
inherit ``dpsgd=True``, i.e. **shadows are trained under the same DP guarantee as the target**
(calibrated LiRA/RMIA, matching the cifar DP example).

Kept as a separate, self-contained file (no cross-imports into ``gdd_model.py``) because
LeakPro reloads the model module by basename when training shadows; intra-``utils`` relative
imports would not resolve. The architecture is duplicated deliberately for that reason.

The MLP uses only ``Linear`` / ``ReLU`` / ``Dropout`` — all Opacus-compatible (no BatchNorm) —
so ``ModuleValidator.fix`` (applied defensively in the handler) is a no-op here.
"""

from torch import Tensor, nn


class GddMLP_DPsgd(nn.Module):  # noqa: N801 - mirrors cifar's ResNet18_DPsgd naming
    """Single hidden-layer MLP over the GDD_ENS genomic features, DP-SGD-ready.

    Args:
        input_size: Number of input features (set from the prepared population).
        hidden_size: Hidden layer width.
        num_classes: Number of tumor-type classes.
        dropout: Dropout probability.
        dpsgd: When True, the DP-SGD handler trains this model privately with Opacus.

    """

    def __init__(self, input_size: int = 4599, hidden_size: int = 128,
                 num_classes: int = 38, dropout: float = 0.1, dpsgd: bool = True) -> None:
        super().__init__()
        # Stored verbatim so get_model_init_params can recover the shadow-model recipe,
        # including dpsgd (so shadows are also trained under DP).
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.dpsgd = dpsgd

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
