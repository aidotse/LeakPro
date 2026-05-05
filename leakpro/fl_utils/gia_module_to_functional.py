#
# Copyright 2023-2026 AI Sweden
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Utils used for GIA training."""

from collections import OrderedDict
from typing import Optional

import torch
from torch import cuda, nn
from torch.func import functional_call

from leakpro.utils.import_helper import Any, Dict, Self


class MetaModule(nn.Module):
    """Run a network with a provided parameter dict (stateless).

    This avoids monkey-patching module.forward and is compatible with ConvNeXt.
    """

    def __init__(self: Self, net: nn.Module, device: Optional[torch.device] = None) -> None:

        if device is None:
            device = torch.device("cuda" if cuda.is_available() else "cpu")
        super().__init__()
        self.net = net.to(device)

        self.parameters = OrderedDict(self.net.named_parameters())

        self.buffers = OrderedDict(self.net.named_buffers())

    def forward(self: Self, input: torch.Tensor, parameters: Dict[str, Any]) -> torch.Tensor:
        """Forward through net using provided parameters (and existing buffers)."""
        state = {}
        state.update(parameters)
        state.update(self.buffers)

        return functional_call(self.net, state, (input,))
