"""Utils used for GIA training."""
from collections import OrderedDict

import torch
from torch import cuda, device, nn

from leakpro.utils.import_helper import Any, Dict, Self

from torch.func import functional_call

class MetaModule(nn.Module):
    """Run a network with a provided parameter dict (stateless).

    This avoids monkey-patching module.forward and is compatible with ConvNeXt.
    """

    def __init__(self: Self, net: nn.Module) -> None:
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        super().__init__()
        self.net = net.to(gpu_or_cpu)

        self.parameters = OrderedDict(self.net.named_parameters())

        self.buffers = OrderedDict(self.net.named_buffers())

    def forward(self: Self, input: torch.Tensor, parameters: Dict[str, Any]) -> torch.Tensor:
        """Forward through net using provided parameters (and existing buffers)."""
        state = {}
        state.update(parameters)
        state.update(self.buffers)

        return functional_call(self.net, state, (input,))
