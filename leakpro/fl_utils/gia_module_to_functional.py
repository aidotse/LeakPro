"""Utils used for GIA training."""
from collections import OrderedDict
from functools import partial

import torch
import torch.nn.functional as f
from torch import cuda, device, nn

from leakpro.utils.import_helper import Any, Dict, Self

# Predefined mapping for known functional calls and their parameters
# If your model contains more modules than the ones specified here, add them
# by mapping the module name to its functional representation in
# <name_to_functional_mapping>. Then find and add their attributes
# e.g. through inspection of the module's class definition, or by
# referring to the PyTorch documentation. This allows for the conversion
# of module-based representations to their corresponding functional API calls.
# Below is a mapping of common layers and their respective parameters.

def has_inner_modules(net_module: nn.Module) -> bool:
    """
    Checks if the given PyTorch module contains inner submodules.

    Args:
        net_module (nn.Module): The PyTorch module to check.

    Returns:
        bool: True if the module contains inner submodules, otherwise False.
    """
    return any(isinstance(submodule, nn.Module) for submodule in net_module.children())


functional_params = {
    "conv2d": ["weight", "bias", "stride", "padding", "dilation", "groups"],
    "batch_norm": ["weight", "bias", "eps", "momentum", "training", "running_mean", "running_var"],
    "linear": ["weight", "bias"],
    "layer_norm": ["weight", "bias", "normalized_shape", "eps"],
    "embedding": ["weight"],
}

name_to_functional_mapping = {
    "conv2d": "conv2d",
    "batchnorm1d": "batch_norm",
    "batchnorm2d": "batch_norm",
    "batchnorm3d": "batch_norm",
    "linear": "linear",
    "layernorm": "layer_norm",
    "embedding": "embedding",
}


class MetaModule(nn.Module):
    """Trace a network and then replace its module calls with functional calls.

    This allows for backpropagation with respect to weights for "normal" PyTorch networks.
    """

    def __init__(self: Self, net: nn.Module) -> None:
        """Init with network."""
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        super().__init__()
        self.net = net
        self.net.to(gpu_or_cpu)
        self.parameters = OrderedDict(self.net.named_parameters())
        self.original_forwards = {}  # Store original forward methods here

    def forward(self: Self, input: torch.Tensor, parameters: Dict[str, Any]) -> torch.Tensor:
        """Forward through module using functional call."""

        # iterator for the new parameters
        param_gen = iter(parameters.values())

        # Iterate over all modules in the network
        for name, net_module in self.net.named_modules():
            # Get the name of the module in lower case
            class_name = net_module.__class__.__name__.lower()
            functional_name = name_to_functional_mapping.get(class_name)

            # Check if the module has a functional equivalent
            if functional_name in functional_params:
                # Save the original forward method
                self.original_forwards[name] = net_module.forward

                # Get all parameters that are present in the functional call
                params = functional_params[functional_name]
                # Create a dictionary with the parameters for the functional call
                kwargs = {param: getattr(net_module, param) for param in params if hasattr(net_module, param)}

                # Replace the parameters with the ones from the iterator
                if "weight" in kwargs and kwargs["weight"] is not None:
                    kwargs["weight"] = next(param_gen)
                if "bias" in kwargs and kwargs["bias"] is not None:
                    kwargs["bias"] = next(param_gen)

                # Replace the forward method with a partial functional call using new parameters
                net_module.forward = partial(getattr(f, functional_name), **kwargs)

        # Forward the input through the network
        output = self.net(input)

        # Restore the original forward methods
        for name, net_module in self.net.named_modules():
            if name in self.original_forwards:
                net_module.forward = self.original_forwards[name]

        return output
