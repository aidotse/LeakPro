import torch.nn as nn
import torch.optim as optim
from leakpro.import_helper import Dict, Any

# Dictionaries to map user input to PyTorch optimizer and loss function classes
optimizer_mapping = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "RMSprop": optim.RMSprop,
    # Add more optimizers as needed
}

loss_mapping = {
    "MSE": nn.MSELoss,
    "CrossEntropy": nn.CrossEntropyLoss,
    "NLL": nn.NLLLoss,
    # Add more loss functions as needed
}


def get_optimizer(name: str, params:nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """Creates an optimizer based on the given name and configuration.

    Args:
    ----
        name: the name of the optimizer (as a string)
        params: the parameters to optimize
        config: a dictionary of configuration options.

    Returns:
    -------
        optim.Optimizer: The optimizer object.

    Raises:
    ------
        ValueError: If the loss function name is unknown.

    """
    if name not in optimizer_mapping:
        raise ValueError(f"Unknown optimizer: {name}")

    optimizer_class = optimizer_mapping[name]
    return optimizer_class(params=params, **config)


# Function to get loss function from mapping
def get_loss_function(name: str) -> nn.Module:
    """Get the loss function based on the given name.

    Args:
    ----
        name (str): The name of the loss function.

    Returns:
    -------
        nn.Module: The loss function.

    Raises:
    ------
        ValueError: If the loss function name is unknown.

    """
    if name in loss_mapping:
        return loss_mapping[name]()
    raise ValueError(f"Unknown loss function: {name}")