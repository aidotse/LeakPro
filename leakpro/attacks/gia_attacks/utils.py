"""Module that contains utility classes, functions and more for gradient inversion attack on a target."""

from dataclasses import dataclass, field

from torch.nn import CrossEntropyLoss

from leakpro.fl_utils.gia_optimizers import MetaAdam, MetaSGD  # noqa: F401


@dataclass
class InvertingConfig:
    """Possible configs for the Inverting Gradients attack."""

    # total variation scale for smoothing the reconstructions after each iteration
    tv_reg: float = 1.0e-06
    # learning rate on the attack optimizer
    attack_lr: float = 0.1
    # iterations for the attack steps
    at_iterations: int = 8000
    # MetaOptimizer, see MetaSGD for implementation
    optimizer: object = field(default_factory=lambda: MetaSGD())
    # Client loss function
    criterion: object = field(default_factory=lambda: CrossEntropyLoss())
    # Number of epochs for the client attack
    epochs: int = 1
    # if to use median pool 2d on images, can improve attack on high higher resolution (100+)
    median_pooling: bool = False
    # if we compare difference only for top 10 layers with largest changes. Potentially good for larger models.
    top10norms: bool = False

def invertingconfigdictmap(config_dict: dict) -> InvertingConfig:
    """Map a dictionary of parameters to an InvertingConfig object.

    Args:
        config_dict: Dictionary containing parameter names and values

    Returns:
        InvertingConfig: Configuration object with parameters set from dictionary

    """

    # Make sure config is a dictionary
    if not isinstance(config_dict, dict):
        raise ValueError("Config must be a dictionary")

    # Make sure config is not empty
    if not config_dict:
        raise ValueError("Config dictionary cannot be empty")

    # Initialize dummy configuration object
    config = InvertingConfig()

    for key, value in config_dict.items():
        if key == "optimizer":
            # Map optimizer string name to class
            optimizer_class = globals()[value]
            config.optimizer = optimizer_class()
        elif key == "criterion":
            # Map criterion string name to class
            criterion_class = globals()[value]
            config.criterion = criterion_class()
        else:
            if key not in InvertingConfig.__dataclass_fields__:
                raise AttributeError(f"Unknown parameter: {key}, does not exist in InvertingConfig")
            # Set other attributes directly
            setattr(config, key, value)

    return config
