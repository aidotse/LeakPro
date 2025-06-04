"""Utility functions for loading and storing."""

import hashlib
import json

from torch.nn import Module


def hash_config(config: dict) -> str:
    """Generate a SHA-256 hash of a dictionary."""

    # Convert the config to a canonical JSON string (sorted keys ensures consistency)
    config_str = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(config_str.encode("utf-8")).hexdigest()

def hash_model(model: Module) -> str:
    """Generate a SHA-256 hash of a PyTorch model's state dictionary.

    This function takes into account the model weights by iterating
    over the state dictionary (which includes both parameters and buffers)
    and updating the hash with both the key names and the corresponding tensor values.
    """
    hasher = hashlib.sha256()
    state_dict = model.state_dict()

    # Sort keys to ensure consistent ordering across models
    for key in sorted(state_dict.keys()):
        hasher.update(key.encode("utf-8"))
        # Convert tensor to CPU, detach, convert to numpy and then to bytes
        tensor_bytes = state_dict[key].detach().cpu().numpy().tobytes()
        hasher.update(tensor_bytes)

    return hasher.hexdigest()

def hash_attack(config:dict, target_model:Module) -> str:
    """Generate a SHA-256 hash of the attack configuration and target model.

    This function combines the hash of the attack configuration and the hash of the model
    to create a unique identifier for the attack.
    """
    config_hash = hash_config(config)
    model_hash = hash_model(target_model)
    combined_hash = hashlib.sha256((config_hash + model_hash).encode("utf-8")).hexdigest()
    return combined_hash  # noqa: RET504
