"""Contains utility functions for MIA, MiNVA, and GIA result classes."""

from collections import defaultdict

import numpy as np
from pydantic import BaseModel


def get_result_fixed_fpr(fpr: list, tpr: list) -> dict:
    """Find TPR values for fixed TPRs."""
    # Function to find TPR at given FPR thresholds
    def find_tpr_at_fpr(fpr_array:np.ndarray, tpr_array:np.ndarray, threshold:float) -> float:
        """Find tpr for a given fpr."""
        try:
            # Find the last index where FPR is less than the threshold
            valid_index = np.where(fpr_array < threshold)[0][-1]
            return float(f"{tpr_array[valid_index] * 100:.4f}")
        except IndexError:
            # Return None or some default value if no valid index found
            return "N/A"

    # Compute TPR values at various FPR thresholds
    return {"TPR@1.0%FPR": find_tpr_at_fpr(fpr, tpr, 0.01),
            "TPR@0.1%FPR": find_tpr_at_fpr(fpr, tpr, 0.001),
            "TPR@0.01%FPR": find_tpr_at_fpr(fpr, tpr, 0.0001),
            "TPR@0.0%FPR": find_tpr_at_fpr(fpr, tpr, 0.0)}

def get_config_name(config: BaseModel) -> str:
    """Create id from the attack config."""

    config = dict(sorted(config.items()))

    exclude = ["attack_data_dir"]

    config_name = ""
    for key, value in zip(list(config.keys()), list(config.values())):
        if key in exclude:
            pass
        elif type(value) is bool:
            config_name += f"-{key}"
        else:
            config_name += f"-{key}={value}"
    return config_name

def reduce_to_unique_labels(results: list) -> list:
    """Reduce very long labels to unique and distinct ones."""
    strings = [res.id for res in results]

    # Dictionary to store name as key and a list of configurations as value
    name_configs = defaultdict(list)

    # Parse each string and store configurations
    for s in strings:
        parts = s.split("-")
        name = parts[0]  # The first part is the name
        config = "-".join(parts[1:]) if len(parts) > 1 else ""  # The rest is the configuration
        name_configs[name].append(config)  # Store the configuration under the name

    def find_common_suffix(configs: list) -> str:
        """Helper function to find the common suffix among multiple configurations."""
        if not configs:
            return ""

        # Split each configuration by "-" and zip them in reverse to compare backwards
        reversed_configs = [config.split("-")[::-1] for config in configs]
        common_suffix = []

        for elements in zip(*reversed_configs):
            if all(e == elements[0] for e in elements):
                common_suffix.append(elements[0])
            else:
                break

        # Return the common suffix as a string, reversed back to normal order
        return "-".join(common_suffix[::-1])

    result = []

    # Process each name and its configurations
    for name, configs in name_configs.items():
        if len(configs) > 1:
            # Find the common suffix for the configurations
            common_suffix = find_common_suffix(configs)

            # Remove the common suffix from each configuration
            trimmed_configs = [config[:-(len(common_suffix) + 1)] if common_suffix and config.endswith(common_suffix)
                                                                                    else config for config in configs]

            # Process configurations based on whether they share the same pattern
            for config in trimmed_configs:
                if config:
                    result.append(f"{name}-{config}")
                else:
                    result.append(name)
        else:
            # If only one configuration, just return the string as is
            result.append(f"{name}")

    return result
