"""Contains utility functions for MIA, MiNVA, and GIA result classes."""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel


def create_roc_plot(result_objects:list, save_dir: str = "", save_name: str = "") -> None:
    """Plot method for MIAResult. This method can be used by individual result objects or multiple.

    Args:
    ----
        result_objects (list): List of MIAResult objects to plot.
        save_dir (str): Directory to save the plot.
        save_name (str): Name of the plot.

    """

    filename = f"{save_dir}/{save_name}"

    # Create plot for results
    assert isinstance(result_objects, list), "Results must be a list of MIAResult objects"

    reduced_labels = reduce_to_unique_labels(result_objects)
    for res, label in zip(result_objects, reduced_labels):

        plt.fill_between(res.fpr, res.tpr, alpha=0.15)
        plt.plot(res.fpr, res.tpr, label=label)

    # Plot baseline (random guess)
    range01 = np.linspace(0, 1)
    plt.plot(range01, range01, "--", label="Random guess")

    # Set plot parameters
    plt.yscale("log")
    plt.xscale("log")
    plt.xlim(left=1e-5)
    plt.ylim(bottom=1e-5)
    plt.tight_layout()
    plt.grid()
    plt.legend(bbox_to_anchor =(0.5,-0.27), loc="lower center")

    plt.xlabel("False positive rate (FPR)")
    plt.ylabel("True positive rate (TPR)")
    plt.title(save_name+" ROC Curve")
    plt.savefig(fname=f"{filename}.png", dpi=1000, bbox_inches="tight")
    plt.clf()

def get_config_name(config: BaseModel) -> str:
    """Create id from the attack config.

    Args:
    ----
        config (BaseModel): The attack configuration.

    Returns:
    -------
        str: The id of the attack configuration.

    """

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
    """Reduce very long labels to unique and distinct ones.

    Args:
    ----
        results (list): List of result objects.

    Returns:
    -------
        list: List of unique labels.

    """

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
