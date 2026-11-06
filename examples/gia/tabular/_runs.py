#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""How this section's examples are run: full-size runs for ``common.run_all`` (run name -> module, flags)."""
TAB = "examples.gia.tabular"
SEEDS = ["--seeds", "0", "1", "2"]
SWEEP = ["--device", "cpu", *SEEDS, "--max-iter", "2000"]

FULL = {
    "tabular_attack": (f"{TAB}.tabular_attack", ["--device", "cpu"]),
    "california_housing": (f"{TAB}.california_housing", ["--device", "cpu"]),
    "architecture": (f"{TAB}.leakage_factors.architecture", ["--device", "cpu", "--seeds", "0", "1", "--max-iter", "2000"]),
    "attack_iterations": (f"{TAB}.leakage_factors.attack_iterations", ["--device", "cpu", *SEEDS, "--iterations", "1000", "3000"]),
    "batch_sizes": (f"{TAB}.leakage_factors.batch_sizes", [*SWEEP, "--batch-sizes", "1", "4", "16", "64"]),
    "leakage_fedavg": (f"{TAB}.leakage_factors.fedavg", SWEEP),
    "label_knowledge": (f"{TAB}.leakage_factors.label_knowledge", SWEEP),
    "training_exposure": (f"{TAB}.leakage_factors.training_exposure", SWEEP),
}
