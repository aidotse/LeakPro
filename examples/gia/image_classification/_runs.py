#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""How this section's examples are run: full-size runs for ``common.run_all`` (run name -> module, flags)."""
IC = "examples.gia.image_classification"

FULL = {
    "compare_attacks": (f"{IC}.compare_attacks", ["--num-images", "4", "--seeds", "1", "--iter-scale", "0.2"]),
    "bn_statistics": (f"{IC}.bn_statistics", ["--num-images", "4", "--seeds", "3", "--iter-scale", "0.2"]),
    "label_inference": (f"{IC}.label_inference", ["--device", "cuda", "--trials", "20"]),
    "multi_seed": (f"{IC}.multi_seed", ["--num-images", "4", "--iter-scale", "0.4"]),
    "fedavg": (f"{IC}.fedavg", ["--num-images", "2", "--iter-scale", "0.1"]),
    "scaling": (f"{IC}.scaling", ["--sweep", "batch"]),
    "scaling_epochs": (f"{IC}.scaling", ["--sweep", "epochs", "--iter-scale", "0.3"]),
    "optimization_choices": (f"{IC}.optimization_choices", ["--num-images", "4", "--seeds", "2", "--iter-scale", "0.3"]),
}
