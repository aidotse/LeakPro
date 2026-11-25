#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""How this section's examples are run: full-size runs for ``common.run_all`` and the cases of the --quick test.

``FULL`` maps a run name (= output directory) to (module, flags of the full run); ``NEEDS_DATA`` holds the runs
that need a manual dataset download (module, flags, data path). ``REQUIRES`` names the optional packages a
--quick case needs.
"""
from pathlib import Path

GP = "examples.gia.generative_priors"
REPO = Path(__file__).resolve().parents[3]

FULL = {
    "gan_priors": (f"{GP}.gan_priors", ["--num-images", "4"]),
    "diffusion_priors": (f"{GP}.diffusion_priors", ["--dataset", "cifar10", "--num-images", "4"]),
}
NEEDS_DATA = {
    "diffusion_priors_celeba": (f"{GP}.diffusion_priors", ["--dataset", "celeba", "--num-images", "1"],
                                REPO / "data/celeba/celeba/img_align_celeba"),
}
REQUIRES = {"generative_priors/diffusion_priors": ("diffusers",), "generative_priors/gan_priors": ("cma",)}
