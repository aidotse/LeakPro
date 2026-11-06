#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""How this section's examples are run: full-size runs for ``common.run_all`` and the cases of the --quick test.

``FULL`` maps a run name (= output directory) to (module, flags of the full run). ``REQUIRES`` names the optional
packages a --quick case needs.
"""
FULL = {
    "quickstart": ("examples.gia.getting_started.quickstart", []),
    "hpo_and_split_runs": ("examples.gia.getting_started.hpo_and_split_runs", ["--trials", "20"]),
}
REQUIRES = {"getting_started/hpo_and_split_runs": ("optuna",)}
