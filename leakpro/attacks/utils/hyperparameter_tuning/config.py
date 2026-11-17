#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Configuration for Optuna hyperparameter searches."""

from typing import Any, Callable, Literal

import optuna
from pydantic import BaseModel, ConfigDict, Field


class OptunaConfig(BaseModel):
    """Configuration for the Optuna hyperparameter search."""

    seed: int = Field(default=1234,
                      description="Random seed for reproducibility")
    n_trials: int = Field(default=100,
                          description="Number of trials to find the optimal hyperparameters")
    check_interval: int = Field(default=3000,
                          description="Interval of steps between checks")
    direction: Literal["maximize", "minimize"] = Field("maximize",
                                                       description="Direction of the optimization, minimize or maximize")
    pruner: optuna.pruners.BasePruner = Field(default=optuna.pruners.MedianPruner(n_warmup_steps=5),
                                              description="Number of steps before pruning of experiments will be available")
    objective: Callable[[Any], float] = Field(lambda x: x, description="Objective function to optimize")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")  # Prevent extra fields
