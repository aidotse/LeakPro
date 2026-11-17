#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Configuration and output handling for attack-owned cached-signal rotation.

ROC aggregation adapts Rickard Brännvall's robustness PR #351:
https://github.com/aidotse/LeakPro/pull/351
Rotations share a model pool; their percentiles are not calibrated confidence intervals.
"""

import json
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.utils.logger import logger
from leakpro.utils.save_load import hash_config
from leakpro.utils.seed import MAX_SEED

if TYPE_CHECKING:
    from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA


class RobustnessConfig(BaseModel):
    """Original target followed by distinct shadow targets from the attack's own pool."""

    n_runs: StrictInt = Field(ge=2)
    seed: Optional[StrictInt] = Field(default=None, ge=0, le=MAX_SEED)
    model_config = ConfigDict(extra="forbid")

    def validate_attack(self, attack: "AbstractMIA") -> None:
        """Check support, pool size and training settings before model preparation."""
        if not callable(getattr(attack, "run_robustness", None)):
            raise ValueError(f"{type(attack).__name__} does not support cached-signal robustness")
        if self.n_runs > attack.num_shadow_models + 1:
            raise ValueError("n_runs must be between 2 and num_shadow_models + 1")
        setup, pool = attack.handler.target_model_metadata, ShadowModelHandler()
        expected = (attack.handler.target_model_blueprint.__name__, attack.handler.configs.target.module_path,
                    setup.init_params, setup.optimizer.name.lower(), setup.optimizer.params or {},
                    setup.criterion.name.lower(), setup.criterion.params or {}, setup.epochs,
                    setup.data_loader.params.get("batch_size"))
        actual = (pool.model_class, pool.model_path, pool.init_params, pool.optimizer_name, pool.optimizer_config,
                  pool.criterion_name, pool.loss_config, pool.epochs, pool.batch_size)
        if actual != expected:
            raise ValueError("Robustness requires the same target and shadow training setup; training sizes may differ")


def save_robustness_results(attack: "AbstractMIA", signals: dict[str, np.ndarray], membership: np.ndarray,
                            training_sizes: list[int], targets: list[int], rotations: Iterable[tuple],
                            seed: Optional[int], output_dir: Optional[str]) -> list[MIAResult]:
    """Save attack-produced scores and summarize completed rotations.

    Each item supplies the target row, reference rows, scores and kept-record mask.
    Model rows are original target first, followed by the attack's shadow-model order.
    This consumer only records results; the attack selects and evaluates every rotation.
    Native result metadata identifies the saved experiment under ``robustness/<experiment_id>``.
    The original-target result carries the aggregate of the rotated targets.
    """
    records = attack.audit_dataset["data"]
    model_ids = [-1, *map(int, attack.shadow_model_indices)]
    if len(set(training_sizes)) > 1:
        logger.warning(f"Robustness target/shadow training sizes differ: {training_sizes}. "
                       "Results mix training sizes; compare the original and rotated targets accordingly.")
    signal_hashes = {}
    for name, values in signals.items():
        digest = sha256(f"{values.dtype}:{values.shape}".encode())
        digest.update(np.ascontiguousarray(values).data)
        signal_hashes[name] = digest.hexdigest()
    identity = {"attack": type(attack).__name__, "config": attack.configs.model_dump(), "seed": seed,
                "targets": [model_ids[index] for index in targets], "models": model_ids,
                "records": records.tolist(), "membership": membership.tolist(),
                "training_sizes": training_sizes, "signals": signal_hashes}
    root = Path(attack.handler.configs.audit.output_dir if output_dir is None else output_dir)
    directory = root / "robustness" / hash_config(identity)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / "summary.json"
    summary_path.unlink(missing_ok=True)
    results = []
    for target, references, scores, kept in rotations:
        labels = membership[target, kept]
        if not labels.any() or labels.all():
            raise ValueError(f"Robustness target {model_ids[target]} needs scorable members and non-members")
        provenance = {"experiment_id": directory.name, "target": model_ids[target],
                      "references": [model_ids[index] for index in references],
                      "target_training_size": training_sizes[target],
                      "reference_training_sizes": [training_sizes[index] for index in references],
                      "audit_record_ids": records[kept].tolist()}
        result = MIAResult.from_full_scores(labels, scores.reshape(-1, 1),
                                           result_name=type(attack).__name__.removeprefix("Attack"),
                                           metadata={**attack.configs.model_dump(), "robustness": provenance},
                                           audit_indices=records[kept])
        results.append(result)
    summary = summarize_robustness(results)
    results[0].set_robustness(summary)
    for run_index, result in enumerate(results):
        result.save(attack_obj=attack, output_dir=str(directory / f"run_{run_index}"))
    temporary = directory / ".summary.json.tmp"
    temporary.write_text(json.dumps({"setup": identity,
                                     "runs": [result.metadata["robustness"] for result in results],
                                     "rotated_target_summary": summary}, indent=2, allow_nan=False))
    temporary.replace(summary_path)
    return results


def summarize_robustness(results: list[MIAResult]) -> dict[str, Any]:
    """Adapt PR #351's pointwise percentile aggregation to rotated-target results."""
    rotated = results[1:]
    summary = {}
    aucs = [result.roc_auc for result in rotated if result.roc_auc is not None]
    if aucs:
        summary["roc_auc"] = {"mean": float(np.mean(aucs)),
                              "percentiles_2_5_97_5": np.percentile(aucs, [2.5, 97.5]).tolist()}
    if all(result.roc_mode == "full" for result in rotated):
        fpr_grid = np.unique(np.concatenate([[0.0], *[result.fpr for result in rotated]]))
        curves = []
        for result in rotated:
            # MIAResult omits the origin. Include it before taking upper values
            # at repeated FPRs so tied scores do not inflate low-FPR summaries.
            fpr, inverse = np.unique(np.concatenate(([0.0], result.fpr)), return_inverse=True)
            upper_tpr = np.zeros(len(fpr))
            np.maximum.at(upper_tpr, inverse, np.concatenate(([0.0], result.tpr)))
            curves.append(np.interp(fpr_grid, fpr, upper_tpr))
        summary["roc"] = {"fpr": fpr_grid.tolist(), "mean_tpr": np.mean(curves, axis=0).tolist(),
                          "pointwise_tpr_percentiles_2_5_97_5": np.percentile(curves, [2.5, 97.5], axis=0).tolist()}
    return summary
