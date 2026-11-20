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
    """Draw targets from a prepared pool and omit their complementary splits from references."""

    n_runs: Optional[StrictInt] = Field(default=None, ge=2)
    seed: Optional[StrictInt] = Field(default=None, ge=0, le=MAX_SEED)
    model_config = ConfigDict(extra="forbid")

    def validate_attack(self, attack: "AbstractMIA") -> None:
        """Check support, pool size and training settings before model preparation."""
        if not callable(getattr(attack, "run_robustness", None)):
            raise ValueError(f"{type(attack).__name__} does not support cached-signal robustness")
        if self.n_runs is not None and self.n_runs > attack.num_shadow_models + 1:
            raise ValueError("n_runs must be between 2 and num_shadow_models + 1")
        setup, pool = attack.handler.target_model_metadata, ShadowModelHandler()
        if pool.handler is not attack.handler:
            raise ValueError("The active shadow pool belongs to a different audit. Recreate this audit before running it.")
        expected = (attack.handler.target_model_blueprint.__name__, attack.handler.configs.target.module_path,
                    setup.init_params, setup.optimizer.name.lower(), setup.optimizer.params or {},
                    setup.criterion.name.lower(), setup.criterion.params or {}, setup.epochs,
                    setup.data_loader.params.get("batch_size", 1))
        actual = (pool.model_class, pool.model_path, pool.init_params, pool.optimizer_name, pool.optimizer_config,
                  pool.criterion_name, pool.loss_config, pool.epochs, pool.batch_size)
        if actual != expected:
            raise ValueError("Robustness requires the same target and shadow training setup; training sizes may differ")


    def reference_models(self, population: np.ndarray, training_indices: list) -> dict[int, np.ndarray]:
        """Draw distinct targets and return the references available to each one."""
        n_runs = self.n_runs if self.n_runs is not None else len(training_indices) - 1
        assignments = [np.isin(population, split) for split in training_indices]
        keys = [np.packbits(members).tobytes() for members in assignments]
        targets = np.random.default_rng(self.seed).permutation(len(keys))[:n_runs]
        references_by_target = {}
        for target in targets.tolist():
            complement = np.packbits(~assignments[target]).tobytes()
            references = [row for row, key in enumerate(keys) if row != target and key != complement]
            if not references:
                raise ValueError("Robustness needs a reference model outside the target's complementary split")
            references_by_target[target] = np.asarray(references, dtype=int)
        return references_by_target


def save_robustness_results(attack: "AbstractMIA", signals: dict[str, np.ndarray], membership: np.ndarray,
                            training_sizes: list[int], references_by_target: dict[int, np.ndarray], rotations: Iterable[tuple],
                            seed: Optional[int], output_dir: Optional[str]) -> list[MIAResult]:
    """Save attack-produced scores and summarize completed rotations.

    Each item supplies the target row, reference rows, scores and kept-record mask.
    Model rows are original target first, followed by the attack's shadow-model order.
    This consumer only records results; the attack selects and evaluates every rotation.
    Native result metadata identifies the saved experiment under ``robustness/<experiment_id>``.
    The original result carries the aggregate when drawn. Otherwise the first run carries it.
    """
    records = attack.audit_dataset["data"]
    model_ids = [-1, *map(int, attack.shadow_model_indices)]
    pool = ShadowModelHandler()
    checkpoints = [pool.target_model_hash, *(pool.checkpoint_hashes[index] for index in model_ids[1:])]
    if len(set(training_sizes)) > 1:
        logger.warning(f"Robustness target/shadow training sizes differ: {training_sizes}. "
                       "Results mix training sizes; compare the original and rotated targets accordingly.")
    signal_hashes = {}
    for name, values in signals.items():
        digest = sha256(f"{values.dtype}:{values.shape}".encode())
        digest.update(np.ascontiguousarray(values).data)
        signal_hashes[name] = digest.hexdigest()
    identity = {"attack": type(attack).__name__, "config": attack.configs.model_dump(), "seed": seed,
                "targets": [model_ids[index] for index in references_by_target],
                "references": [[model_ids[index] for index in rows] for rows in references_by_target.values()],
                "models": model_ids, "checkpoints": checkpoints,
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
    original = next((res for res in results if res.metadata["robustness"]["target"] == -1), results[0])
    original.set_robustness(summary)
    for run_index, result in enumerate(results):
        result.save(attack_obj=attack, output_dir=str(directory / f"run_{run_index}"))
    MIAResult.create_roc_plot(results, str(directory), rotation_summary=summary)
    temporary = directory / ".summary.json.tmp"
    temporary.write_text(json.dumps({"setup": identity,
                                     "runs": [result.metadata["robustness"] for result in results],
                                     "rotated_target_summary": summary}, indent=2, allow_nan=False))
    temporary.replace(summary_path)
    return results


def summarize_robustness(results: list[MIAResult]) -> dict[str, Any]:
    """Average defined AUCs. Report ROC percentiles when every run defines a curve."""
    summary = {}
    aucs = [result.roc_auc for result in results if result.roc_auc is not None]
    if aucs:
        summary["roc_auc"] = {"mean": float(np.mean(aucs)), "n_defined": len(aucs), "n_total": len(results),
                              "percentiles_2_5_97_5": np.percentile(aucs, [2.5, 97.5]).tolist()}
    if results and all(result.roc_mode == "full" for result in results):
        fpr_grid = np.unique(np.concatenate([[0.0], *[result.fpr for result in results]]))
        # Include both endpoints of each vertical segment, including rises at FPR zero.
        jumps = np.unique(np.concatenate([
            r.fpr[(np.diff(r.fpr, prepend=0.0) == 0) & (np.diff(r.tpr, prepend=0.0) > 0)] for r in results
        ]))
        fpr_grid = np.sort(np.r_[fpr_grid, jumps])
        curves = np.empty((len(results), len(fpr_grid)))
        for row, result in enumerate(results):
            # A vertical segment contributes its upper TPR at the knot. The
            # preceding segment ends at the lower TPR of that same knot.
            knots, first, counts = np.unique(np.r_[0.0, result.fpr], return_index=True, return_counts=True)
            tpr = np.r_[0.0, result.tpr]
            upper = tpr[first + counts - 1]
            slopes = np.r_[(tpr[first[1:]] - upper[:-1]) / np.diff(knots), 0.0]
            left = np.searchsorted(knots, fpr_grid, side="right") - 1
            curve = upper[left] + (fpr_grid - knots[left]) * slopes[left]
            vertical = np.isin(knots, jumps)
            curve[np.searchsorted(fpr_grid, knots[vertical])] = tpr[first[vertical]]
            curves[row] = curve
        summary["roc"] = {
            "fpr": fpr_grid.tolist(),
            "mean_tpr": np.mean(curves, axis=0).tolist(),
            "pointwise_tpr_percentiles_2_5_97_5": np.percentile(curves, [2.5, 97.5], axis=0, overwrite_input=True).tolist(),
        }
    return summary
