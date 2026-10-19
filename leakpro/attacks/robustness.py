#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Model-level leave-one-out evaluation for configured membership inference attacks.

Target selection and ROC aggregation adapt Rickard Brännvall's robustness PR #351:
https://github.com/aidotse/LeakPro/pull/351
The original sampling.py also supports bootstrap reference sampling; this runner
implements target rotation without replacement and keeps attack calculations intact.
"""

import json
from copy import copy, deepcopy
from pathlib import Path
from types import MethodType
from typing import Any, Optional

import numpy as np

from leakpro.attacks.mia_attacks.attack_factory_mia import AttackFactoryMIA
from leakpro.attacks.utils.distillation_model_handler import DistillationModelHandler
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.signals.signal_extractor import PytorchModel
from leakpro.utils.save_load import hash_config, hash_model
from leakpro.utils.seed import MAX_SEED, seed_everything


class Robustness:
    """Run an MIA on its original target, then on distinct shadows from a fixed pool.

    All models must use the target's training setup and training-set size. The pool
    is trained once on subsets of the original audit records. Reference-based
    attacks must accept that pool under their normal data and shadow-count rules.
    The original target joins the references when a shadow becomes the target.

    Args:
        handler: An initialized LeakPro MIA handler.
        attack_name: Name registered in AttackFactoryMIA.
        attack_config: Parameters of the attack, without the "attack" routing key.
        n_runs: Total runs including the original target, from 2 to num_shadow_models + 1.
        num_shadow_models: Number of models in each run's reference pool.
        seed: Optional seed for pool training, target order, and attack preparation.

    Notes:
        The summary contains empirical percentiles across rotated targets, excluding
        the original-target result. These dependent runs do not establish calibrated
        confidence intervals. Execution is sequential because model handlers are shared.

    """

    def __init__(self, handler:MIAHandler, attack_name:str, attack_config:dict,
                 n_runs:int, num_shadow_models:int, seed:Optional[int]=None) -> None:
        if attack_name not in AttackFactoryMIA.attack_classes:
            raise ValueError(f"Unknown attack type: {attack_name}")
        if type(num_shadow_models) is not int or num_shadow_models < 1:
            raise ValueError("num_shadow_models must be a positive integer")
        if type(n_runs) is not int or not 2 <= n_runs <= num_shadow_models + 1:
            raise ValueError("n_runs must be between 2 and num_shadow_models + 1")
        if seed is not None and (type(seed) is not int or not 0 <= seed <= MAX_SEED):
            raise ValueError(f"seed must be between 0 and {MAX_SEED}, or None")
        config = AttackFactoryMIA.attack_classes[attack_name].AttackConfig(**attack_config)
        if hasattr(config, "num_shadow_models") and config.num_shadow_models != num_shadow_models:
            raise ValueError("The attack's num_shadow_models must equal the robustness pool size")
        self.handler = handler
        self.attack_name = attack_name
        self.attack_config = config.model_dump()
        self.n_runs = n_runs
        self.num_shadow_models = num_shadow_models
        self.seed = seed
        self.results = []
        self.summary = {}
        self._records = np.concatenate((handler.train_indices, handler.test_indices))
        if len(np.unique(self._records)) != len(self._records):
            raise ValueError("Audit training and test records must be unique and disjoint")
        if len(handler.train_indices) == 0 or len(handler.test_indices) == 0:
            raise ValueError("The original audit needs both training and test records")
        self._positions = {int(record): row for row, record in enumerate(self._records)}

    def run(self) -> list[dict]:
        """Prepare the model pool, run the unchanged attack, and save an empirical summary."""
        handlers = (ShadowModelHandler, DistillationModelHandler)
        snapshots = [(factory, copy(vars(factory())) if factory.is_created() else None) for factory in handlers]
        factory_state = (AttackFactoryMIA.shadow_model_handler, AttackFactoryMIA.distillation_model_handler)
        self.results = []
        self.summary = {}
        try:
            if self.seed is not None:
                seed_everything(self.seed)
            self._prepare_pool()
            order = np.random.default_rng(self.seed).permutation(self._shadow_indices)[:self.n_runs - 1]
            self.target_indices = [-1, *map(int, order)]
            identity = {"attack": self.attack_name, "config": self.attack_config, "seed": self.seed,
                        "targets": self.target_indices, "models": self._model_hashes, "records": self._records.tolist()}
            self.output_dir = Path(self.handler.configs.audit.output_dir) / "robustness" / hash_config(identity)[:16]
            self.output_dir.mkdir(parents=True, exist_ok=True)
            summary_path = self.output_dir / "summary.json"
            summary_path.unlink(missing_ok=True)
            for run_index, target_index in enumerate(self.target_indices):
                if self.seed is not None:
                    seed_everything((self.seed + run_index) % (MAX_SEED + 1))
                self._run_once(run_index, target_index)
            self.summary = self._summarize()
            temporary = self.output_dir / ".summary.json.tmp"
            temporary.write_text(json.dumps({"setup": identity, "runs": [entry["metadata"] for entry in self.results],
                                             "rotated_target_summary": self.summary}, indent=2, allow_nan=False))
            temporary.replace(summary_path)
            return self.results
        finally:
            for factory, state in snapshots:
                if state is None:
                    factory.delete_instance()
                else:
                    instance = factory()
                    vars(instance).clear()
                    vars(instance).update(state)
            AttackFactoryMIA.shadow_model_handler, AttackFactoryMIA.distillation_model_handler = factory_state

    def _prepare_pool(self) -> None:
        """Reuse compatible trained shadows and verify the target and shadow setup agree."""
        pool_context = self._copy_handler()
        # Ordinary shadow metadata does not identify the sampling strategy. Keep the
        # random pool separate so balanced caches cannot silently change this design.
        pool_context.configs.audit.output_dir = str(Path(self.handler.configs.audit.output_dir) / "robustness" / "pool")
        self._pool_handler = copy(ShadowModelHandler(pool_context))
        setup = self.handler.target_model_metadata
        pool = self._pool_handler
        expected = (self.handler.target_model_blueprint.__name__, self.handler.configs.target.module_path,
                    setup.init_params, setup.optimizer.name.lower(), setup.optimizer.params,
                    setup.criterion.name.lower(), setup.criterion.params, setup.epochs,
                    setup.data_loader.params.get("batch_size"))
        actual = (pool.model_class, pool.model_path, pool.init_params, pool.optimizer_name, pool.optimizer_config,
                  pool.criterion_name, pool.loss_config, pool.epochs, pool.batch_size)
        if actual != expected:
            raise ValueError("Robustness requires the same target and shadow training setup")
        # The handler truncates population_size * fraction. Round the ratio upward
        # by one float step so an exact count such as 15 / 22 cannot become 14.
        fraction = np.nextafter(len(self.handler.train_indices) / len(self._records), 1.0)
        self._shadow_indices = pool.create_shadow_models(self.num_shadow_models, self._records,
                                                        training_fraction=fraction, sampling_method="random")
        self._shadow_indices = list(map(int, self._shadow_indices))
        self._metadata = {-1: setup}
        self._metadata.update(zip(self._shadow_indices, pool.get_shadow_model_metadata(self._shadow_indices)))
        for metadata in self._metadata.values():
            training = np.asarray(metadata.train_indices)
            if (len(training) != len(self.handler.train_indices) or len(np.unique(training)) != len(training)
                    or not np.all(np.isin(training, self._records))):
                raise ValueError("Each pool model must train on a target-sized subset of the audit records")
        self._model_hashes = {str(index): hash_model(self._load_model(index).model_obj)
                              for index in [-1, *self._shadow_indices]}

    def _load_model(self, index:int) -> PytorchModel:
        """Load a pool member without loading all model weights into memory."""
        if index == -1:
            return PytorchModel(self.handler.target_model, self.handler.get_criterion())
        return self._pool_handler._load_shadow_model(index)

    def _load_logits(self, index:int, records:np.ndarray) -> np.ndarray:
        """Reuse the PR's cached-signal approach, aligning every row by record identity."""
        if index == -1:
            logits = self._pool_handler.load_logits(name=f"target_{self._pool_handler.target_model_hash}")
        else:
            logits = self._pool_handler.load_logits(indx=index)
        return logits[[self._positions[int(record)] for record in records]]

    def _copy_handler(self) -> MIAHandler:
        """Copy the execution context and rebind user callbacks to the new handler."""
        handler = copy(self.handler)
        for name, value in vars(handler).items():
            if isinstance(value, MethodType) and value.__self__ is self.handler:
                setattr(handler, name, MethodType(value.__func__, handler))
        handler.configs = deepcopy(self.handler.configs)
        handler.target_model_metadata = self.handler.target_model_metadata.model_copy(deep=True)
        handler.dataloader_config = deepcopy(self.handler.dataloader_config)
        if getattr(handler, "modality_extension", None) is not None:
            handler.modality_extension = deepcopy(handler.modality_extension)
        return handler

    def _run_once(self, run_index:int, target_index:int) -> None:
        """Bind a fresh attack to this target and all remaining reference models."""
        handler = self._copy_handler()
        handler.configs.audit.output_dir = str(self.output_dir / f"run_{run_index}")
        handler.target_model = self._load_model(target_index).model_obj
        target_metadata = self._metadata[target_index]
        handler.train_indices = np.asarray(target_metadata.train_indices)
        handler.test_indices = self._records[~np.isin(self._records, handler.train_indices)]
        handler.target_model_metadata = self.handler.target_model_metadata.model_copy(deep=True, update={
            "train_indices": handler.train_indices.tolist(), "test_indices": handler.test_indices.tolist(),
            "num_train": target_metadata.num_train, "train_result": target_metadata.train_result,
            "test_result": target_metadata.test_result})
        references = [index for index in [-1, *self._shadow_indices] if index != target_index]
        attack = AttackFactoryMIA.create_attack(self.attack_name, deepcopy(self.attack_config), handler)
        ShadowModelHandler(handler).use_reference_pool(references, self._load_model,
                                                       self._metadata.__getitem__, self._load_logits)
        attack.prepare_attack()
        result = attack.run_attack()
        if not isinstance(result, MIAResult):
            raise TypeError("Robustness requires an attack returning MIAResult")
        metadata = {"target": target_index, "references": references,
                    "target_hash": self._model_hashes[str(target_index)],
                    "audit_record_count": len(result.true), "member_count": int(np.sum(result.true))}
        metadata["audit_record_ids"] = None if result.audit_indices is None else result.audit_indices.tolist()
        metadata["attack_id"] = attack.attack_id
        metadata["result_id"] = result.id
        result.save(attack_obj=attack, output_dir=handler.configs.audit.output_dir)
        self.results.append({"metadata": metadata, "result_object": result})

    def _summarize(self) -> dict[str, Any]:
        """Adapt PR #351's pointwise percentile aggregation to rotated-target results."""
        rotated = [entry["result_object"] for entry in self.results[1:]]
        summary = {"n_targets": len(rotated), "interpretation": "empirical variation across shared-pool targets"}
        aucs = [result.roc_auc for result in rotated if result.roc_auc is not None]
        if aucs:
            summary["roc_auc"] = {"mean": float(np.mean(aucs)),
                                  "percentiles_2_5_97_5": np.percentile(aucs, [2.5, 97.5]).tolist()}
        if all(result.roc_mode == "full" for result in rotated):
            fpr_grid = np.unique(np.concatenate([result.fpr for result in rotated]))
            curves = []
            for result in rotated:
                # Take the upper ROC value at each repeated FPR, independent of tie ordering.
                fpr, inverse = np.unique(result.fpr, return_inverse=True)
                upper_tpr = np.zeros(len(fpr))
                np.maximum.at(upper_tpr, inverse, result.tpr)
                curves.append(np.interp(fpr_grid, fpr, upper_tpr))
            summary["roc"] = {"fpr": fpr_grid.tolist(), "mean_tpr": np.mean(curves, axis=0).tolist(),
                              "pointwise_tpr_percentiles_2_5_97_5": np.percentile(curves, [2.5, 97.5], axis=0).tolist()}
        return summary
