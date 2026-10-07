#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Carlini et al. diffusion training-data extraction attack."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator, model_validator
from torch import Tensor

from leakpro.attacks.extraction_attacks.abstract_extraction import AbstractExtraction, AttackState, ExtractionConfig
from leakpro.attacks.extraction_attacks.protocols import ExtractionAdapter
from leakpro.attacks.extraction_attacks.utils_generative import (
    json_safe,
    normalize_conditions,
    progress_batches,
    require_authorized,
    to_zero_one,
    validate_image_batch,
)
from leakpro.attacks.extraction_attacks.utils_generative.carlini_graph import clique_medoid, maximum_clique
from leakpro.attacks.extraction_attacks.utils_generative.image_metrics import (
    carlini_reference_scores,
    nearest_reference,
    tiled_l2_pairwise,
)
from leakpro.reporting.extraction_result import CandidateRecord, ExtractionResult
from leakpro.utils.device import get_device
from leakpro.utils.import_helper import Self
from leakpro.utils.save_load import hash_config


def _condition_hash(value: object) -> str:  # noqa: C901 - condition types must have distinct identities
    """Identify a Carlini condition without conflating tensors and JSON values."""
    if isinstance(value, Tensor):
        tensor = value.detach().to(device="cpu").contiguous()
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all():
            raise ValueError("Extraction conditions must not contain NaN or infinity.")
        identity = {"type": "tensor", "dtype": str(tensor.dtype), "shape": list(tensor.shape),
                    "digest": hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
    elif isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays are not supported as extraction conditions.")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("Extraction conditions must not contain NaN or infinity.")
        array = np.ascontiguousarray(value)
        identity = {"type": "ndarray", "dtype": str(array.dtype), "shape": list(array.shape),
                    "digest": hashlib.sha256(array.tobytes()).hexdigest()}
    elif isinstance(value, BaseModel):
        identity = {"type": f"{type(value).__module__}.{type(value).__qualname__}",
                    "value": _condition_hash(value.model_dump(mode="json"))}
    elif isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Extraction condition dictionaries require string keys.")
        identity = {"type": "dict", "value": {key: _condition_hash(item) for key, item in value.items()}}
    elif isinstance(value, (list, tuple)):
        identity = {"type": type(value).__name__, "value": [_condition_hash(item) for item in value]}
    elif isinstance(value, bytes):
        identity = {"type": "bytes", "value": value.hex()}
    else:
        identity = {"type": type(value).__name__, "value": json_safe(value)}
    return hash_config(identity)


class AttackCarliniExtraction(AbstractExtraction):
    """Generate-and-filter attack from USENIX Security 2023.

    ``conditional_black_box`` implements prompt repetition, tiled-L2 graph
    construction, maximum-clique filtering, and clique ranking.
    ``unconditional_reference_audit`` implements the adaptive nearest-neighbor
    score used for the paper's CIFAR-10 extraction audit.
    """

    class AttackConfig(ExtractionConfig):
        """Carlini generation and candidate-scoring settings."""

        image_range: Literal["zero_one", "minus_one_one"] = "zero_one"
        generation_batch_size: int = Field(default=64, ge=1)
        distance_block_size: int = Field(default=64, ge=1)
        distance_device: str = Field(default="cpu", pattern=r"^(auto|cpu|(?:cuda|hpu|mps)(?::[0-9]+)?)$")
        mode: Literal["conditional_black_box", "unconditional_reference_audit"] = "conditional_black_box"
        num_generations_per_condition: int = Field(default=500, ge=2)
        num_unconditional_generations: int = Field(default=1_000_000, ge=1)
        tile_grid: tuple[int, int] = (4, 4)
        tiled_l2_threshold: float = Field(default=0.1, gt=0)
        min_clique_size: int = Field(default=10, ge=2)
        verification_l2_threshold: float = Field(default=0.15, gt=0)
        reference_alpha: float = Field(default=0.5, gt=0)
        reference_neighbors: int = Field(default=50, ge=2)
        ratio_threshold: float = Field(default=1.0, gt=0)
        max_extractions: int | None = Field(default=None, ge=1)

        @field_validator("tile_grid")
        @classmethod
        def validate_tile_grid(cls, value: tuple[int, int]) -> tuple[int, int]:
            """Reject an image grid with a nonpositive dimension."""
            if any(size < 1 for size in value):
                raise ValueError("tile_grid values must be positive.")
            return value

        @model_validator(mode="after")
        def validate_clique_budget(self) -> Self:
            """Require enough generations to form a qualifying clique."""
            if self.mode == "conditional_black_box" and self.min_clique_size > self.num_generations_per_condition:
                raise ValueError("min_clique_size cannot exceed num_generations_per_condition.")
            return self

    def __init__(
        self,
        adapter: ExtractionAdapter[Tensor],
        configs: AttackConfig | dict[str, Any],
        *,
        audit_hash: str,
        conditions: Sequence[Any] | None = None,
        reference_images: Tensor | None = None,
    ) -> None:
        self.adapter = adapter
        self.config = configs if isinstance(configs, self.AttackConfig) else self.AttackConfig(**configs)
        self.configs = self.config
        self.optuna_params = 0
        self.conditions = normalize_conditions(conditions)
        self.audit_hash = (
            hash_config({"audit_hash": audit_hash, "conditions": [_condition_hash(c) for c in self.conditions]})
            if self.conditions is not None else audit_hash
        )
        self.reference_images = reference_images
        self.state = AttackState.CREATED
        identity_config = self.config.model_dump(mode="json", exclude={"overwrite_results"})
        if self.config.mode == "unconditional_reference_audit":
            identity_config["reference_score"] = "candidate_neighbors"
        result_hash = hash_config({"audit_hash": self.audit_hash, "config": identity_config})[:16]
        self.result_id = f"carlini-diffusion-extraction-{result_hash}"
        self.attack_id = self.result_id
        self._references_zero_one: Tensor | None = None
        self._sampling_calls = 0
        self._initialize_trace()

    def description(self) -> dict[str, str]:
        """Return the attack's reference and implemented scope."""
        return {
            "title": "Carlini Diffusion Training-Data Extraction",
            "reference": (
                "Nicholas Carlini et al., Extracting Training Data from Diffusion Models, "
                "USENIX Security 2023, https://www.usenix.org/conference/usenixsecurity23/presentation/carlini"
            ),
            "summary": "Generate candidates, then retain unusually repeatable or reference-neighbor outliers.",
            "threat_model": (
                "Black-box generation for conditional clique filtering; reference-data audit for unconditional extraction."
            ),
            "scope": (
                "Implements Sections 4.2.1 and 5.1. The unpublished tiled-L2 graph-edge threshold remains configurable "
                "and must be calibrated for the audited generator. Selecting the clique medoid as the retained image "
                "is an implementation-defined representative policy; the paper specifies clique qualification and "
                "condition ranking, but not that selection rule."
            ),
        }

    def prepare_attack(self) -> None:
        """Prepare this attack exactly once."""
        self._prepare_once(self._prepare_attack)

    def _prepare_attack(self) -> None:  # noqa: C901 - preparation checks must precede target sampling
        """Validate authorization, model surface, conditions, and reference data."""
        require_authorized(self.config.authorized_audit)
        if self.config.distance_device == "auto" and get_device().type not in {"cpu", "cuda", "hpu", "mps"}:
            raise ValueError("Automatic extraction distance device must be CPU, CUDA, HPU, or MPS.")
        if (
            not isinstance(self.adapter, ExtractionAdapter)
            or not callable(getattr(self.adapter, "sample", None))
            or not hasattr(self.adapter, "image_shape")
        ):
            raise TypeError("Carlini requires an adapter with sample() and image_shape.")
        if (
            not isinstance(self.adapter.image_shape, tuple)
            or len(self.adapter.image_shape) != 3
            or any(not isinstance(size, int) or isinstance(size, bool) or size < 1 for size in self.adapter.image_shape)
        ):
            raise ValueError("adapter.image_shape must be a positive CHW tuple.")
        if self.config.mode == "conditional_black_box" and not self.conditions:
            raise ValueError("conditional_black_box mode requires at least one condition.")
        if self.config.mode == "conditional_black_box" and any(condition is None for condition in self.conditions or []):
            raise ValueError("conditional_black_box conditions must not contain None.")
        if self.config.mode == "unconditional_reference_audit" and self.reference_images is None:
            raise ValueError("unconditional_reference_audit mode requires reference_images.")
        if self.reference_images is not None:
            references = validate_image_batch(self.reference_images, self.adapter.image_shape)
            self._references_zero_one = to_zero_one(references.detach().cpu(), self.config.image_range)
            if (
                self.config.mode == "unconditional_reference_audit"
                and self._references_zero_one.shape[0] < self.config.reference_neighbors
            ):
                raise ValueError("reference_images contains fewer samples than reference_neighbors.")
        if self.config.mode == "conditional_black_box":
            _channels, height, width = self.adapter.image_shape
            if height % self.config.tile_grid[0] or width % self.config.tile_grid[1]:
                raise ValueError("adapter image dimensions must be divisible by tile_grid.")
        self._record_trace(
            "prepared",
            mode=self.config.mode,
            condition_count=len(self.conditions or []),
            reference_count=0 if self._references_zero_one is None else int(self._references_zero_one.shape[0]),
        )

    def _generate(self, count: int, condition: object | None, seed_offset: int) -> Tensor:
        generated: list[Tensor] = []
        for batch_index, (start, end) in enumerate(
            progress_batches(count, self.config.generation_batch_size, "Carlini generation")
        ):
            batch_size = end - start
            conditions = [condition] * batch_size if condition is not None else None
            batch = self.adapter.sample(
                batch_size,
                conditions=conditions,
                seed=self.config.random_seed + seed_offset + batch_index,
            )
            self._sampling_calls += 1
            batch = validate_image_batch(batch, self.adapter.image_shape, expected_count=batch_size)
            generated.append(to_zero_one(batch.detach().cpu(), self.config.image_range))
        return torch.cat(generated, dim=0)

    def _conditional_attack(self) -> ExtractionResult:
        retained_images: list[Tensor] = []
        records: list[CandidateRecord] = []
        condition_summaries: list[dict[str, Any]] = []
        if self.conditions is None:
            raise RuntimeError("Conditional inputs were not prepared.")
        for condition_index, condition in enumerate(self.conditions):
            hash = _condition_hash(condition)
            condition_id = f"condition:{condition_index}:{hash[:12]}"
            images = self._generate(
                self.config.num_generations_per_condition,
                condition,
                seed_offset=condition_index * 1_000_003,
            )
            distances = tiled_l2_pairwise(
                images,
                tile_grid=self.config.tile_grid,
                block_size=self.config.distance_block_size,
                device=self.config.distance_device,
            )
            adjacency = distances.le(self.config.tiled_l2_threshold)
            adjacency.fill_diagonal_(False)
            clique = maximum_clique(adjacency)
            summary = {
                "condition_index": condition_index,
                "condition_hash": hash,
                "largest_clique_size": len(clique),
                "qualified": len(clique) >= self.config.min_clique_size,
            }
            if len(clique) >= self.config.min_clique_size:
                medoid_index, mean_distance = clique_medoid(clique, distances)
                image_index = len(retained_images)
                retained_images.append(images[medoid_index])
                records.append(
                    CandidateRecord(
                        image_index=image_index,
                        source=condition_id,
                        score=mean_distance,
                        support_indices=clique,
                        metadata={
                            "condition_index": condition_index,
                            "condition_hash": hash,
                            "medoid_generation_index": medoid_index,
                            "largest_clique_size": len(clique),
                            "tiled_l2_threshold": self.config.tiled_l2_threshold,
                        },
                    )
                )
                summary["mean_clique_distance"] = mean_distance
            condition_summaries.append(summary)
            self._record_trace(
                "condition_complete",
                condition_index=condition_index,
                generated=int(images.shape[0]),
                largest_clique_size=len(clique),
                qualified=summary["qualified"],
            )

        if retained_images:
            images_tensor = torch.stack(retained_images)
            order = sorted(range(len(records)), key=lambda index: (records[index].score, records[index].source))
            images_tensor = images_tensor[order]
            records = [records[index].model_copy(update={"image_index": new_index}) for new_index, index in enumerate(order)]
        else:
            images_tensor = torch.empty((0, *self.adapter.image_shape), dtype=torch.float32)

        verified_count = 0
        if records and self._references_zero_one is not None:
            nearest_indices, nearest_distances = nearest_reference(
                images_tensor,
                self._references_zero_one,
                block_size=self.config.distance_block_size,
                device=self.config.distance_device,
            )
            updated: list[CandidateRecord] = []
            for index, record in enumerate(records):
                verified = float(nearest_distances[index]) <= self.config.verification_l2_threshold
                verified_count += int(verified)
                updated.append(
                    record.model_copy(
                        update={
                            "nearest_reference_index": int(nearest_indices[index]),
                            "nearest_reference_distance": float(nearest_distances[index]),
                            "verified": verified,
                        }
                    )
                )
            records = updated

        self._record_trace(
            "run_complete",
            sampling_calls=self._sampling_calls,
            images_generated=len(self.conditions) * self.config.num_generations_per_condition,
            candidate_count=len(records),
        )
        return ExtractionResult(
            name="Carlini Diffusion Extraction Result",
            result_id=self.result_id,
            config=self.config,
            images=images_tensor,
            candidates=records,
            metrics={
                "mode": self.config.mode,
                "conditions_audited": len(self.conditions),
                "images_generated": len(self.conditions) * self.config.num_generations_per_condition,
                "candidate_count": len(records),
                "verified_count": verified_count if self._references_zero_one is not None else None,
                "condition_summaries": condition_summaries,
            },
            provenance={**self.description(), "audit_hash": self.audit_hash},
            execution_trace=self.execution_trace,
            overwrite=self.config.overwrite_results,
        )

    def _unconditional_reference_attack(self) -> ExtractionResult:
        if self._references_zero_one is None:
            raise RuntimeError("Reference images were not prepared.")
        best_by_reference: dict[int, tuple[float, Tensor, CandidateRecord]] = {}
        total_generated = 0
        for batch_index, (start, end) in enumerate(
            progress_batches(self.config.num_unconditional_generations, self.config.generation_batch_size,
                             "Carlini generation and reference scoring")
        ):
            batch = self.adapter.sample(
                end - start,
                conditions=None,
                seed=self.config.random_seed + batch_index,
            )
            self._sampling_calls += 1
            batch = to_zero_one(
                validate_image_batch(batch, self.adapter.image_shape, expected_count=end - start).detach().cpu(),
                self.config.image_range,
            )
            scores = carlini_reference_scores(
                batch,
                self._references_zero_one,
                neighbors=self.config.reference_neighbors,
                alpha=self.config.reference_alpha,
                block_size=self.config.distance_block_size,
                device=self.config.distance_device,
            )
            for local_index in torch.nonzero(scores.ratios.le(self.config.ratio_threshold), as_tuple=False).flatten().tolist():
                reference_index = int(scores.nearest_indices[local_index])
                ratio = float(scores.ratios[local_index])
                nearest_distance = float(scores.nearest_distances[local_index])
                record = CandidateRecord(
                    image_index=0,
                    source="unconditional",
                    score=ratio,
                    nearest_reference_index=reference_index,
                    nearest_reference_distance=nearest_distance,
                    verified=nearest_distance <= self.config.verification_l2_threshold,
                    metadata={
                        "generation_index": total_generated + local_index,
                        "neighbor_mean_distance": float(scores.neighbor_mean_distances[local_index]),
                        "reference_alpha": self.config.reference_alpha,
                    },
                )
                previous = best_by_reference.get(reference_index)
                if previous is None or ratio < previous[0]:
                    best_by_reference[reference_index] = (ratio, batch[local_index], record)
            total_generated += batch.shape[0]

        ordered = sorted(best_by_reference.values(), key=lambda item: (item[0], item[2].nearest_reference_index))
        if self.config.max_extractions is not None:
            ordered = ordered[: self.config.max_extractions]
        images = (
            torch.stack([item[1] for item in ordered])
            if ordered
            else torch.empty((0, *self.adapter.image_shape), dtype=torch.float32)
        )
        records = [item[2].model_copy(update={"image_index": index}) for index, item in enumerate(ordered)]
        self._record_trace(
            "run_complete",
            sampling_calls=self._sampling_calls,
            images_generated=total_generated,
            candidate_count=len(records),
        )
        return ExtractionResult(
            name="Carlini Unconditional Reference Audit Result",
            result_id=self.result_id,
            config=self.config,
            images=images,
            candidates=records,
            metrics={
                "mode": self.config.mode,
                "images_generated": total_generated,
                "candidate_count": len(records),
                "unique_reference_matches": len(records),
                "ratio_threshold": self.config.ratio_threshold,
            },
            provenance={**self.description(), "audit_hash": self.audit_hash},
            execution_trace=self.execution_trace,
            overwrite=self.config.overwrite_results,
        )

    def run_attack(self) -> ExtractionResult:
        """Run this prepared attack exactly once."""
        return self._execute_once(self._run_attack)

    def _run_attack(self) -> ExtractionResult:
        """Run the configured Carlini extraction path."""
        if self.config.mode == "conditional_black_box":
            result = self._conditional_attack()
        else:
            result = self._unconditional_reference_attack()
        return result
