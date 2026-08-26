#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Serializable extraction result contract."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from leakpro.attacks.extraction_attacks.utils import json_safe


@dataclass
class _BundlePaths:
    result: Optional[Path]
    data: Optional[Path]


class CandidateRecord(BaseModel):
    """Metadata for one retained image candidate."""

    image_index: int = Field(ge=0)
    source: str
    score: Optional[float] = None
    support_indices: List[int] = Field(default_factory=list)
    nearest_reference_index: Optional[int] = Field(default=None, ge=0)
    nearest_reference_distance: Optional[float] = Field(default=None, ge=0)
    verified: Optional[bool] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")


class ExtractionResult:
    """LeakPro-compatible result containing images, metrics, and provenance."""

    def __init__(
        self,
        *,
        name: str,
        result_id: str,
        config: BaseModel | dict[str, Any],
        images: Tensor,
        candidates: list[CandidateRecord],
        metrics: dict[str, Any],
        provenance: dict[str, Any],
        execution_trace: list[dict[str, Any]] | None = None,
        overwrite: bool = False,
    ) -> None:
        if re.fullmatch(r"[A-Za-z0-9._-]+", result_id) is None:
            raise ValueError("result_id contains unsafe path characters.")
        if images.ndim != 4:
            raise ValueError("ExtractionResult.images must be BCHW.")
        if not torch.isfinite(images).all():
            raise ValueError("ExtractionResult.images contains NaN or infinity.")
        if len(candidates) != images.shape[0]:
            raise ValueError("Candidate metadata count must match the number of images.")
        self.name = name
        self.id = result_id
        self.config = config.model_dump(mode="json") if isinstance(config, BaseModel) else dict(config)
        self.images = images.detach().to(device="cpu", dtype=torch.float32).contiguous()
        self.candidates = candidates
        self.metrics = json_safe(metrics)
        self.provenance = json_safe(provenance)
        self.execution_trace = json_safe(execution_trace or [])
        self.overwrite = overwrite

    @property
    def result(self) -> dict[str, Any]:
        """Return a report-friendly metadata view without embedding image pixels."""
        return {
            "name": self.name,
            "id": self.id,
            "config": json_safe(self.config),
            "metrics": self.metrics,
            "provenance": self.provenance,
            "execution_trace": self.execution_trace,
            "candidate_count": len(self.candidates),
            "candidates": [candidate.model_dump(mode="json") for candidate in self.candidates],
        }

    def save(self, attack_obj: object | None = None, output_dir: str | os.PathLike[str] = "./leakpro_output") -> None:  # noqa: ARG002
        """Stage and publish one result while holding an exclusive result claim."""
        output_root = Path(output_dir).resolve()
        results_dir = output_root / "results"
        result_dir = results_dir / self.id
        data_dir = output_root / "data_objects"
        data_path = data_dir / f"{self.id}.json"
        claims_dir = output_root / ".extraction_result_claims"
        results_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        claims_dir.mkdir(parents=True, exist_ok=True)
        claim = claims_dir / self.id
        try:
            claim.mkdir()
        except FileExistsError as error:
            raise FileExistsError(f"Extraction result {self.id!r} is already being saved.") from error
        try:
            if not self.overwrite and (result_dir.exists() or data_path.exists()):
                raise FileExistsError(
                    f"Extraction result {self.id!r} already exists. Set overwrite_results=true to replace it."
                )
            staged = self._stage_bundle(results_dir, data_dir, self.result)
            self._publish_bundle(staged, result_dir, data_path, results_dir, data_dir)
        finally:
            claim.rmdir()

    def _stage_bundle(self, results_dir: Path, data_dir: Path, metadata: dict[str, Any]) -> _BundlePaths:
        staged_result_dir = Path(tempfile.mkdtemp(dir=results_dir, prefix=f".{self.id}-stage-"))
        staged_data_path: Optional[Path] = None
        try:
            self._write_json(staged_result_dir / "result.json", metadata)
            np.savez_compressed(staged_result_dir / "candidates.npz", images=self.images.numpy())
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=data_dir,
                prefix=f".{self.id}-stage-",
                suffix=".tmp",
                delete=False,
            ) as stream:
                staged_data_path = Path(stream.name)
                self._dump_json(stream, metadata)
        except BaseException:
            shutil.rmtree(staged_result_dir, ignore_errors=True)
            if staged_data_path is not None:
                staged_data_path.unlink(missing_ok=True)
            raise
        return _BundlePaths(staged_result_dir, staged_data_path)

    def _publish_bundle(
        self,
        staged: _BundlePaths,
        result_dir: Path,
        data_path: Path,
        results_dir: Path,
        data_dir: Path,
    ) -> None:
        if staged.result is None or staged.data is None:
            raise RuntimeError("The staged extraction bundle is incomplete.")
        backups = _BundlePaths(None, None)
        commit_succeeded = False
        try:
            if not self.overwrite and (result_dir.exists() or data_path.exists()):
                raise FileExistsError(
                    f"Extraction result {self.id!r} appeared while saving; the competing bundle was left unchanged."
                )
            if self.overwrite:
                backups = self._backup_existing(result_dir, data_path, results_dir, data_dir)
            os.replace(staged.result, result_dir)
            os.replace(staged.data, data_path)
            commit_succeeded = True
        except BaseException:
            result_was_published = not staged.result.exists() and result_dir.exists()
            data_was_published = not staged.data.exists() and data_path.exists()
            if result_was_published:
                shutil.rmtree(result_dir, ignore_errors=True)
            if data_was_published:
                data_path.unlink(missing_ok=True)
            self._restore_backups(backups, result_dir, data_path)
            raise
        finally:
            self._remove_bundle(staged)
            if commit_succeeded:
                self._remove_bundle(backups)

    def _backup_existing(self, result_dir: Path, data_path: Path, results_dir: Path, data_dir: Path) -> _BundlePaths:
        backups = _BundlePaths(None, None)
        try:
            if result_dir.exists():
                backups.result = self._unused_directory_path(results_dir, f".{self.id}-backup-")
                os.replace(result_dir, backups.result)
            if data_path.exists():
                backups.data = self._unused_file_path(data_dir, f".{self.id}-backup-", ".json")
                os.replace(data_path, backups.data)
        except BaseException:
            self._restore_backups(backups, result_dir, data_path)
            raise
        return backups

    @staticmethod
    def _restore_backups(backups: _BundlePaths, result_dir: Path, data_path: Path) -> None:
        if backups.result is not None and backups.result.exists():
            if result_dir.exists():
                shutil.rmtree(result_dir)
            os.replace(backups.result, result_dir)
        if backups.data is not None and backups.data.exists():
            data_path.unlink(missing_ok=True)
            os.replace(backups.data, data_path)

    @staticmethod
    def _remove_bundle(bundle: _BundlePaths) -> None:
        if bundle.result is not None and bundle.result.exists():
            shutil.rmtree(bundle.result, ignore_errors=True)
        if bundle.data is not None and bundle.data.exists():
            bundle.data.unlink(missing_ok=True)

    @staticmethod
    def _unused_directory_path(parent: Path, prefix: str) -> Path:
        path = Path(tempfile.mkdtemp(dir=parent, prefix=prefix))
        path.rmdir()
        return path

    @staticmethod
    def _unused_file_path(parent: Path, prefix: str, suffix: str) -> Path:
        descriptor, name = tempfile.mkstemp(dir=parent, prefix=prefix, suffix=suffix)
        os.close(descriptor)
        path = Path(name)
        path.unlink()
        return path

    @staticmethod
    def _dump_json(stream: TextIO, payload: dict[str, Any]) -> None:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")

    @classmethod
    def _write_json(cls, path: Path, payload: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as stream:
            cls._dump_json(stream, payload)
