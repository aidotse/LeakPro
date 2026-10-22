#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""End-to-end test of the webapp PET optimization flow against a live TestClient.

The runner is executed in-process instead of as a detached subprocess, on a tiny
tabular problem (two trials, one shadow model), so the whole path
start -> poll -> verify -> adopt, plus its rejection paths, runs in well under a
minute on CPU. Everything that scores a configuration is the real library code.
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pytest

pytest.importorskip("fastapi", reason="webapp extras (fastapi) not installed")

from leakpro.utils.import_helper import Self  # noqa: E402

ARCH = """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self, num_classes=3, num_features=4):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.fc = nn.Sequential(nn.Linear(num_features, 16), nn.ReLU(), nn.Linear(16, num_classes))
    def forward(self, x):
        return self.fc(x)
"""

MODEL = "m1"


def _blobs(n: int = 400, classes: int = 3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, classes, size=n)
    x = np.eye(classes, 4)[y] * 2.0 + rng.normal(scale=1.5, size=(n, 4))
    return x.astype(np.float32), y.astype(np.int64)


class TestPetFlow:
    """start -> campaign -> verify -> adopt, with the runner executed in-process."""

    @pytest.fixture(autouse=True)
    def _setup(self: Self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from fastapi.testclient import TestClient

        from leakpro.webapp.backend import main as backend_main
        from leakpro.webapp.backend import pet_runner, security
        from leakpro.webapp.backend.dataio import ArrayDataset

        monkeypatch.setattr(backend_main, "JOBS_ROOT", tmp_path)
        # Small enough to finish in seconds; the population rule (2 x target)
        # and the resolution fallback (100 non-members cannot resolve 1%) still apply.
        monkeypatch.setattr(pet_runner, "MAX_TARGET", 120)
        monkeypatch.setattr(pet_runner, "MAX_TEST", 100)
        monkeypatch.setattr(pet_runner, "VERIFY_N_SHADOW", 2)

        self.calls: list[list[str]] = []

        def spawn_in_process(job_id: str, model_name: str, extra: list[str]) -> None:
            self.calls.append(extra)
            job_dir = backend_main._job_dir(job_id)
            if extra:
                pet_runner.run_verification(job_dir, model_name, int(extra[1]))
            else:
                pet_runner.run_search(job_dir, model_name)

        monkeypatch.setattr(backend_main, "_pet_spawn", spawn_in_process)

        self.client = TestClient(backend_main.app)
        self.auth = {"Authorization": f"Bearer {security.API_TOKEN}"}
        self.backend = backend_main
        self.pet_runner = pet_runner

        self.job_id = self.client.post("/jobs", headers=self.auth).json()["job_id"]
        job_dir = backend_main._job_dir(self.job_id)
        x, y = _blobs()
        data_path = job_dir / "data_converted.pkl"
        joblib.dump(ArrayDataset(data=x, targets=y, feature_names=[f"f{i}" for i in range(4)]), data_path)
        (job_dir / "arch.py").write_text(ARCH)
        (job_dir / "models" / MODEL).mkdir(parents=True)
        job = backend_main._jobs[self.job_id]
        job["data_path"] = str(data_path)
        job["data_meta"] = {"data_type": "tabular"}
        job["models"] = [{
            "name": MODEL, "source": "trained", "status": "ready", "test_accuracy": 0.8,
            "train_params": {"epochs": 5, "optimizer": "adam", "learning_rate": 0.05, "batch_size": 32},
        }]
        backend_main._save_job(self.job_id)

    # ── helpers ───────────────────────────────────────────────────────────
    def _start(self: Self, **params: object) -> object:
        body = {"n_trials": 2, "n_shadow": 1, **params}
        return self.client.post(f"/jobs/{self.job_id}/pet/start", params={"model_name": MODEL},
                                json=body, headers=self.auth)

    def _campaign(self: Self, model: str = MODEL) -> dict:
        res = self.client.get(f"/jobs/{self.job_id}/pet/campaign", params={"model_name": model}, headers=self.auth)
        assert res.status_code == 200, res.text
        return res.json()

    # ── the happy path, in one test so the expensive search runs once ─────
    def test_search_verify_adopt(self: Self) -> None:
        """The full flow against real library code on a tiny problem."""
        assert self._campaign()["status"] == "idle"

        assert self._start().status_code == 200
        run = self._campaign()
        assert run["status"] == "done"
        assert run["n_trials"] == 2
        assert run["baseline_utility"] == 0.8
        # 100 non-members cannot resolve 1% FPR: the runner says so and measures at 10%.
        assert run["proxy_fpr"] == 0.1
        assert "10%" in run["resolution_warning"]
        assert len(run["settings"]) == 2
        for setting in run["settings"]:
            assert set(setting["config"]) == {"noise_multiplier", "max_grad_norm", "learning_rate", "batch_size"}
            assert 0.0 <= setting["utility"] <= 1.0
            assert setting["proxy_fpr"] == 0.1
            assert setting["tuning_accounted"] is False
        measured = [s for s in run["settings"] if s["attack_tpr"] is not None]
        assert measured, "no trial resolved a TPR"
        assert set(run["best_indices"]) <= {s["index"] for s in measured}

        # Verification re-audits the saved model with more shadow models; no retraining.
        index = run["best_indices"][0]
        res = self.client.post(f"/jobs/{self.job_id}/pet/verify", params={"model_name": MODEL},
                               json={"index": index}, headers=self.auth)
        assert res.status_code == 200, res.text
        assert self.calls[-1] == ["--verify", str(index)]
        res = self.client.get(f"/jobs/{self.job_id}/pet/verify",
                              params={"model_name": MODEL, "index": index}, headers=self.auth)
        verification = res.json()
        assert verification["status"] == "done", verification
        assert verification["estimated"]["utility"] == pytest.approx(
            next(s["utility"] for s in run["settings"] if s["index"] == index))
        assert verification["verified"]["n_shadow"] == 2
        assert verification["verified"]["utility"] == verification["estimated"]["utility"]
        assert "attack_tpr" in verification["verified"]

        # Adopting an unverified setting is refused; the verified one lands as a result row.
        other = next(s["index"] for s in run["settings"] if s["index"] != index)
        res = self.client.post(f"/jobs/{self.job_id}/pet/adopt", params={"model_name": MODEL},
                               json={"index": other}, headers=self.auth)
        assert res.status_code == 409
        res = self.client.post(f"/jobs/{self.job_id}/pet/adopt", params={"model_name": MODEL},
                               json={"index": index}, headers=self.auth)
        assert res.status_code == 200, res.text
        rows = self.backend._jobs[self.job_id]["results"]
        adopted = next(r for r in rows if r["model_name"] == f"{MODEL} (optimized)")
        assert adopted["optimized"] is True and adopted["dpsgd"] is True
        assert adopted["pet_setting_index"] == index
        assert set(adopted["pet_config"]) == {"noise_multiplier", "max_grad_norm", "learning_rate", "batch_size"}
        assert adopted["pet_proxy_fpr"] == 0.1
        # Persisted: a restart would reload it.
        state = json.loads((self.backend._job_dir(self.job_id) / "state.json").read_text())
        assert any(r["model_name"] == f"{MODEL} (optimized)" for r in state["results"])

        # Starting again resumes the study: one more trial, the first two kept.
        assert self._start(n_trials=3).status_code == 200
        resumed = self._campaign()
        assert resumed["status"] == "done"
        assert [s["index"] for s in resumed["settings"]] == [0, 1, 2]

    # ── rejection paths, cheap ────────────────────────────────────────────
    def test_unknown_model_is_404(self: Self) -> None:
        res = self.client.post(f"/jobs/{self.job_id}/pet/start", params={"model_name": "nope"},
                               json={}, headers=self.auth)
        assert res.status_code == 404

    def test_traversal_in_model_name_is_400(self: Self) -> None:
        res = self.client.get(f"/jobs/{self.job_id}/pet/campaign", params={"model_name": "../x"}, headers=self.auth)
        assert res.status_code == 400

    def test_running_runner_blocks_a_second_start(self: Self) -> None:
        out = self.pet_runner.pet_dir(self.backend._job_dir(self.job_id), MODEL)
        self.pet_runner.write_status(out, status="running", model_name=MODEL)  # pid = this process
        assert self._start().status_code == 409
        assert self.calls == []

    def test_vanished_runner_reads_as_failed_and_can_restart(self: Self) -> None:
        out = self.pet_runner.pet_dir(self.backend._job_dir(self.job_id), MODEL)
        out.mkdir(parents=True)
        (out / "status.json").write_text(json.dumps({"status": "running", "model_name": MODEL, "pid": 2**22 + 7}))
        run = self._campaign()
        assert run["status"] == "failed"
        assert "resume" in run["error"]
        assert self._start().status_code == 200  # not blocked by the stale heartbeat

    def test_verify_without_results_is_404(self: Self) -> None:
        res = self.client.post(f"/jobs/{self.job_id}/pet/verify", params={"model_name": MODEL},
                               json={"index": 0}, headers=self.auth)
        assert res.status_code == 404
        res = self.client.get(f"/jobs/{self.job_id}/pet/verify",
                              params={"model_name": MODEL, "index": 0}, headers=self.auth)
        assert res.status_code == 404

    def test_unauthenticated_is_401(self: Self) -> None:
        assert self.client.get(f"/jobs/{self.job_id}/pet/campaign", params={"model_name": MODEL}).status_code == 401

    def test_json_safe_replaces_infinity(self: Self) -> None:
        """A non-private epsilon is inf; browsers reject bare Infinity in JSON."""
        assert self.pet_runner.json_safe({"epsilon": float("inf"), "x": [np.float64(1.5)]}) == {"epsilon": None, "x": [1.5]}


def test_module_runs_as_a_script() -> None:
    """The spawner invokes the runner with -m; it must parse its own argv."""
    from leakpro.webapp.backend import pet_runner

    assert pet_runner.main.__doc__
    assert os.path.exists(pet_runner.__file__)
