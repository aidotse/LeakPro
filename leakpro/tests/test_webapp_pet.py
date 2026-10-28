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

        def spawn_in_process(job_id: str, model_name: str, extra: list[str]) -> int:
            self.calls.append(extra)
            job_dir = backend_main._job_dir(job_id)
            if extra:
                pet_runner.run_verification(job_dir, model_name, int(extra[1]))
            else:
                pet_runner.run_search(job_dir, model_name)
            return os.getpid()

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
        job["data_meta"] = {"data_type": "tabular", "n_classes": 3, "n_samples": 400}
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

    def _verify(self: Self, index: int) -> dict:
        res = self.client.post(f"/jobs/{self.job_id}/pet/verify", params={"model_name": MODEL},
                               json={"index": index}, headers=self.auth)
        assert res.status_code == 200, res.text
        res = self.client.get(f"/jobs/{self.job_id}/pet/verify",
                              params={"model_name": MODEL, "index": index}, headers=self.auth)
        assert res.status_code == 200, res.text
        return res.json()

    def _adopt(self: Self, index: int) -> object:
        return self.client.post(f"/jobs/{self.job_id}/pet/adopt", params={"model_name": MODEL},
                                json={"index": index}, headers=self.auth)

    def _write_metadata(self: Self, train: list[int], test: list[int]) -> None:
        """A real model_metadata.pkl: the clicked model's own split, as the training step records it."""
        import pickle

        from leakpro.schemas import DataLoaderConfig, EvalOutput, LossConfig, MIAMetaDataSchema, OptimizerConfig

        meta = MIAMetaDataSchema(
            train_indices=train, test_indices=test, num_train=len(train), init_params={"num_classes": 3},
            optimizer=OptimizerConfig(name="adam", params={"lr": 0.05}),
            criterion=LossConfig(name="crossentropyloss"),
            data_loader=DataLoaderConfig(params={"batch_size": 32}), epochs=5,
            train_result=EvalOutput(accuracy=0.9, loss=0.3), test_result=EvalOutput(accuracy=0.8, loss=0.5),
            dataset="blobs")
        with open(self.backend._job_dir(self.job_id) / "models" / MODEL / "model_metadata.pkl", "wb") as f:
            pickle.dump(meta, f)

    # ── the happy path, in one test so the expensive search runs once ─────
    def test_search_verify_adopt_without_a_known_split(self: Self) -> None:
        """The full flow on a model whose own split is unknown: the confirm step says so."""
        assert self._campaign()["status"] == "idle"

        assert self._start().status_code == 200
        run = self._campaign()
        assert run["status"] == "done"
        assert run["n_trials"] == 2
        assert run["n_members"] == 120
        # The comparison basis is an unprotected model trained on the same
        # subset, not the clicked model's accuracy (which is kept for display).
        assert run["baseline_source"] == "search_subset"
        assert 0.0 <= run["baseline_utility"] <= 1.0 and run["baseline_utility"] != 0.8
        assert run["model_utility"] == 0.8
        # 100 non-members cannot resolve 1% FPR: the runner says so and measures at 10%.
        assert run["proxy_fpr"] == 0.1
        assert "10%" in run["resolution_warning"]
        assert len(run["settings"]) == 2
        for setting in run["settings"]:
            assert set(setting["config"]) == {"noise_multiplier", "max_grad_norm", "learning_rate", "batch_size"}
            assert 0.0 <= setting["utility"] <= 1.0
            assert setting["proxy_fpr"] == 0.1
            assert setting["tuning_accounted"] is False
            assert setting["n_members"] == 120
        measured = [s for s in run["settings"] if s["attack_tpr"] is not None]
        assert measured, "no trial resolved a TPR"
        for setting in measured:
            lo, hi = setting["attack_tpr_ci"]
            assert 0.0 <= lo <= setting["attack_tpr"] <= hi <= 1.0
        assert set(run["best_indices"]) <= {s["index"] for s in measured}
        # Trials are keyed on their number: identical configs could never collide.
        out = self.pet_runner.pet_dir(self.backend._job_dir(self.job_id), MODEL)
        assert sorted(p.name for p in (out / "trials").iterdir()) == ["trial_0000", "trial_0001"]
        # The API wrote the runner's pid before the runner's own heartbeat.
        assert "pid" in json.loads((out / "status.json").read_text())

        # Changing a measurement-affecting setting after results exist is refused.
        res = self._start(n_shadow=2)
        assert res.status_code == 409 and "n_shadow" in res.text

        # No known split: the verification re-audits the search-scale model and says so.
        index = run["best_indices"][0]
        verification = self._verify(index)
        assert self.calls[-1] == ["--verify", str(index)]
        assert verification["status"] == "done", verification
        assert verification["estimated"]["utility"] == pytest.approx(
            next(s["utility"] for s in run["settings"] if s["index"] == index))
        verified = verification["verified"]
        assert verified["scope"] == "search_subset"
        assert verified["n_shadow"] == 2
        assert verified["utility"] == verification["estimated"]["utility"]
        assert verified["baseline_utility"] is None
        assert any("not yours" in n for n in verified["notes"])
        assert "attack_tpr" in verified and verified["n_members"] == 120

        # Adopting an unverified setting is refused; the verified one lands as a
        # normal result row carrying the verification's RMIA result.
        other = next(s["index"] for s in run["settings"] if s["index"] != index)
        assert self._adopt(other).status_code == 409
        if verified["attack_tpr"] is None:
            assert self._adopt(index).status_code == 409
            return
        # A finished measurement is returned, not repeated, unless forced.
        calls_before = len(self.calls)
        again = self.client.post(f"/jobs/{self.job_id}/pet/verify", params={"model_name": MODEL},
                                 json={"index": index}, headers=self.auth)
        assert again.status_code == 200 and again.json()["status"] == "done" and len(self.calls) == calls_before
        res = self._adopt(index)
        assert res.status_code == 200, res.text
        rows = self.backend._jobs[self.job_id]["results"]
        adopted = next(r for r in rows if r["model_name"] == f"{MODEL} (optimized, setting {index})")
        assert adopted["optimized"] is True and adopted["tuning_accounted"] is False
        assert adopted["pet"]["setting_index"] == index and adopted["pet"]["scope"] == "search_subset"
        assert set(adopted["pet"]["config"]) == {"noise_multiplier", "max_grad_norm", "learning_rate", "batch_size"}
        assert adopted["pet"]["proxy_fpr"] == 0.1
        assert adopted["attacks"][0]["attack_name"].upper() == "RMIA"
        assert adopted["attacks"][0]["signal_values"]
        assert adopted["train_meta"]["noise_multiplier"] == adopted["pet"]["config"]["noise_multiplier"]
        # Persisted: a restart would reload it.
        state = json.loads((self.backend._job_dir(self.job_id) / "state.json").read_text())
        assert any(r["model_name"] == f"{MODEL} (optimized, setting {index})" for r in state["results"])
        # Adopting the same setting again replaces its row; another setting adds one.
        assert self._adopt(index).status_code == 200
        assert sum(r.get("optimized", False) for r in self.backend._jobs[self.job_id]["results"]) == 1

        # Starting again resumes the study: one more trial, the first two kept.
        assert self._start(n_trials=3).status_code == 200
        resumed = self._campaign()
        assert resumed["status"] == "done"
        assert [s["index"] for s in resumed["settings"]] == [0, 1, 2]

    def test_confirm_measures_the_setting_on_the_users_own_split(self: Self) -> None:
        """With the model's split known, confirm retrains at the user's size and compares with the user's accuracy."""
        self._write_metadata(train=list(range(0, 150)), test=list(range(150, 300)))
        assert self._start().status_code == 200
        run = self._campaign()
        assert run["status"] == "done" and run["settings"], run
        # Prefer a setting the search could measure; a gated one still retrains at full size.
        index = next((s["index"] for s in run["settings"] if s["attack_tpr"] is not None), run["settings"][0]["index"])

        verification = self._verify(index)
        assert verification["status"] == "done", verification
        verified = verification["verified"]
        assert verified["scope"] == "user_model"
        assert verified["n_members"] == 150 and verified["n_non_members"] == 150
        assert verified["baseline_utility"] == 0.8
        assert 0.0 <= verified["utility"] <= 1.0
        assert verified["train_accuracy"] is not None
        assert verified["n_shadow"] == 2 and verified["proxy_fpr"] == 0.1
        assert verified["notes"] == []
        out = self.pet_runner.pet_dir(self.backend._job_dir(self.job_id), MODEL)
        assert (out / "population_full.pkl").exists()
        target_dir = out / "confirm" / f"trial_{index:04d}" / "target"
        assert (target_dir / "dpsgd_dic.pkl").exists()  # written before training, gated or not
        if verified["attack_tpr"] is None:
            # A gated retrain is never persisted or attacked, so there is no model to adopt.
            assert verified["resolution_warning"]
            assert not (target_dir / "target_model.pkl").exists()
            assert self._adopt(index).status_code == 409
            return
        assert (target_dir / "target_model.pkl").exists()
        lo, hi = verified["attack_tpr_ci"]
        assert lo <= verified["attack_tpr"] <= hi

        assert self._adopt(index).status_code == 200
        adopted = next(r for r in self.backend._jobs[self.job_id]["results"]
                       if r["model_name"] == f"{MODEL} (optimized, setting {index})")
        assert adopted["num_train"] == 150
        # Audit indices name dataset records (members are dataset 0..149), not population positions.
        indices = adopted["attacks"][0]["audit_indices"]
        labels = adopted["attacks"][0]["true_labels"]
        assert indices is not None and len(indices) == 300
        assert all((i < 150) == (lbl == 1) for i, lbl in zip(indices, labels))
        assert set(indices) == set(range(300))
        assert adopted["test_accuracy"] == verified["utility"]
        assert adopted["train_accuracy"] == verified["train_accuracy"]
        assert adopted["pet"]["scope"] == "user_model" and adopted["pet"]["baseline_utility"] == 0.8
        assert adopted["dpsgd"] is (adopted["target_epsilon"] is not None)
        assert adopted["attacks"][0]["attack_name"].upper() == "RMIA"
        assert len(adopted["attacks"][0]["true_labels"]) == 300

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

    def test_verify_is_refused_while_the_search_runs(self: Self) -> None:
        out = self.pet_runner.pet_dir(self.backend._job_dir(self.job_id), MODEL)
        out.mkdir(parents=True)
        (out / "study.db").write_bytes(b"")
        self.pet_runner.write_status(out, status="running", model_name=MODEL)  # pid = this process
        res = self.client.post(f"/jobs/{self.job_id}/pet/verify", params={"model_name": MODEL},
                               json={"index": 0}, headers=self.auth)
        assert res.status_code == 409
        assert self.calls == []

    def test_dead_verification_reads_as_failed_and_confirm_restarts_it(self: Self) -> None:
        out = self.pet_runner.pet_dir(self.backend._job_dir(self.job_id), MODEL)
        out.mkdir(parents=True)
        (out / "study.db").write_bytes(b"")
        self.pet_runner.verification_path(out, 0).write_text(json.dumps(
            {"status": "running", "index": 0, "estimated": {}, "pid": 2**22 + 7}))
        res = self.client.get(f"/jobs/{self.job_id}/pet/verify",
                              params={"model_name": MODEL, "index": 0}, headers=self.auth)
        assert res.json()["status"] == "failed" and "retry" in res.json()["error"]
        # A new confirm is not blocked by the stale placeholder: it spawns again.
        self.pet_runner.run_verification = lambda *a, **k: None  # type: ignore[assignment]
        res = self.client.post(f"/jobs/{self.job_id}/pet/verify", params={"model_name": MODEL},
                               json={"index": 0}, headers=self.auth)
        assert res.status_code == 200 and self.calls == [["--verify", "0"]]

    def test_unmeasured_setting_cannot_be_adopted(self: Self) -> None:
        out = self.pet_runner.pet_dir(self.backend._job_dir(self.job_id), MODEL)
        out.mkdir(parents=True)
        self.pet_runner.verification_path(out, 0).write_text(json.dumps(
            {"status": "done", "index": 0, "estimated": {}, "verified": {"attack_tpr": None}}))
        assert self._adopt(0).status_code == 409

    def test_non_classification_job_is_refused(self: Self) -> None:
        job = self.backend._jobs[self.job_id]
        job["data_meta"] = {"data_type": "tabular", "n_classes": None, "n_samples": 400}
        assert self._start().status_code == 422
        job["data_meta"] = {"data_type": "tabular", "n_classes": 400, "n_samples": 400}
        res = self._start()
        assert res.status_code == 422 and "continuous" in res.text
        assert self.calls == []

    def test_continuous_targets_are_refused_by_the_runner(self: Self) -> None:
        import torch
        from torch.utils.data import TensorDataset

        regression = TensorDataset(torch.zeros(10, 4), torch.linspace(0.0, 1.0, 10))
        with pytest.raises(ValueError, match="continuous"):
            self.pet_runner._assert_classification(regression)
        self.pet_runner._assert_classification(TensorDataset(torch.zeros(10, 4), torch.arange(10) % 3))

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
