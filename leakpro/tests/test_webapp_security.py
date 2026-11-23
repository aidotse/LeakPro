#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Regression tests for the webapp backend security controls.

These cover the vulnerabilities fixed in the webapp hardening pass:
unauthenticated access, cross-site requests, path traversal through
``model_name``, unconfined server-side paths, and code execution during
metadata deserialization.
"""

import contextlib
import os
import pickle
import tempfile
from pathlib import Path

import pytest

# Webapp tests: run wherever the webapp extras are installed
# (pip install -e ".[webapp]"); skipped elsewhere.
pytest.importorskip("fastapi", reason="webapp extras (fastapi) not installed")

from fastapi import HTTPException  # noqa: E402

from leakpro.utils.import_helper import Self


class TestPathValidation:
    """safe_name and confine must reject traversal and absolute-path escapes."""

    def test_safe_name_rejects_escapes(self: Self) -> None:
        """Names that would escape the job directory are refused."""
        from leakpro.webapp.backend.security import safe_name

        # `Path("/a/b") / "/etc"` is `/etc`, so an absolute value is an escape.
        for bad in ["../../etc", "/etc/cron.d", "..", ".", "a/b", "a\\b", "", ".hidden", "x" * 65]:
            with pytest.raises(HTTPException):
                safe_name(bad, "model_name")

    def test_safe_name_accepts_ordinary_names(self: Self) -> None:
        """Names the UI actually produces are unaffected."""
        from leakpro.webapp.backend.security import safe_name

        for good in ["model1", "my_model", "resnet-18", "a.b", "uploaded_model"]:
            assert safe_name(good, "model_name") == good

    def test_confine_blocks_paths_outside_roots(self: Self) -> None:
        """A server-side path outside the permitted roots is refused."""
        from leakpro.webapp.backend.security import confine

        with tempfile.TemporaryDirectory() as root:
            inside = Path(root) / "data.pkl"
            inside.write_bytes(b"x")
            assert confine(str(inside), [Path(root)]) == inside.resolve()

            with pytest.raises(HTTPException):
                confine("/etc/passwd", [Path(root)])
            with pytest.raises(HTTPException):
                confine(f"{root}/../../etc/passwd", [Path(root)])

    def test_safe_suffix_ignores_hostile_filenames(self: Self) -> None:
        """Only a plain extension is ever taken from a client filename."""
        from leakpro.webapp.backend.security import safe_suffix

        assert safe_suffix("data.pkl") == ".pkl"
        assert safe_suffix("../../evil") == ""
        assert safe_suffix(None) == ""

    def test_trailing_newline_rejected(self: Self) -> None:
        """`$` under match() accepts a trailing newline; fullmatch must not."""
        from leakpro.webapp.backend.security import safe_name, safe_suffix

        with pytest.raises(HTTPException):
            safe_name("model1\n")
        assert safe_suffix("x.pkl\n") == ""

    def test_non_ascii_token_is_invalid_not_error(self: Self) -> None:
        """compare_digest raises TypeError on non-ASCII str; must return False."""
        from leakpro.webapp.backend.security import token_is_valid

        assert token_is_valid("\u00fc") is False


class _Canary:
    """Pickle payload that would run a command on a vulnerable loader."""

    def __init__(self: Self, marker: str) -> None:
        self.marker = marker

    def __reduce__(self: Self) -> tuple:
        """Return the os.system call a vulnerable unpickler would execute."""
        return (os.system, (f"touch {self.marker}",))


class _FakeMetadata:
    """Stand-in for a real metadata object of a non-allowlisted class."""

    def __init__(self: Self) -> None:
        self.train_indices = [1, 2]
        self.optimizer = "adam"
        self.epochs = 5


class TestRestrictedUnpickler:
    """The restricted unpickler must never resolve an attacker-chosen callable."""

    def test_does_not_execute_payload(self: Self) -> None:
        """A malicious pickle is inert under RestrictedUnpickler."""
        from leakpro.webapp.backend.security import RestrictedUnpickler

        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "EXECUTED"
            payload = Path(tmp) / "payload.pkl"
            payload.write_bytes(pickle.dumps(_Canary(str(marker))))

            # Sanity check: the payload really is dangerous to a plain loader.
            with open(payload, "rb") as handle:
                pickle.load(handle)  # noqa: S301
            assert marker.exists(), "payload should execute under plain pickle.load"
            marker.unlink()

            # Refusing to load (raising) is an acceptable outcome; executing is not.
            with open(payload, "rb") as handle, contextlib.suppress(Exception):
                RestrictedUnpickler(handle).load()
            assert not marker.exists(), "RestrictedUnpickler must not execute the payload"

    def test_preserves_field_names(self: Self) -> None:
        """Metadata validation still sees which fields a pickle contains."""
        from leakpro.webapp.backend.security import RestrictedUnpickler

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta.pkl"
            path.write_bytes(pickle.dumps({"epochs": 5, "train_indices": [1, 2]}))
            with open(path, "rb") as handle:
                loaded = RestrictedUnpickler(handle).load()
            assert set(loaded) == {"epochs", "train_indices"}

    def test_preserves_field_names_of_custom_class(self: Self) -> None:
        """A non-allowlisted class becomes a stub whose __dict__ keeps the fields.

        This is the case validate_model_metadata actually depends on: real
        metadata is a ModelMetadata *object*, and its field names survive only
        through the stub's __setstate__.
        """
        from leakpro.webapp.backend.security import RestrictedUnpickler

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta.pkl"
            path.write_bytes(pickle.dumps(_FakeMetadata()))
            with open(path, "rb") as handle:
                loaded = RestrictedUnpickler(handle).load()
            assert type(loaded).__name__ == "Stub__FakeMetadata"
            assert set(loaded.__dict__) == {"train_indices", "optimizer", "epochs"}


class TestApiBoundary:
    """Every route except the static SPA requires a valid token and origin."""

    @pytest.fixture(autouse=True)
    def _isolated_jobs_root(self: Self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Keep test jobs out of the real webapp_jobs/ (and out of _load_jobs)."""
        from leakpro.webapp.backend import main as backend_main

        monkeypatch.setattr(backend_main, "JOBS_ROOT", tmp_path)
        self.jobs_root = tmp_path

    def _client(self: Self) -> tuple:
        from fastapi.testclient import TestClient

        from leakpro.webapp.backend import security
        from leakpro.webapp.backend.main import app

        return TestClient(app), {"Authorization": f"Bearer {security.API_TOKEN}"}

    def test_unauthenticated_requests_rejected(self: Self) -> None:
        """No token means no access to the API surface."""
        client, _ = self._client()
        assert client.get("/jobs").status_code == 401
        assert client.post("/jobs").status_code == 401
        assert client.post("/jobs", headers={"Authorization": "Bearer nope"}).status_code == 401

    def test_cross_site_origin_rejected(self: Self) -> None:
        """A valid token from a foreign origin is still refused."""
        client, auth = self._client()
        res = client.post("/jobs", headers={**auth, "Origin": "https://evil.example"})
        assert res.status_code == 403

    def test_traversal_in_model_name_rejected(self: Self) -> None:
        """model_name cannot redirect an upload outside the job directory."""
        client, auth = self._client()
        job_id = client.post("/jobs", headers=auth).json()["job_id"]

        escape_target = self.jobs_root.parent / "pwned_abs"
        for hostile in ["../../../../escaped/pwned", str(escape_target)]:
            res = client.post(
                f"/jobs/{job_id}/upload/weights",
                params={"model_name": hostile},
                files={"file": ("w.pkl", b"data", "application/octet-stream")},
                headers=auth,
            )
            assert res.status_code == 400
        assert not escape_target.exists()

    def test_query_token_not_accepted_on_http(self: Self) -> None:
        """?token= must not authenticate HTTP routes — it lands in access logs."""
        from leakpro.webapp.backend import security

        client, _ = self._client()
        assert client.get("/jobs", params={"token": security.API_TOKEN}).status_code == 401

    def test_non_ascii_credentials_get_401_not_500(self: Self) -> None:
        """Garbage credentials must be rejected, not crash the auth path."""
        client, _ = self._client()
        assert client.get("/jobs", params={"token": "\u00fc"}).status_code == 401

    def test_static_spa_paths_stay_public(self: Self) -> None:
        """Fail-closed auth must still leave the SPA shell reachable."""
        client, _ = self._client()
        for path in ["/", "/index.html", "/assets/anything.js"]:
            assert client.get(path).status_code not in (401, 403), path
        # Anything else is protected by default
        assert client.get("/some/future/route").status_code == 401

    def test_server_side_path_confined(self: Self) -> None:
        """An arbitrary absolute path cannot be read through data-path."""
        client, auth = self._client()
        job_id = client.post("/jobs", headers=auth).json()["job_id"]
        res = client.post(f"/jobs/{job_id}/data-path", json={"path": "/etc/passwd"}, headers=auth)
        assert res.status_code == 400

    def test_legitimate_upload_still_works(self: Self) -> None:
        """The hardening does not break the normal flow."""
        client, auth = self._client()
        job_id = client.post("/jobs", headers=auth).json()["job_id"]
        res = client.post(
            f"/jobs/{job_id}/upload/weights",
            params={"model_name": "good_model"},
            files={"file": ("w.pkl", b"data", "application/octet-stream")},
            headers=auth,
        )
        assert res.status_code == 200
