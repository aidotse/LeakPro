#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the webapp's safe dataset formats (.npz / Parquet / CSV).

Safe-format uploads must be converted into a server-generated dataset object
without ever unpickling the uploaded bytes, and a safe-format file smuggling
pickled objects must be rejected rather than falling back to a pickle load.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="webapp extras (fastapi) not installed")

from fastapi import HTTPException  # noqa: E402

from leakpro.utils.import_helper import Self


class TestConvertUpload:
    """convert_upload turns safe formats into a server-generated dataset."""

    def test_npz_roundtrip(self: Self) -> None:
        """An .npz with data + targets becomes a loadable ArrayDataset."""
        import joblib

        from leakpro.webapp.backend.dataio import CONVERTED_NAME, convert_upload

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "up.npz"
            np.savez(src, data=np.random.rand(10, 3, 8, 8).astype(np.float32),
                     targets=np.arange(10) % 2)
            result = convert_upload(src, Path(tmp))
            assert result is not None
            path, meta = result
            assert path.name == CONVERTED_NAME
            assert meta.n_samples == 10
            assert meta.shape == [3, 8, 8]
            assert meta.data_type == "image"
            assert meta.n_classes == 2

            ds = joblib.load(path)
            assert len(ds) == 10
            x, y = ds[0]
            assert tuple(x.shape) == (3, 8, 8)
            assert int(y) in (0, 1)
            # Population interface used by LeakPro core
            assert ds.data[np.array([0, 1])].shape[0] == 2
            assert len(ds.targets) == 10

    def test_parquet_keeps_feature_names(self: Self) -> None:
        """Tabular columns survive as feature_names; label column is detected."""
        import joblib
        import pandas as pd

        from leakpro.webapp.backend.dataio import convert_upload

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "up.parquet"
            pd.DataFrame({"age": [1.0, 2.0], "bmi": [3.0, 4.0], "label": [0, 1]}).to_parquet(src)
            path, meta = convert_upload(src, Path(tmp))
            assert meta.data_type == "tabular"
            assert meta.shape == [2]
            ds = joblib.load(path)
            assert ds.feature_names == ["age", "bmi"]

    def test_npz_with_pickled_objects_rejected(self: Self) -> None:
        """Object arrays (pickle inside .npz) are refused, never pickle-loaded."""
        from leakpro.webapp.backend.dataio import convert_upload

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "up.npz"
            hostile = np.array([{"a": 1}], dtype=object)
            np.savez(src, data=hostile, targets=np.zeros(1))
            with pytest.raises(HTTPException) as exc:
                convert_upload(src, Path(tmp))
            assert exc.value.status_code == 400

    def test_npz_missing_targets_rejected(self: Self) -> None:
        """A clear 400, not a silent zero-label dataset, for missing targets."""
        from leakpro.webapp.backend.dataio import convert_upload

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "up.npz"
            np.savez(src, data=np.random.rand(4, 2))
            with pytest.raises(HTTPException) as exc:
                convert_upload(src, Path(tmp))
            assert exc.value.status_code == 400

    def test_corrupt_uploads_rejected_with_400(self: Self) -> None:
        """Truncated/empty/malformed files get a clean 400, never a raw error.

        These raise non-ValueError exceptions from the underlying readers
        (zipfile.BadZipFile for a truncated .npz, EOFError for an empty .npy,
        pandas parse errors for a bad table), and upload_data() has no
        surrounding try/except — so anything unhandled here is a 500 on the
        primary upload flow.
        """
        from leakpro.webapp.backend.dataio import convert_upload

        with tempfile.TemporaryDirectory() as tmp:
            # Truncated .npz (broken zip structure)
            whole = Path(tmp) / "ok.npz"
            np.savez(whole, data=np.random.rand(4, 2), targets=np.zeros(4))
            truncated = Path(tmp) / "trunc.npz"
            truncated.write_bytes(whole.read_bytes()[:40])
            # Empty .npy
            empty = Path(tmp) / "empty.npy"
            empty.write_bytes(b"")
            # Malformed parquet
            bad_parquet = Path(tmp) / "bad.parquet"
            bad_parquet.write_bytes(b"not parquet at all")

            for src in (truncated, empty, bad_parquet):
                with pytest.raises(HTTPException) as exc:
                    convert_upload(src, Path(tmp))
                assert exc.value.status_code == 400, src.name

    def test_unknown_format_falls_through(self: Self) -> None:
        """Legacy formats return None so callers use the existing loaders."""
        from leakpro.webapp.backend.dataio import convert_upload

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "up.pkl"
            src.write_bytes(b"x")
            assert convert_upload(src, Path(tmp)) is None


class TestUploadEndpoint:
    """The upload endpoint converts safe formats at the trust boundary."""

    @pytest.fixture(autouse=True)
    def _isolated_jobs_root(self: Self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Keep test jobs out of the real webapp_jobs/ (and out of _load_jobs)."""
        from leakpro.webapp.backend import main as backend_main

        monkeypatch.setattr(backend_main, "JOBS_ROOT", tmp_path)

    def _client(self: Self) -> tuple:
        from fastapi.testclient import TestClient

        from leakpro.webapp.backend import security
        from leakpro.webapp.backend.main import app

        return TestClient(app), {"Authorization": f"Bearer {security.API_TOKEN}"}

    def test_npz_upload_converts_and_serves_samples(self: Self) -> None:
        """An .npz upload yields meta, a converted data_path, working samples."""
        import io

        client, auth = self._client()
        job_id = client.post("/jobs", headers=auth).json()["job_id"]

        buf = io.BytesIO()
        np.savez(buf, data=np.random.rand(6, 4).astype(np.float32),
                 targets=np.array([0, 1, 0, 1, 0, 1]))
        buf.seek(0)
        res = client.post(f"/jobs/{job_id}/upload/data",
                          files={"file": ("d.npz", buf, "application/octet-stream")},
                          headers=auth)
        assert res.status_code == 200, res.text
        meta = res.json()
        assert meta["n_samples"] == 6
        assert meta["data_type"] == "tabular"

        sample = client.get(f"/jobs/{job_id}/sample_data/0", headers=auth)
        assert sample.status_code == 200, sample.text
        assert len(sample.json()["features"]) == 4

    def test_truncated_npz_upload_rejected(self: Self) -> None:
        """A corrupt upload gets 400 through the endpoint, not an unhandled 500."""
        import io

        client, auth = self._client()
        job_id = client.post("/jobs", headers=auth).json()["job_id"]

        buf = io.BytesIO()
        np.savez(buf, data=np.random.rand(4, 2), targets=np.zeros(4))
        res = client.post(f"/jobs/{job_id}/upload/data",
                          files={"file": ("d.npz", buf.getvalue()[:40], "application/octet-stream")},
                          headers=auth)
        assert res.status_code == 400, res.text

    def test_hostile_npz_upload_rejected(self: Self) -> None:
        """A pickled-object .npz is rejected with 400 at upload time."""
        import io

        client, auth = self._client()
        job_id = client.post("/jobs", headers=auth).json()["job_id"]

        buf = io.BytesIO()
        np.savez(buf, data=np.array([{"a": 1}], dtype=object), targets=np.zeros(1))
        buf.seek(0)
        res = client.post(f"/jobs/{job_id}/upload/data",
                          files={"file": ("d.npz", buf, "application/octet-stream")},
                          headers=auth)
        assert res.status_code == 400
