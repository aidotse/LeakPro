"""Safe dataset formats for the webapp.

Pickled datasets execute arbitrary code when loaded — unavoidable for the
legacy ``.pkl`` path, but a model or dataset fetched from a model zoo or a
Kaggle page is exactly the kind of file a privacy-audit user wants to inspect
*before* trusting. This module gives them formats that structurally cannot
execute code:

- ``.npz`` (image / time-series / arrays): ``data`` + ``targets`` entries,
  loaded with ``allow_pickle=False``.
- ``.parquet`` / ``.csv`` / ``.json`` / ``.jsonl`` (tabular): column names
  become ``feature_names``; the label column is auto-detected.
- ``.npy``: a bare data array (labels default to zeros), ``allow_pickle=False``.

Uploads in these formats are converted once, at the trust boundary, into an
:class:`ArrayDataset` that the server pickles itself. Everything downstream —
the training path, the audit worker, LeakPro core's ``_load_population`` —
keeps consuming a pickle, but it is now a *server-generated* pickle of a known
class; the user-supplied bytes are never unpickled by anyone.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import torch
from fastapi import HTTPException

from .inspector import _guess_type
from .models import DataMeta

#: Formats that load without any code-execution risk.
SAFE_DATA_SUFFIXES = {".npz", ".npy", ".parquet", ".csv", ".json", ".jsonl"}

_DATA_KEYS = ("data", "x", "X", "features", "samples")
_TARGET_KEYS = ("targets", "y", "Y", "labels", "label")
_LABEL_COLUMNS = ("label", "target", "y", "class")

#: Filename of the server-generated dataset pickle inside a job directory.
CONVERTED_NAME = "data_converted.pkl"


class ArrayDataset(torch.utils.data.Dataset):
    """Array-backed dataset with the interface the rest of LeakPro expects.

    ``.data`` and ``.targets`` hold the *raw* arrays (so ``sample_data`` /
    ``sample_image`` and core's ``population.data[indices]`` see the original
    values, matching the semantics of the example ``UserDataset`` classes);
    ``__getitem__`` applies the same normalisation the training path applies:
    float32, ÷255 for raw pixel data, channel-last → channel-first.
    """

    def __init__(self, data: np.ndarray, targets: np.ndarray,
                 feature_names: Optional[list] = None, **_kwargs: Any) -> None:  # noqa: ANN401
        self.data = data
        self.targets = targets
        self.feature_names = feature_names

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        """Return one (input, label) pair as normalised tensors."""
        x = torch.as_tensor(np.asarray(self.data[idx]), dtype=torch.float32)
        if np.asarray(self.data[idx]).dtype == np.uint8 or x.max() > 1.0:
            x = x / 255.0
        if x.ndim == 3 and x.shape[-1] in (1, 3, 4) and x.shape[0] > 4:
            x = x.permute(2, 0, 1).contiguous()
        y = torch.as_tensor(self.targets[idx], dtype=torch.long)
        return x, y


def _pick(mapping: Any, keys: tuple) -> Optional[np.ndarray]:  # noqa: ANN401
    for key in keys:
        if key in mapping:
            return np.asarray(mapping[key])
    return None


def _from_npz(path: Path) -> tuple[np.ndarray, np.ndarray, None]:
    try:
        with np.load(path, allow_pickle=False) as npz:
            data = _pick(npz, _DATA_KEYS)
            targets = _pick(npz, _TARGET_KEYS)
            names = list(npz.files)
    except (ValueError, zipfile.BadZipFile, OSError, EOFError) as e:
        # ValueError: pickled/object arrays. BadZipFile/OSError/EOFError: a
        # truncated or otherwise corrupt upload, not necessarily malicious —
        # either way this boundary must reject with 400, never raise raw.
        raise HTTPException(
            status_code=400,
            detail="Could not read the .npz file (corrupt, or contains pickled objects).",
        ) from e
    if data is None:
        raise HTTPException(
            status_code=400,
            detail=f"No data array found in .npz (looked for {_DATA_KEYS}; file has {names}).",
        )
    if targets is None:
        raise HTTPException(
            status_code=400,
            detail=f"No target array found in .npz (looked for {_TARGET_KEYS}; file has {names}).",
        )
    if data.ndim == 0 or targets.ndim == 0:
        raise HTTPException(status_code=400,
                            detail="Expected arrays in the .npz, got a 0-dimensional scalar.")
    if len(targets) != len(data):
        raise HTTPException(status_code=400,
                            detail=f"data has {len(data)} rows but targets has {len(targets)}.")
    return data, targets, None


def _from_npy(path: Path) -> tuple[np.ndarray, np.ndarray, None]:
    try:
        data = np.load(path, allow_pickle=False)
    except (ValueError, OSError, EOFError) as e:
        raise HTTPException(
            status_code=400,
            detail="Could not read the .npy file (corrupt, empty, or contains pickled "
                   "objects); or use .npz with data + targets entries.",
        ) from e
    if data.ndim == 0:
        # np.load succeeds on a scalar array, but len() below would TypeError.
        raise HTTPException(status_code=400,
                            detail="Expected an array, got a scalar .npy file.")
    return data, np.zeros(len(data), dtype=np.int64), None


def _from_table(path: Path) -> tuple[np.ndarray, np.ndarray, list]:
    import pandas as pd
    suffix = path.suffix.lower()
    try:
        if suffix == ".parquet":
            df = pd.read_parquet(path)
        elif suffix == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_json(path, lines=(suffix == ".jsonl"))
    except Exception as e:  # noqa: BLE001 - pandas/pyarrow raise many types here
        raise HTTPException(status_code=400,
                            detail=f"Could not parse {suffix} file: {e}") from e

    if df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="File has no columns.")
    label_col = next((c for c in _LABEL_COLUMNS if c in df.columns), None)
    if label_col is None:
        label_col = df.columns[-1]
    features = df.drop(columns=[label_col])
    try:
        data = features.to_numpy(dtype=np.float32)
        targets = df[label_col].to_numpy()
    except (TypeError, ValueError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Non-numeric feature columns; encode them first ({e}).",
        ) from e
    return data, targets, list(features.columns)


def convert_upload(src: Path, out_dir: Path) -> Optional[tuple[Path, DataMeta]]:
    """Convert a safe-format dataset into a server-generated pickle.

    Returns ``(path, meta)`` for a recognised safe format, ``None`` for
    anything else (the caller falls back to the legacy loaders). Malformed
    safe-format files raise ``HTTPException(400)`` — they are never handed to
    a pickle-based fallback.
    """
    suffix = src.suffix.lower()
    if suffix not in SAFE_DATA_SUFFIXES:
        return None

    if suffix == ".npz":
        data, targets, feature_names = _from_npz(src)
    elif suffix == ".npy":
        data, targets, feature_names = _from_npy(src)
    else:
        data, targets, feature_names = _from_table(src)

    dataset = ArrayDataset(data=data, targets=targets, feature_names=feature_names)
    dest = out_dir / CONVERTED_NAME
    joblib.dump(dataset, dest)

    item_shape = list(data.shape[1:]) if data.ndim > 1 else [1]
    meta = DataMeta(
        data_type="tabular" if feature_names else _guess_type(item_shape),
        shape=item_shape,
        n_samples=int(data.shape[0]),
        n_classes=int(len(np.unique(targets))),
        dtype=str(data.dtype),
    )
    return dest, meta
