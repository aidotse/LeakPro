"""Dataset shape/type auto-detection for Step 1 → Step 2."""

from __future__ import annotations

import pickle
import io
from pathlib import Path
from typing import Any

import numpy as np

from .models import DataMeta


class _SafeUnpickler(pickle.Unpickler):
    """Unpickler that returns a placeholder for unknown classes instead of raising."""
    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            # Return a dummy class that accepts any args
            return type(name, (), {"__reduce__": lambda self: (type(self), ())})


def _try_load(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=True)
    if suffix in {".pkl", ".pickle"}:
        # Add common LeakPro handler directories to sys.path so custom
        # classes (e.g. cifar_handler.UserDataset) can be unpickled.
        import sys
        _leakpro_root = Path(__file__).parents[3]
        _extra_paths = [
            str(_leakpro_root / "examples" / "mia" / "cifar"),
            str(_leakpro_root / "examples" / "mia" / "tabular_mia"),
            str(_leakpro_root),
        ]
        for p in _extra_paths:
            if p not in sys.path:
                sys.path.insert(0, p)
        try:
            with open(path, "rb") as f:
                return pickle.load(f)  # noqa: S301
        except Exception:
            # Fall back to safe unpickler if import still fails
            with open(path, "rb") as f:
                return _SafeUnpickler(f).load()  # noqa: S301
    if suffix == ".pt":
        import torch
        return torch.load(path, map_location="cpu", weights_only=False)
    if suffix == ".csv":
        import pandas as pd
        return pd.read_csv(path)
    if suffix in {".jsonl", ".json"}:
        import pandas as pd
        return pd.read_json(path, lines=(suffix == ".jsonl"))
    if suffix == ".parquet":
        import pandas as pd
        return pd.read_parquet(path)
    return None


def _shape_from_array(arr: Any) -> tuple[list[int], str]:
    if hasattr(arr, "shape"):
        return list(arr.shape[1:] if len(arr.shape) > 1 else arr.shape), str(arr.dtype)
    return [], "unknown"


def inspect(data_path: Path) -> DataMeta:
    """Inspect an uploaded dataset file and return auto-detected metadata."""
    obj = _try_load(data_path)
    if obj is None:
        return DataMeta(data_type="unknown", shape=[], n_samples=0, dtype="unknown")

    # --- numpy / torch tensor -------------------------------------------------
    import torch
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        arr = np.array(obj) if isinstance(obj, torch.Tensor) else obj
        n_samples = arr.shape[0]
        item_shape = list(arr.shape[1:])
        dtype = str(arr.dtype)
        data_type = _guess_type(item_shape)
        return DataMeta(
            data_type=data_type,
            shape=item_shape,
            n_samples=n_samples,
            dtype=dtype,
        )

    # --- dict (common LeakPro / PyTorch checkpoint format) --------------------
    if isinstance(obj, dict):
        # Try common keys first
        for key in ("data", "images", "x", "X", "features"):
            if key in obj and isinstance(obj[key], (np.ndarray,)):
                return inspect_array(obj[key])
        try:
            import torch
            for key in ("data", "images", "x", "X", "features"):
                if key in obj and isinstance(obj[key], torch.Tensor):
                    return inspect_array(obj[key])
        except ImportError:
            pass
        # Recursively search all values for the first large numpy/tensor array
        arr = _find_array(obj)
        if arr is not None:
            return inspect_array(arr)
        return DataMeta(data_type="unknown", shape=[], n_samples=len(obj), dtype="dict")

    # --- pandas DataFrame (CSV / Parquet / JSONL) -----------------------------
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            n_samples = len(obj)
            n_features = len(obj.columns) - 1  # assume last col is label
            dtype = str(obj.dtypes.iloc[0])
            # Try to detect a label column
            label_col = None
            for c in ("label", "target", "y", "class"):
                if c in obj.columns:
                    label_col = c
                    break
            n_classes = int(obj[label_col].nunique()) if label_col else None
            return DataMeta(
                data_type="tabular",
                shape=[n_features],
                n_samples=n_samples,
                n_classes=n_classes,
                dtype=dtype,
                label_column=label_col,
            )
    except ImportError:
        pass

    # --- arbitrary object (UserDataset, custom class, etc.) -------------------
    arr = _find_array(obj)
    if arr is not None:
        return inspect_array(arr)

    return DataMeta(data_type="unknown", shape=[], n_samples=0, dtype="unknown")


def inspect_array(arr: Any) -> DataMeta:
    import torch
    if isinstance(arr, torch.Tensor):
        arr = arr.numpy()
    if isinstance(arr, np.ndarray):
        n_samples = arr.shape[0]
        item_shape = list(arr.shape[1:])
        dtype = str(arr.dtype)
        return DataMeta(
            data_type=_guess_type(item_shape),
            shape=item_shape,
            n_samples=n_samples,
            dtype=dtype,
        )
    return DataMeta(data_type="unknown", shape=[], n_samples=0, dtype="unknown")


def _find_array(obj: Any, _depth: int = 0) -> Any:
    """Recursively search an object for the largest numpy array or torch tensor."""
    if _depth > 5:
        return None
    try:
        import torch
        tensor_types = (np.ndarray, torch.Tensor)
    except ImportError:
        tensor_types = (np.ndarray,)

    if isinstance(obj, tensor_types) and hasattr(obj, "shape") and len(obj.shape) >= 1:
        return obj
    if isinstance(obj, dict):
        best = None
        for v in obj.values():
            candidate = _find_array(v, _depth + 1)
            if candidate is not None:
                if best is None or candidate.shape[0] > best.shape[0]:
                    best = candidate
        return best
    if isinstance(obj, (list, tuple)) and len(obj) > 0:
        for item in obj:
            result = _find_array(item, _depth + 1)
            if result is not None:
                return result
    # Check object attributes (handles custom classes loaded as placeholders)
    if hasattr(obj, "__dict__"):
        return _find_array(vars(obj), _depth + 1)
    return None


def _guess_type(shape: list[int]) -> str:
    if len(shape) == 3 and shape[0] in {1, 3, 4}:      # channel-first (C, H, W)
        return "image"
    if len(shape) == 3 and shape[-1] in {1, 3, 4}:     # channel-last (H, W, C)
        return "image"
    if len(shape) == 2 and shape[0] in {1, 3, 4}:
        return "image"
    # CIFAR-style flat images: (3072,) = 3*32*32 or (3*H*W,)
    if len(shape) == 1 and shape[0] in {3072, 1024, 12288}:
        return "image"
    if len(shape) == 2:
        return "time_series"
    if len(shape) == 1:
        return "tabular"
    return "unknown"
