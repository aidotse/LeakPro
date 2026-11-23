"""Data handler for LLM membership inference: pre-tokenised sequences, labels == input ids.

Usage:
    from llm_data_handler import LLMDataHandler
    population = LLMDataHandler.UserDataset(ids, ids, pad_token_id=50256)

``data`` is either a ``(N, T)`` int64 tensor (fixed-length chunks, the EZ-MIA paper's setup) or an
object-dtype NumPy array of 1-D int64 arrays (variable length). It must be an array, not a list:
``MIAHandler.get_dataset`` indexes it with a NumPy index array.

The tokenizer is deliberately *not* stored on the dataset. ``UserDataset.return_params()`` forwards
every non-data attribute into the population pickle and back through ``get_dataset``, so a tokenizer
object here would be pickled along with the population. Tokenise once in ``prepare_target.py``.
"""

import numpy as np

from leakpro.input_handler.abstract_input_handler import AbstractInputHandler


class LLMDataHandler(AbstractInputHandler, role="data"):
    """Provides UserDataset for tokenised text. No training logic."""

    class UserDataset(AbstractInputHandler.UserDataset):
        def __init__(self, data, targets, *, pad_token_id: int, **kwargs) -> None:
            assert len(data) == len(targets), "data and targets must have the same length"
            if isinstance(data, list):
                raise TypeError("data must be a tensor or an object-dtype ndarray, not a list (see module docstring)")
            self.data = data
            self.targets = targets
            self.pad_token_id = int(pad_token_id)
            for key, value in kwargs.items():
                setattr(self, key, value)

        def __getitem__(self, idx: int):  # noqa: ANN204
            return self.data[idx], self.targets[idx]

        def __len__(self) -> int:
            return len(self.targets)

    @staticmethod
    def as_object_array(sequences: list) -> np.ndarray:
        """Pack ragged int sequences into an object-dtype ndarray (variable-length populations)."""
        out = np.empty(len(sequences), dtype=object)
        for i, seq in enumerate(sequences):
            out[i] = np.asarray(seq, dtype=np.int64)
        return out
