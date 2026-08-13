#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""XGBoost length-of-stay classifier, wrapped so LeakPro can audit it.

LeakPro's MIA machinery assumes every target is a ``torch.nn.Module``: shadow models are
persisted with ``state_dict()``/``load_state_dict()``, hashed by iterating state-dict tensors,
queried through ``PytorchModel`` (``.to(device)``, ``.eval()``, ``model(tensor)``), and handed to
a torch optimizer built from ``model.parameters()``.

An XGBoost booster satisfies none of that. ``XGBLOS`` is a *compatibility shim*: it is a real
``nn.Module`` as far as LeakPro can tell, but every call is forwarded to a booster held in a
byte buffer. Nothing here is meaningful PyTorch behaviour, and no gradients flow.

Consequences you must respect:

* Only logit- and loss-based attacks work (lira, rmia, attack_p, ...).
* Gradient- and distillation-based attacks (HSJ, loss_trajectory, seq_mia) will NOT work.
  ``get_grad`` would differentiate a constant dummy parameter and return zeros.

See ``plans/2026-08-11-los-xgboost-notebook.md`` in the workspace for the full rationale.
"""

import numpy as np
import xgboost as xgb
from torch import Tensor, as_tensor, frombuffer, uint8, zeros
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys


class XGBLOS(nn.Module):
    """XGBoost binary classifier exposed through the ``nn.Module`` interface.

    The fitted booster is serialised into the ``booster_bytes`` buffer, which is what makes
    ``state_dict()``, ``torch.save``, and ``leakpro.utils.save_load.hash_model`` work unchanged:
    buffers are part of the state dict, and ``hash_model`` needs tensor values it can call
    ``.numpy().tobytes()`` on.

    Args:
    ----
        input_dim (int): Number of input features. Only used for validation and metadata.
        n_estimators (int): Number of boosting rounds.
        max_depth (int): Maximum tree depth.
        learning_rate (float): Boosting learning rate (eta).
        subsample (float): Row subsampling ratio per tree.
        colsample_bytree (float): Column subsampling ratio per tree.
        reg_lambda (float): L2 regularisation on leaf weights.
        tree_method (str): XGBoost tree construction algorithm.
        random_state (int): Seed for the subsampling RNG.

    """

    def __init__(self,
                 input_dim: int,
                 n_estimators: int = 200,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 subsample: float = 1.0,
                 colsample_bytree: float = 1.0,
                 reg_lambda: float = 1.0,
                 tree_method: str = "hist",
                 random_state: int = 1234) -> None:
        super().__init__()

        # These attribute names must match the __init__ parameter names exactly:
        # leakpro.utils.conversion.get_model_init_params() reconstructs init_params by
        # getattr-ing the constructor signature off the instance.
        self.input_dim = input_dim
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.tree_method = tree_method
        self.random_state = random_state

        self.init_params = {
            "input_dim": input_dim,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_lambda": reg_lambda,
            "tree_method": tree_method,
            "random_state": random_state,
        }

        # A constant placeholder parameter. Torch optimizers raise on an empty parameter list,
        # and get_target_replica() builds one. Kept at a fixed value so that hash_model() is
        # driven entirely by the booster bytes and never by parameter initialisation noise.
        self._dummy = nn.Parameter(zeros(1))

        # The serialised booster. Empty until fit() or load_state_dict() runs.
        self.register_buffer("booster_bytes", zeros(0, dtype=uint8))
        self._booster = None

    # ------------------------------------------------------------------
    # Booster (de)serialisation
    # ------------------------------------------------------------------
    def _set_booster_bytes(self, raw: bytes) -> None:
        """Store a serialised booster and invalidate the cached handle."""
        # bytearray() copies, which avoids torch's read-only-buffer warning on frombuffer.
        self.booster_bytes = frombuffer(bytearray(raw), dtype=uint8)
        self._booster = None

    def _get_booster(self) -> xgb.Booster:
        """Return the booster, deserialising it from the buffer on first use."""
        if self._booster is None:
            if self.booster_bytes.numel() == 0:
                raise RuntimeError(
                    "XGBLOS has no fitted booster. Call fit() before using the model, or load "
                    "a state dict produced by a fitted model."
                )
            booster = xgb.Booster()
            booster.load_model(bytearray(self.booster_bytes.detach().cpu().numpy().tobytes()))
            self._booster = booster
        return self._booster

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit the booster on the given design matrix and binary labels.

        Args:
        ----
            x (np.ndarray): Feature matrix of shape (n_samples, input_dim).
            y (np.ndarray): Binary labels of shape (n_samples,).

        """
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {x.shape[1]}")

        classifier = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            tree_method=self.tree_method,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
            eval_metric="logloss",
        )
        classifier.fit(x, y)
        self._set_booster_bytes(classifier.get_booster().save_raw(raw_format="ubj"))

    # ------------------------------------------------------------------
    # nn.Module interface
    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """Return raw margins (log-odds) of shape (n_samples, 1).

        Raw margins rather than probabilities is the deliberate choice: it makes the model
        consistent with ``BCEWithLogitsLoss`` and with ``PytorchModel.get_rescaled_logits``,
        which applies its own sigmoid on the single-output branch. The (n, 1) shape is required
        because ``PytorchModel.get_loss`` infers the class count from ``output.shape[1]``.
        """
        samples = x.detach().cpu().numpy()
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        samples = np.ascontiguousarray(samples)

        margin = self._get_booster().inplace_predict(samples, predict_type="margin")
        margin = np.asarray(margin)
        if margin.ndim == 1:
            margin = margin.reshape(-1, 1)
        return as_tensor(margin, dtype=x.dtype, device=x.device)

    def load_state_dict(self, state_dict: dict, strict: bool = True, assign: bool = False) -> _IncompatibleKeys:  # noqa: ARG002
        """Restore the booster from a state dict.

        Overridden because the default implementation is strict about buffer *shapes*, and
        ``booster_bytes`` has a different length for every fitted model. ``strict`` and
        ``assign`` are accepted for signature compatibility and ignored.
        """
        if "booster_bytes" not in state_dict:
            raise KeyError("State dict has no 'booster_bytes' entry; it was not produced by XGBLOS.")

        buffer = state_dict["booster_bytes"]
        if buffer.numel() == 0:
            raise ValueError("State dict contains an empty booster; the source model was never fitted.")

        self.booster_bytes = buffer.detach().clone()
        self._booster = None
        self._get_booster()  # fail here, not at first prediction, if the bytes are corrupt
        return _IncompatibleKeys([], [])
