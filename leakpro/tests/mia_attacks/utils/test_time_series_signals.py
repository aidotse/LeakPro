"""Tests for DTW/MSM signals, random shadow sampling, and optimizer param filtering."""
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import optim

from leakpro.signals.signal import SIGNAL_REGISTRY, MSM, create_signal_instance
from leakpro.signals.utils.msm import mv_msm_distance, mv_msm_cost


# ---------------------------------------------------------------------------
# MSM distance unit tests (no external dependency)
# ---------------------------------------------------------------------------

class TestMvMsmCost:
    def test_between_bounds_returns_cost(self) -> None:
        """When a is between b and c, cost is the base cost parameter."""
        result = mv_msm_cost(np.array([2.0]), np.array([1.0]), np.array([3.0]))
        assert result == pytest.approx(np.array([1.0]))

    def test_outside_bounds_returns_cost_plus_min_distance(self) -> None:
        """When a is outside [b, c], cost includes the minimum distance."""
        # a=0, b=1, c=3: min(|0-1|, |0-3|) = 1 => cost + 1 = 2
        result = mv_msm_cost(np.array([0.0]), np.array([1.0]), np.array([3.0]))
        assert result == pytest.approx(np.array([2.0]))


class TestMvMsmDistance:
    def test_identical_series_is_zero(self) -> None:
        """Distance between a series and itself should be 0."""
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert mv_msm_distance(x, x) == pytest.approx(0.0)

    def test_different_series_is_positive(self) -> None:
        x = np.array([[0.0], [0.0], [0.0]])
        y = np.array([[1.0], [1.0], [1.0]])
        assert mv_msm_distance(x, y) > 0

    def test_output_is_scalar(self) -> None:
        x = np.random.rand(4, 2)
        y = np.random.rand(4, 2)
        result = mv_msm_distance(x, y)
        assert isinstance(result, float)

    def test_different_lengths(self) -> None:
        x = np.random.rand(3, 1)
        y = np.random.rand(5, 1)
        result = mv_msm_distance(x, y)
        assert isinstance(result, float)
        assert result >= 0


# ---------------------------------------------------------------------------
# MSM Signal end-to-end test (mocked handler/model)
# ---------------------------------------------------------------------------

class TestMSMSignal:
    def _make_loader(self, n_samples: int = 4, horizon: int = 3, n_vars: int = 2):
        """Return a simple DataLoader with synthetic time series data."""
        data = torch.randn(n_samples, 5, n_vars)   # (N, lookback, vars)
        targets = torch.randn(n_samples, horizon, n_vars)
        dataset = torch.utils.data.TensorDataset(data, targets)
        return torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

    def _make_model(self, horizon: int = 3, n_vars: int = 2):
        model = MagicMock()
        # get_logits returns a numpy array shaped (batch, horizon, n_vars)
        model.get_logits.side_effect = lambda x: x[:, :horizon, :].numpy()
        return model

    def test_msm_signal_returns_one_array_per_model(self) -> None:
        loader = self._make_loader()
        model = self._make_model()
        handler = MagicMock()
        handler.get_dataloader.return_value = loader

        signal = MSM()
        results = signal([model], handler, np.arange(4))

        assert len(results) == 1
        assert results[0].shape == (4,)

    def test_msm_signal_values_non_negative(self) -> None:
        loader = self._make_loader()
        model = self._make_model()
        handler = MagicMock()
        handler.get_dataloader.return_value = loader

        signal = MSM()
        results = signal([model], handler, np.arange(4))
        assert np.all(results[0] >= 0)


# ---------------------------------------------------------------------------
# DTW Signal — ImportError when sktime is absent
# ---------------------------------------------------------------------------

class TestDTWSignal:
    def test_dtw_raises_import_error_without_sktime(self) -> None:
        """DTW signal must raise ImportError when sktime is not installed."""
        from leakpro.signals import signal as sig_module

        original = sig_module.HAS_SKTIME
        try:
            sig_module.HAS_SKTIME = False
            dtw = sig_module.DTW()
            with pytest.raises(ImportError, match="sktime"):
                dtw([], MagicMock(), np.arange(2))
        finally:
            sig_module.HAS_SKTIME = original

    def test_dtw_in_signal_registry(self) -> None:
        assert "DTW" in SIGNAL_REGISTRY

    def test_create_signal_instance_dtw(self) -> None:
        signal = create_signal_instance("DTW")
        from leakpro.signals.signal import DTW
        assert isinstance(signal, DTW)


# ---------------------------------------------------------------------------
# MSM in signal registry
# ---------------------------------------------------------------------------

class TestMSMRegistry:
    def test_msm_in_signal_registry(self) -> None:
        assert "MSM" in SIGNAL_REGISTRY

    def test_create_signal_instance_msm(self) -> None:
        signal = create_signal_instance("MSM")
        assert isinstance(signal, MSM)


# ---------------------------------------------------------------------------
# Random shadow model sampling
# ---------------------------------------------------------------------------

class TestRandomShadowSampling:
    def test_sample_shadow_indices_default(self) -> None:
        """Default implementation returns correct number of indices."""
        from leakpro.tests.input_handler.image_input_handler import ImageInputHandler

        # Access via the abstract method directly (not through a full handler)
        from leakpro.input_handler.abstract_input_handler import AbstractInputHandler

        # Instantiate a concrete subclass through ImageInputHandler mock path
        handler = MagicMock(spec=AbstractInputHandler)
        handler.sample_shadow_indices = AbstractInputHandler.sample_shadow_indices.__get__(handler)

        population = list(range(100))
        fraction = 0.3
        result = handler.sample_shadow_indices(population, fraction)

        assert len(result) == int(len(population) * fraction)
        assert set(result).issubset(set(population))

    def test_shadow_model_handler_random_sampling(self, image_handler) -> None:
        """create_shadow_models with sampling_method='random' completes without error."""
        from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
        from leakpro.tests.constants import get_shadow_model_config
        from leakpro.schemas import ShadowModelConfig

        shadow_config = ShadowModelConfig(**get_shadow_model_config())
        image_handler.configs.shadow_model = shadow_config

        if ShadowModelHandler.is_created():
            ShadowModelHandler.delete_instance()
        sm = ShadowModelHandler(image_handler)

        indices = sm.create_shadow_models(
            2,
            image_handler.test_indices,
            training_fraction=0.5,
            online=False,
            sampling_method="random"
        )
        assert len(indices) > 0

    def test_shadow_model_handler_invalid_sampling_method(self, image_handler) -> None:
        """create_shadow_models raises ValueError for unknown sampling_method."""
        from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
        from leakpro.tests.constants import get_shadow_model_config
        from leakpro.schemas import ShadowModelConfig

        shadow_config = ShadowModelConfig(**get_shadow_model_config())
        image_handler.configs.shadow_model = shadow_config

        if ShadowModelHandler.is_created():
            ShadowModelHandler.delete_instance()
        sm = ShadowModelHandler(image_handler)

        with pytest.raises(ValueError, match="Invalid sampling_method"):
            sm.create_shadow_models(
                2,
                image_handler.test_indices,
                training_fraction=0.5,
                sampling_method="unknown_method"
            )


# ---------------------------------------------------------------------------
# get_optimizer() parameter filtering
# ---------------------------------------------------------------------------

class TestOptimizerParamFiltering:
    def _make_handler(self, optimizer_name: str, params: dict):
        """Build a minimal MIAHandler-like mock with optimizer metadata."""
        from leakpro.input_handler.mia_handler import MIAHandler
        from leakpro.schemas import OptimizerConfig

        handler = MagicMock(spec=MIAHandler)
        handler.target_model_metadata = MagicMock()
        handler.target_model_metadata.optimizer = OptimizerConfig(name=optimizer_name, params=params)
        return handler

    def test_invalid_params_are_filtered(self) -> None:
        """get_optimizer() silently drops params not accepted by the optimizer."""
        from leakpro.input_handler.mia_handler import MIAHandler

        handler = self._make_handler("sgd", {"lr": 0.01, "nonexistent_param": 42})
        model = MagicMock()
        model.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]

        result = MIAHandler.get_optimizer(handler, model)
        assert isinstance(result, optim.SGD)

    def test_valid_params_are_passed_through(self) -> None:
        """get_optimizer() keeps valid params intact."""
        from leakpro.input_handler.mia_handler import MIAHandler

        handler = self._make_handler("adam", {"lr": 0.001, "weight_decay": 1e-4})
        model = MagicMock()
        model.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]

        result = MIAHandler.get_optimizer(handler, model)
        assert isinstance(result, optim.Adam)
        assert result.defaults["lr"] == pytest.approx(0.001)
        assert result.defaults["weight_decay"] == pytest.approx(1e-4)
