"""Tests for DTW/MSM signals, random shadow sampling, and optimizer param filtering."""
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import optim

from leakpro.signals.signal import SIGNAL_REGISTRY, DTW, MSM, create_signal_instance
from leakpro.signals.utils.dtw import mv_dtw_distance
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


def _textbook_msm_distance(x: np.ndarray, y: np.ndarray, cost: float = 1.0) -> float:
    """Naive univariate MSM (Stefan et al., 2013), written straight from the recurrence."""
    n, m = len(x), len(y)
    accumulated = np.full((n, m), np.inf)
    accumulated[0, 0] = abs(x[0] - y[0])
    for i in range(1, n):
        accumulated[i, 0] = accumulated[i - 1, 0] + mv_msm_cost(x[i], x[i - 1], y[0], cost)
    for j in range(1, m):
        accumulated[0, j] = accumulated[0, j - 1] + mv_msm_cost(y[j], x[0], y[j - 1], cost)
    for i in range(1, n):
        for j in range(1, m):
            accumulated[i, j] = min(
                accumulated[i - 1, j - 1] + abs(x[i] - y[j]),
                accumulated[i - 1, j] + mv_msm_cost(x[i], x[i - 1], y[j], cost),
                accumulated[i, j - 1] + mv_msm_cost(y[j], x[i], y[j - 1], cost),
            )
    return float(accumulated[-1, -1])


class TestMvMsmIsTheIndependentVariant:
    """mv_msm_distance is MSM_I: standard MSM per variable, summed over variables."""

    def test_univariate_matches_the_textbook_recurrence(self) -> None:
        """The vectorised DP must agree with the naive recurrence it replaces."""
        rng = np.random.default_rng(0)
        x, y = rng.normal(size=(7, 1)), rng.normal(size=(5, 1))
        assert mv_msm_distance(x, y) == pytest.approx(_textbook_msm_distance(x[:, 0], y[:, 0]))

    def test_multivariate_is_the_sum_over_variables(self) -> None:
        """Each variable is aligned on its own path, so the distance decomposes exactly."""
        rng = np.random.default_rng(1)
        x, y = rng.normal(size=(6, 3)), rng.normal(size=(8, 3))
        per_variable = sum(
            mv_msm_distance(x[:, [v]], y[:, [v]]) for v in range(x.shape[1])
        )
        assert mv_msm_distance(x, y) == pytest.approx(per_variable)

    def test_one_dimensional_input_is_treated_as_one_variable(self) -> None:
        """Shape (n,) and shape (n, 1) must give the same distance."""
        rng = np.random.default_rng(2)
        x, y = rng.normal(size=6), rng.normal(size=6)
        assert mv_msm_distance(x, y) == pytest.approx(mv_msm_distance(x[:, None], y[:, None]))

    def test_symmetric_in_its_arguments(self) -> None:
        """MSM is a metric, so swapping the series must not change the distance."""
        rng = np.random.default_rng(3)
        x, y = rng.normal(size=(5, 2)), rng.normal(size=(7, 2))
        assert mv_msm_distance(x, y) == pytest.approx(mv_msm_distance(y, x))

    def test_mismatched_variable_count_raises(self) -> None:
        """Comparing series with different variable counts is a caller error, not a broadcast."""
        rng = np.random.default_rng(4)
        with pytest.raises(ValueError, match="same number of variables"):
            mv_msm_distance(rng.normal(size=(4, 2)), rng.normal(size=(4, 3)))

    def test_split_and_merge_charge_the_cost_constant(self) -> None:
        """A larger MSM cost constant makes non-matching alignments more expensive."""
        x, y = np.array([[0.0], [0.0], [0.0]]), np.array([[0.0], [5.0], [0.0], [0.0]])
        assert mv_msm_distance(x, y, cost=2.0) > mv_msm_distance(x, y, cost=1.0)


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
# DTW distance unit tests (no external dependency)
# ---------------------------------------------------------------------------

class TestMvDtwDistance:
    """The in-repo DTW must stay a faithful DTW_D: squared local cost, summed along the path.

    The expected values below were verified against ``sktime.distances.dtw_distance`` (0.40.1)
    before sktime was dropped as a dependency, so they pin the exact semantics the time-series
    signals were originally built on.
    """

    def test_identical_series_is_zero(self) -> None:
        """Distance between a series and itself should be 0."""
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert mv_dtw_distance(x, x) == pytest.approx(0.0)

    def test_local_cost_is_squared_not_absolute(self) -> None:
        """A single unit offset costs 1.0; an absolute-difference cost would also give 1.0,
        so pair this with the scaling test below to pin the square.
        """
        assert mv_dtw_distance(np.array([1.0, 2.0]), np.array([1.0, 3.0])) == pytest.approx(1.0)

    def test_cost_scales_quadratically_with_offset(self) -> None:
        """Doubling the offset must quadruple the distance (squared, not absolute, cost)."""
        one = mv_dtw_distance(np.array([0.0, 0.0]), np.array([0.0, 1.0]))
        two = mv_dtw_distance(np.array([0.0, 0.0]), np.array([0.0, 2.0]))
        assert two == pytest.approx(4.0 * one)

    def test_no_square_root_is_taken(self) -> None:
        """Distance is the raw sum of squared local costs along the path, not its square root."""
        # Two time points each offset by 3 => 9 + 9 = 18, not sqrt(18).
        x = np.array([[0.0], [0.0]])
        y = np.array([[3.0], [3.0]])
        assert mv_dtw_distance(x, y) == pytest.approx(18.0)

    def test_warping_absorbs_a_time_shift(self) -> None:
        """A time-shifted copy must cost far less than the unwarped distance.

        This is the regression guard for the axis convention: axis 0 is time. If the axes are
        swapped, warping cannot absorb the shift and the distance collapses to the no-warp value.
        """
        rng = np.random.default_rng(7)
        base = rng.normal(size=(12, 3))
        shifted = np.vstack([base[0:1], base[:-1]])

        warped = mv_dtw_distance(base, shifted)
        unwarped = float(np.sum((base - shifted) ** 2))

        assert warped == pytest.approx(2.916957, abs=1e-5)
        assert unwarped == pytest.approx(49.248474, abs=1e-5)
        assert warped < unwarped / 10

    def test_symmetric(self) -> None:
        """DTW with a symmetric local cost is itself symmetric."""
        rng = np.random.default_rng(3)
        x, y = rng.normal(size=(9, 2)), rng.normal(size=(9, 2))
        assert mv_dtw_distance(x, y) == pytest.approx(mv_dtw_distance(y, x))

    def test_never_exceeds_unwarped_distance(self) -> None:
        """The diagonal path is always available, so DTW <= the aligned squared distance."""
        rng = np.random.default_rng(4)
        x, y = rng.normal(size=(15, 3)), rng.normal(size=(15, 3))
        assert mv_dtw_distance(x, y) <= np.sum((x - y) ** 2) + 1e-12

    def test_output_is_scalar(self) -> None:
        x = np.random.rand(4, 2)
        y = np.random.rand(4, 2)
        assert isinstance(mv_dtw_distance(x, y), float)

    def test_univariate_1d_input_accepted(self) -> None:
        """Univariate series may be passed as 1d arrays, matching sktime's old behaviour."""
        flat = mv_dtw_distance(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
        column = mv_dtw_distance(np.array([[0.0], [1.0], [0.0]]), np.array([[0.0], [0.0], [1.0]]))
        assert flat == pytest.approx(1.0)
        assert column == pytest.approx(flat)

    def test_different_lengths(self) -> None:
        """Series of unequal length are supported (the cost matrix is rectangular)."""
        x = np.random.rand(3, 1)
        y = np.random.rand(5, 1)
        result = mv_dtw_distance(x, y)
        assert isinstance(result, float)
        assert result >= 0

    def test_single_timepoint_series_maps_to_every_point(self) -> None:
        """A length-1 series has exactly one valid warping path: onto every point of the other.

        Its cost is therefore the summed squared distance to all of them. sktime could not be
        used as a reference here, because its input normaliser silently transposes (k, 1) arrays.
        """
        x = np.array([[1.0, 2.0]])
        y = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        expected = np.sum((x[0] - y) ** 2)  # 5 + 1 + 1 = 7
        assert mv_dtw_distance(x, y) == pytest.approx(expected)
        assert mv_dtw_distance(y, x) == pytest.approx(expected)

    def test_mismatched_variable_count_raises(self) -> None:
        """Series must agree on the number of variables."""
        with pytest.raises(ValueError, match="same number of variables"):
            mv_dtw_distance(np.random.rand(4, 2), np.random.rand(4, 3))


# ---------------------------------------------------------------------------
# DTW Signal end-to-end test (mocked handler/model)
# ---------------------------------------------------------------------------

class TestDTWSignal:
    def test_dtw_signal_returns_one_array_per_model(self) -> None:
        """DTW signal should return one non-negative array per model, one value per sample."""
        data = torch.randn(4, 5, 2)
        targets = torch.randn(4, 3, 2)
        dataset = torch.utils.data.TensorDataset(data, targets)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

        model = MagicMock()
        model.get_logits.side_effect = lambda x: x[:, :3, :].numpy()
        handler = MagicMock()
        handler.get_dataloader.return_value = loader

        results = DTW()([model], handler, np.arange(4))

        assert len(results) == 1
        assert results[0].shape == (4,)
        assert np.all(results[0] >= 0)

    def test_dtw_in_signal_registry(self) -> None:
        assert "DTW" in SIGNAL_REGISTRY

    def test_create_signal_instance_dtw(self) -> None:
        signal = create_signal_instance("DTW")
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
