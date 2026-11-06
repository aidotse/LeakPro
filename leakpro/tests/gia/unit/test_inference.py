#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for IDLGInference, DimitrovInference helper methods.

infer() is called directly with a hand-crafted Context mock — no model,
no GradientDriver, no run_attack.
"""
import types

import pytest
import torch


from leakpro.attacks.gia_attacks.modular.components.target_inference import DimitrovInference, IDLGInference
from leakpro.attacks.gia_attacks.modular.core.protocol import FedAvg, FedSGD
from leakpro.attacks.gia_attacks.modular.core.state import ClientObservation, Context


def _ctx_with_last_grad(last_grad: torch.Tensor, batch_size: int = 1, task=None):
    """Minimal Context-like namespace exposing observation.gradients and input_shape."""
    obs = ClientObservation(gradients=[torch.zeros(1), last_grad])
    ctx = types.SimpleNamespace(observation=obs, input_shape=(batch_size, 3, 32, 32), task=task)
    return ctx


class TestIDLGInference:
    def setup_method(self):
        self.infer = IDLGInference()

    # ── 1-D bias gradient ─────────────────────────────────────────────────────

    def test_1d_bias_grad_selects_most_negative_index(self):
        """For a 1-D gradient, infer returns argmin (the index with the largest negative value).

        Tensor: [0.5, 0.5, -999.0, 0.5]  =>  argmin == 2
        """
        TARGET_IDX = 2
        grad = torch.tensor([0.5, 0.5, -999.0, 0.5])
        result = self.infer.infer(_ctx_with_last_grad(grad))
        assert result.value.item() == TARGET_IDX

    def test_1d_bias_grad_with_last_index_minimum(self):
        """argmin at the last position to guard against off-by-one."""
        TARGET_IDX = 4
        grad = torch.tensor([1.0, 2.0, 3.0, 4.0, -10.0])
        result = self.infer.infer(_ctx_with_last_grad(grad))
        assert result.value.item() == TARGET_IDX

    # ── 2-D weight gradient ───────────────────────────────────────────────────

    def test_2d_weight_grad_selects_row_with_most_negative_sum(self):
        """For a 2-D gradient, infer sums across dim=1 then takes argmin.

        Row sums:
          row 0: 1 + 2 = 3.0
          row 1: 0.5 + (-10.0) = -9.5   <-- minimum => TARGET_IDX = 1
          row 2: 3 + 4 = 7.0
        """
        TARGET_IDX = 1
        grad = torch.tensor([
            [1.0,  2.0],
            [0.5, -10.0],
            [3.0,  4.0],
        ])
        result = self.infer.infer(_ctx_with_last_grad(grad))
        assert result.value.item() == TARGET_IDX

    def test_2d_weight_grad_with_first_row_minimum(self):
        """argmin at row 0 — verifies the first class can also be inferred."""
        TARGET_IDX = 0
        grad = torch.tensor([
            [-5.0, -6.0],   # sum = -11.0  <-- minimum
            [ 1.0,  2.0],   # sum =   3.0
            [ 0.0,  0.5],   # sum =   0.5
        ])
        result = self.infer.infer(_ctx_with_last_grad(grad))
        assert result.value.item() == TARGET_IDX

    # ── Return shape ──────────────────────────────────────────────────────────

    def test_return_value_is_1d_tensor_of_length_1(self):
        """infer() wraps the scalar label in a 1-element tensor (unsqueeze(0))."""
        grad = torch.tensor([0.1, -0.5, 0.3])
        result = self.infer.infer(_ctx_with_last_grad(grad))
        assert result.value.shape == (1,)


# ── IDLGInference — batch_size > 1 ────────────────────────────────────────────

class TestIDLGInferenceBatchGt1:
    def setup_method(self):
        self.infer = IDLGInference()

    def test_batch_gt1_uses_argsort_not_argmin(self):
        """For batch_size > 1, infer returns the top-k most-negative labels via argsort.

        Gradient: [0.5, -2.0, -1.0, 0.1]
        argsort ascending: [1, 2, 3, 0]  (most negative first)
        batch_size=2 → labels should be [1, 2].
        """
        grad = torch.tensor([0.5, -2.0, -1.0, 0.1])
        result = self.infer.infer(_ctx_with_last_grad(grad, batch_size=2))
        assert result.value.tolist() == [1, 2]

    def test_batch_gt1_returns_correct_count(self):
        """Number of returned labels equals batch_size."""
        grad = torch.tensor([0.1, 0.2, -0.5, 0.3, -0.1])
        result = self.infer.infer(_ctx_with_last_grad(grad, batch_size=3))
        assert result.value.shape == (3,)

    def test_batch_size_1_still_uses_argmin(self):
        """batch_size=1 code path (argmin) is unaffected by the multi-sample branch."""
        grad = torch.tensor([0.5, -2.0, -1.0, 0.1])
        result = self.infer.infer(_ctx_with_last_grad(grad, batch_size=1))
        assert result.value.item() == 1  # argmin index


# ── IDLGInference — updates mode ──────────────────────────────────────────────

class TestIDLGInferenceUpdatesMode:
    def setup_method(self):
        self.infer = IDLGInference()

    def test_updates_mode_negates_gradient_before_argmin(self):
        """Under FedAvg the gradient is negated so argmin reflects ∇L sign.

        Original gradient (ΔW): [1.0, 2.0, 3.0]
        After negation:          [-1.0, -2.0, -3.0] → argmin = 2
        Without negation:        argmin of [1.0, 2.0, 3.0] = 0
        """
        task_stub = types.SimpleNamespace(protocol=FedAvg())
        grad = torch.tensor([1.0, 2.0, 3.0])

        result_with = self.infer.infer(_ctx_with_last_grad(grad, task=task_stub))
        result_without = self.infer.infer(_ctx_with_last_grad(grad))

        assert result_with.value.item() == 2    # argmin of negated grad
        assert result_without.value.item() == 0  # argmin of original grad

    def test_gradients_mode_does_not_negate(self):
        """FedSGD must leave the gradient unchanged."""
        task_stub = types.SimpleNamespace(protocol=FedSGD())
        grad = torch.tensor([1.0, 2.0, 3.0])

        result = self.infer.infer(_ctx_with_last_grad(grad, task=task_stub))
        assert result.value.item() == 0  # argmin without negation

    def test_task_none_does_not_negate(self):
        """ctx.task=None (no protocol) must behave as FedSGD."""
        # _ctx_with_last_grad passes task=None by default
        result = self.infer.infer(_ctx_with_last_grad(torch.tensor([1.0, 2.0, 3.0]), task=None))
        assert result.value.item() == 0  # no negation


# ── DimitrovInference helper methods ──────────────────────────────────────────

class TestDimitrovInferenceHelpers:
    def setup_method(self):
        self.d = DimitrovInference()

    # --- _calc_raw_counts ---

    def test_calc_raw_counts_matches_formula(self):
        """counts = K*ps - (K*dW) / (O*S) evaluated element-wise."""
        ps = torch.tensor([0.3, 0.3, 0.4])
        output_sum = torch.tensor(2.0)
        weight_grad_sum = torch.tensor([0.1, 0.1, 0.1])
        total_steps = 5
        batch_size_k = 3

        result = self.d._calc_raw_counts(ps, output_sum, weight_grad_sum, total_steps, batch_size_k)
        expected = batch_size_k * ps - (batch_size_k * weight_grad_sum) / (output_sum * total_steps)
        assert torch.allclose(result, expected)

    def test_calc_raw_counts_zero_gradient_returns_scaled_probs(self):
        """When weight_grad_sum=0, raw_counts = K * ps."""
        ps = torch.tensor([0.2, 0.5, 0.3])
        output_sum = torch.tensor(1.0)
        weight_grad_sum = torch.zeros(3)
        result = self.d._calc_raw_counts(ps, output_sum, weight_grad_sum, total_steps=10, batch_size_k=4)
        expected = 4 * ps
        assert torch.allclose(result, expected)

    # --- _round_label_counts ---

    def test_round_label_counts_sums_to_batch_size(self):
        """Output must always sum to batch_size_k."""
        counts = torch.tensor([0.87, 0.87, 1.17])
        result = self.d._round_label_counts(counts, batch_size_k=3)
        assert result.sum().item() == 3

    def test_round_label_counts_all_nonnegative(self):
        """No negative counts in output."""
        counts = torch.tensor([0.87, 0.87, 1.17])
        result = self.d._round_label_counts(counts, batch_size_k=3)
        assert (result >= 0).all()

    def test_round_label_counts_negative_raw_clamped(self):
        """Negative raw counts must be clamped and the sum still equals batch_size_k."""
        counts = torch.tensor([-0.5, -0.2, 3.8])
        result = self.d._round_label_counts(counts, batch_size_k=3)
        assert result.sum().item() == 3
        assert (result >= 0).all()

    def test_round_label_counts_exact_integers_unchanged(self):
        """If raw counts are already exact integers that sum to K, output equals floor."""
        counts = torch.tensor([1.0, 1.0, 1.0])
        result = self.d._round_label_counts(counts, batch_size_k=3)
        assert result.sum().item() == 3
        assert (result >= 0).all()

    def test_round_label_counts_overcount_is_corrected(self):
        """Floor sums > batch_size_k must be reduced to exactly batch_size_k."""
        # floors: [1,1,1,1] = 4 but batch_size_k = 3
        counts = torch.tensor([1.9, 1.9, 1.9, 1.9])
        result = self.d._round_label_counts(counts, batch_size_k=3)
        assert result.sum().item() == 3


from types import SimpleNamespace  # noqa: E402


def _settings_ctx(protocol, n: int = 6) -> SimpleNamespace:
    return SimpleNamespace(task=SimpleNamespace(protocol=protocol), input_shape=(n, 3, 8, 8))


class TestDimitrovClientSettings:
    def test_fedavg_protocol_supplies_the_settings(self) -> None:
        d = DimitrovInference()
        assert d._client_settings(_settings_ctx(FedAvg(local_epochs=2, batch_size=3, lr=0.5))) == (2, 3, 0.5)

    def test_full_batch_fedavg_uses_the_batch_size_of_the_input(self) -> None:
        assert DimitrovInference()._client_settings(_settings_ctx(FedAvg(local_epochs=2, lr=0.5))) == (2, 6, 0.5)

    def test_explicit_arguments_win(self) -> None:
        d = DimitrovInference(epochs=1, learning_rate=0.2)
        assert d._client_settings(_settings_ctx(FedAvg(local_epochs=2, batch_size=3, lr=0.5))) == (1, 3, 0.2)

    def test_fedsgd_is_one_full_batch_step(self) -> None:
        assert DimitrovInference()._client_settings(_settings_ctx(FedSGD())) == (1, 6, 0.1)

    def test_no_protocol_behaves_as_fedsgd(self) -> None:
        assert DimitrovInference()._client_settings(SimpleNamespace(task=None, input_shape=(4, 3, 8, 8))) == (1, 4, 0.1)


# ── DimitrovInference label recovery from a FedAvg payload ────────────────────

def _fedavg_ctx(lr: float, labels: torch.Tensor) -> Context:
    from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
    from leakpro.attacks.gia_attacks.modular.components.modalities.image import ImageNormalizationModality
    from leakpro.attacks.gia_attacks.modular.core.axes import AxisManager
    from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo
    from leakpro.attacks.gia_attacks.modular.core.task import Task

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Flatten(), torch.nn.Linear(12, 32), torch.nn.ReLU(), torch.nn.Linear(32, 10)
    )
    task = Task(CrossEntropy(), protocol=FedAvg(local_epochs=1, lr=lr), model_mode="eval")
    images = torch.randn(labels.numel(), 3, 2, 2)
    out = task.victim_signal(model, images, TargetInfo(value=labels))
    obs = ClientObservation(gradients=[g.detach().clone() for g in out.gradients])
    modality = ImageNormalizationModality(mean=[0.0] * 3, std=[1.0] * 3, device=torch.device("cpu"))
    return Context(
        model=model, observation=obs, modality=modality, task=task,
        input_shape=tuple(images.shape), device=torch.device("cpu"), axes=AxisManager(),
    )


class TestDimitrovLabelRecovery:
    @pytest.mark.parametrize("lr", [0.01, 0.1])
    def test_fedavg_payload_recovers_the_label_multiset(self, lr: float) -> None:
        truth = torch.tensor([0, 0, 1, 2, 2, 2, 5, 7])
        recovered = DimitrovInference().infer(_fedavg_ctx(lr, truth)).value
        overlap = sum(min((recovered == c).sum().item(), (truth == c).sum().item()) for c in range(10))
        assert overlap >= 7


class TestGengLabelRecovery:
    def test_fedavg_payload_at_the_fixed_geng_lr_recovers_the_label_multiset(self) -> None:
        from leakpro.attacks.gia_attacks.modular.components.target_inference import GengInference

        truth = torch.tensor([0, 0, 1, 2, 2, 2, 5, 7])
        recovered = GengInference().infer(_fedavg_ctx(0.1, truth)).value
        overlap = sum(min((recovered == c).sum().item(), (truth == c).sum().item()) for c in range(10))
        assert overlap >= 7
