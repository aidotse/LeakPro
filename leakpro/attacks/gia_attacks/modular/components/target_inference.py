#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Target (label) inference components that estimate the labels from the observed update."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from leakpro.attacks.gia_attacks.modular.core.interfaces import TargetInference
from leakpro.attacks.gia_attacks.modular.core.protocol import FedAvg
from leakpro.attacks.gia_attacks.modular.core.requirements import Requirement
from leakpro.attacks.gia_attacks.modular.core.state import Context, TargetInfo
from leakpro.utils.logger import logger

TargetStrategy = Literal["avg", "start", "end"]


class IDLGInference(TargetInference):
    """Analytical label extraction from the last gradient layer (iDLG, Zhao et al. 2020).

    For a cross-entropy loss the ground-truth label minimizes the last-layer gradient.
    Supports both single-sample (batch_size=1) and multi-sample (batch_size>1) extraction.
    Works on both 1D bias gradients and 2D weight gradients.

    With ``N > 1`` samples it returns the ``N`` classes with the most negative last-layer gradient, so it
    assumes the batch's labels are distinct; a batch with repeated labels gets wrong labels.

    Raises:
        ValueError: From ``infer`` when the batch has more samples than the model has classes, since ``N``
            distinct labels cannot exist then.

    """

    def requirements(self) -> frozenset[Requirement]:
        """The observed gradients the labels are read from."""
        return frozenset({"gradients"})

    def infer(self, ctx: Context) -> TargetInfo:
        """Infers labels from the sign of the last-layer gradient."""
        batch_size = ctx.input_shape[0]
        last_grad = ctx.observation.gradients[-1]

        # If the observation is an update (ΔW = -η∇L), negate to restore gradient signs.
        if isinstance(getattr(ctx.task, "protocol", None), FedAvg):
            last_grad = -last_grad

        flat_grad = last_grad if last_grad.dim() == 1 else last_grad.sum(dim=1)
        if batch_size > flat_grad.numel():
            raise ValueError(
                f"IDLGInference returns one distinct label per sample, but the batch has {batch_size} samples and the "
                f"model only {flat_grad.numel()} classes; use DimitrovInference for batches with repeated labels."
            )
        labels = flat_grad.argmin().unsqueeze(0) if batch_size == 1 else torch.argsort(flat_grad)[:batch_size]
        return TargetInfo(value=labels)


class DimitrovInference(TargetInference):
    """Dimitrov label inference for multi-epoch FedAvg attacks (Dimitrov et al., TMLR 2022).

    Estimates label counts by interpolating network statistics (softmax outputs and ReLU
    activations) between the server and client models, then rounds fractional counts to
    integers summing to the batch size.

    Formula: counts = K*ps - (K*dW) / (O*s)
    where ps = softmax probs, dW = last-layer weight gradient sum,
    O = last-ReLU activation sum, s = total steps, K = batch size.

    The client's epochs, batch size and learning rate come from the ``FedAvg`` protocol; a constructor
    argument overrides the protocol's value.

    ``O`` is the summed output of the last activation in the model whose type is in ``activation_types``.
    With ``strict_activation=True`` (the default) a model with none of them raises. With
    ``strict_activation=False`` it falls back to the summed absolute logits, a different quantity that
    can give wrong counts, and logs a warning.

    Raises:
        ValueError: From ``infer`` when the payload does not have one tensor per trainable parameter, or
            (strict) when the model has none of ``activation_types``.

    """

    #: Activation modules treated as "the last nonlinearity" when no explicit type is given. The formula
    #: needs the summed output of the layer feeding the classifier head; ReLU is what the paper's
    #: architectures use, and the other common families are listed so such networks need no override.
    DEFAULT_ACTIVATION_TYPES: tuple[type, ...] = (
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.SiLU, nn.ELU,
    )

    def requirements(self) -> frozenset[Requirement]:
        """The observed gradients the labels are read from."""
        return frozenset({"gradients"})

    def __init__(
        self,
        epochs: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        num_dummy_samples: int = 100,
        strategy: TargetStrategy = "avg",
        activation_types: tuple[type, ...] | None = None,
        strict_activation: bool = True,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_dummy_samples = num_dummy_samples
        self.strategy = strategy
        self.activation_types = (
            self.DEFAULT_ACTIVATION_TYPES if activation_types is None else tuple(activation_types)
        )
        self.strict_activation = strict_activation

    def _client_settings(self, ctx: Context) -> tuple[int, int, float]:
        """``(epochs, batch_size, lr)`` of the client.

        An explicit constructor argument wins; otherwise the ``FedAvg`` protocol supplies the value
        (``batch_size=None`` there means the whole input); otherwise the client is one full-batch
        step at lr 0.1.
        """
        protocol = getattr(getattr(ctx, "task", None), "protocol", None)
        n = ctx.input_shape[0]
        if isinstance(protocol, FedAvg):
            epochs, batch_size, lr = protocol.local_epochs, protocol.batch_size or n, protocol.lr
        else:
            epochs, batch_size, lr = 1, n, 0.1
        return (
            self.epochs if self.epochs is not None else epochs,
            self.batch_size if self.batch_size is not None else batch_size,
            self.learning_rate if self.learning_rate is not None else lr,
        )

    def infer(self, ctx: Context) -> TargetInfo:
        """Infers labels using the client's epochs, batch size and learning rate."""
        return self._infer(ctx, *self._client_settings(ctx))

    def _infer(self, ctx: Context, epochs: int, batch_size: int, lr: float) -> TargetInfo:
        """Label inference for a client running ``epochs`` epochs of batch size ``batch_size`` at ``lr``; stores no state."""
        gradients = list(ctx.observation.gradients)
        is_update = isinstance(getattr(ctx.task, "protocol", None), FedAvg)
        if is_update:
            # A FedAvg payload is theta_E - theta_0 = -lr * sum(grad); the formula needs the raw gradient sum.
            gradients = [-g / lr for g in gradients]

        device = ctx.device
        input_shape = ctx.input_shape[1:]  # (C, H, W) — strip the batch dim
        num_samples = ctx.input_shape[0]

        weight_grad_sum = self._extract_weight_gradient_sum(gradients)

        server_params = {name: param.data.clone() for name, param in ctx.model.named_parameters()}
        client_params = self._compute_client_params(gradients, ctx.model, server_params, lr)

        ps_start, output_sum_start = self._calc_label_stats(ctx.model, server_params, input_shape, device)
        ps_end, output_sum_end = self._calc_label_stats(ctx.model, client_params, input_shape, device)

        k_batches = (num_samples + batch_size - 1) // batch_size
        total_steps = k_batches * epochs

        raw_counts = self._compute_label_counts_by_strategy(
            ps_start, output_sum_start, ps_end, output_sum_end,
            weight_grad_sum, total_steps, num_samples, device,
        )

        label_counts = self._round_label_counts(raw_counts, num_samples)

        labels: list[int] = []
        for label_idx, count in enumerate(label_counts):
            labels.extend([label_idx] * int(count.item()))

        labels_tensor = torch.tensor(labels[:num_samples], dtype=torch.long, device=device)
        return TargetInfo(value=labels_tensor)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _calc_label_stats(
        self,
        model: nn.Module,
        params: dict[str, torch.Tensor],
        input_shape: tuple[int, ...],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dummy_inputs = torch.randn(self.num_dummy_samples, *input_shape, device=device)

        # Cloned: state_dict() aliases the live tensors, which load_state_dict overwrites in place.
        original_state = {name: value.clone() for name, value in model.state_dict().items()}
        was_training = model.training
        new_state = {name: params.get(name, value) for name, value in original_state.items()}
        model.load_state_dict(new_state)
        model.eval()
        try:
            return self._label_stats(model, dummy_inputs)
        finally:
            model.load_state_dict(original_state)
            model.train(was_training)

    def _label_stats(self, model: nn.Module, dummy_inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        last_relu_outputs: list[torch.Tensor] = []

        def hook_fn(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:  # noqa: ARG001 -- forward-hook signature
            last_relu_outputs.append(output.detach())

        last_relu = None
        for module in model.modules():
            if isinstance(module, self.activation_types):
                last_relu = module

        if last_relu is None and self.strict_activation:
            names = ", ".join(t.__name__ for t in self.activation_types)
            raise ValueError(
                f"DimitrovInference found no activation of type ({names}) in the model. "
                "The label-count formula needs the summed activation feeding the "
                "classifier head; pass activation_types=(...) for this architecture, "
                "or set strict_activation=False to accept the |output| approximation."
            )

        if last_relu is None:
            logger.warning(
                "DimitrovInference: no activation of the configured types in the model; approximating the summed "
                "activation by the summed absolute logits (strict_activation=False). Label counts may be wrong."
            )
        handle = last_relu.register_forward_hook(hook_fn) if last_relu is not None else None

        with torch.no_grad():
            outputs = model(dummy_inputs)
            softmax_probs = torch.softmax(outputs, dim=1)

        if handle is not None:
            handle.remove()

        ps = softmax_probs.mean(dim=0)
        output_sum = (
            last_relu_outputs[0].sum(dim=-1).mean() if last_relu_outputs
            else outputs.abs().sum(dim=-1).mean()
        )
        return ps, output_sum

    def _extract_weight_gradient_sum(self, gradients: list[torch.Tensor]) -> torch.Tensor:
        for grad in reversed(gradients):
            if grad is not None and isinstance(grad, torch.Tensor) and grad.dim() == 2:
                return grad.sum(dim=1)
        raise ValueError("Could not find last layer weight gradient for DimitrovInference.")

    def _compute_client_params(
        self,
        gradients: list[torch.Tensor],
        model: nn.Module,
        server_params: dict[str, torch.Tensor],
        lr: float,
    ) -> dict[str, torch.Tensor]:
        """The weights after one step of ``lr`` along the payload; frozen parameters keep the server value.

        The payload holds one tensor per trainable parameter, in ``model.parameters()`` order (a frozen
        parameter is not part of it).
        """
        trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
        if len(gradients) != len(trainable):
            raise ValueError(
                f"DimitrovInference: the payload has {len(gradients)} tensors but the model has {len(trainable)} "
                "trainable parameters; the observation does not come from this model."
            )
        client_params = {name: value.clone() for name, value in server_params.items()}
        for (name, param), grad in zip(trainable, gradients):
            if grad is not None:
                client_params[name] = param.data - lr * grad
        return client_params

    def _calc_raw_counts(
        self,
        ps: torch.Tensor,
        output_sum: torch.Tensor,
        weight_grad_sum: torch.Tensor,
        total_steps: int,
        batch_size_k: int,
    ) -> torch.Tensor:
        return batch_size_k * ps - (batch_size_k * weight_grad_sum) / (output_sum * total_steps)

    def _round_label_counts(self, counts: torch.Tensor, batch_size_k: int) -> torch.Tensor:
        counts_floor = torch.clamp(torch.floor(counts), min=0).long()
        counts_rem = batch_size_k - counts_floor.sum()
        remainders = counts - counts_floor.float()

        if counts_rem >= 0:
            _, top_indices = torch.topk(remainders, min(int(counts_rem), len(remainders)))
            counts_floor[top_indices] += 1
        else:
            max_remove = -int(counts_rem)
            can_remove = counts_floor >= 1
            num_can_remove = int(can_remove.sum().item())
            if max_remove <= num_can_remove:
                masked = remainders.clone()
                masked[~can_remove] = float("inf")
                _, bottom = torch.topk(-masked, max_remove)
                counts_floor[bottom] -= 1
            else:
                counts_floor[can_remove] -= 1
                remaining = min(max_remove - num_can_remove, len(counts_floor))
                if remaining > 0:
                    _, top = torch.topk(counts_floor.float(), remaining)
                    counts_floor[top] -= 1

        return torch.clamp(counts_floor, min=0)

    def _compute_label_counts_by_strategy(
        self,
        ps_start: torch.Tensor,
        output_sum_start: torch.Tensor,
        ps_end: torch.Tensor,
        output_sum_end: torch.Tensor,
        weight_grad_sum: torch.Tensor,
        total_steps: int,
        num_samples: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.strategy == "start":
            return self._calc_raw_counts(ps_start, output_sum_start, weight_grad_sum, total_steps, num_samples)
        if self.strategy == "end":
            return self._calc_raw_counts(ps_end, output_sum_end, weight_grad_sum, total_steps, num_samples)

        # "avg" — average over linearly interpolated checkpoints
        coefs = torch.linspace(0, 1, total_steps, device=device)
        all_raw: list[torch.Tensor] = []
        for coef in coefs:
            ps_i = (1 - coef) * ps_start + coef * ps_end
            os_i = (1 - coef) * output_sum_start + coef * output_sum_end
            all_raw.append(self._calc_raw_counts(ps_i, os_i, weight_grad_sum, total_steps, num_samples))
        return torch.stack(all_raw).mean(dim=0)


class GengInference(TargetInference):
    """Label-count inference for a single FedSGD step (Geng et al. 2021).

    The label counts follow from the last-layer weight gradient and the network's softmax and last-activation
    statistics at the server weights: the Dimitrov formula with one full-batch step at lr 0.1 and the
    statistics taken at the start, whatever the protocol says.

    References:
        - Geng, Mou, Li, Li, Wang, Towards General Deep Leakage in Federated Learning, arXiv:2110.09074, 2021.

    """

    def __init__(self, num_dummy_samples: int = 100) -> None:
        self.num_dummy_samples = num_dummy_samples

    def requirements(self) -> frozenset[Requirement]:
        """The observed gradients the labels are read from."""
        return frozenset({"gradients"})

    def infer(self, ctx: Context) -> TargetInfo:
        """Infers labels assuming one epoch of the observed batch size at a fixed learning rate."""
        counts = DimitrovInference(num_dummy_samples=self.num_dummy_samples, strategy="start")
        return counts._infer(ctx, 1, ctx.input_shape[0], 0.1)  # noqa: SLF001 -- same module, fixed client settings


class ZeroLabelInference(TargetInference):
    """Label-unknown baseline: every label is zero.

    No inference is attempted; the attacker simply plugs in a constant.  Use
    ``float_labels=True`` for single-output (binary / regression) tasks.
    """

    def __init__(self, float_labels: bool = False) -> None:
        self.float_labels = float_labels

    def infer(self, ctx: Context) -> TargetInfo:
        """Returns an all-zero label tensor of the batch size."""
        batch_size = ctx.input_shape[0]
        dtype = torch.float32 if self.float_labels else torch.long
        return TargetInfo(value=torch.zeros(batch_size, dtype=dtype, device=ctx.device))
