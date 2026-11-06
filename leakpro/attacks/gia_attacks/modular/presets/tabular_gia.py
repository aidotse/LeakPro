#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Gradient inversion attack preset for mixed tabular data."""
from __future__ import annotations

from typing import Literal

import torch

from leakpro.attacks.gia_attacks.modular.components.aggregators import BestLossAggregator
from leakpro.attacks.gia_attacks.modular.components.drivers import GradientDriver
from leakpro.attacks.gia_attacks.modular.components.drivers.schedules import LinearDecay
from leakpro.attacks.gia_attacks.modular.components.evaluators import TabularEvaluator
from leakpro.attacks.gia_attacks.modular.components.losses import CosineGradientLoss, L2GradientLoss
from leakpro.attacks.gia_attacks.modular.components.modalities.tabular import CategoricalRelaxation, TabularModality
from leakpro.attacks.gia_attacks.modular.components.objectives import BinaryCrossEntropy, CrossEntropy, MeanSquaredError
from leakpro.attacks.gia_attacks.modular.components.representations.tabular import CatInit, TabularRepresentation
from leakpro.attacks.gia_attacks.modular.components.tabular.schema import TabularSchema
from leakpro.attacks.gia_attacks.modular.components.target_inference import IDLGInference, ZeroLabelInference
from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget, JointTarget
from leakpro.attacks.gia_attacks.modular.core.attack import Attack
from leakpro.attacks.gia_attacks.modular.core.interfaces import ModelMode
from leakpro.attacks.gia_attacks.modular.core.protocol import FedAvg, FedSGD
from leakpro.attacks.gia_attacks.modular.core.stage import OptimStage
from leakpro.attacks.gia_attacks.modular.core.task import Task
from leakpro.attacks.gia_attacks.modular.core.threat_model import ThreatModel

_OBJECTIVES = {"multiclass": CrossEntropy, "binary": BinaryCrossEntropy, "regression": MeanSquaredError}
TaskType = Literal["multiclass", "binary", "regression"]
ComputeMode = Literal["gradients", "updates"]
LabelMode = Literal["idlg", "joint", "known", "zeros"]
LrSchedule = Literal["linear"]
LossReduction = Literal["mean", "sum"]


def tabular(
    continuous_cols: list[int],
    categorical_col_groups: list[tuple[int, int]],
    num_features: int,
    num_classes: int,
    *,
    col_means: dict[int, float] | None = None,
    col_stds: dict[int, float] | None = None,
    feature_names: list[str] | None = None,
    tol: float = 0.319,
    infer_labels: bool = True,
    lr: float = 0.01,
    max_iter: int = 2000,
    model_mode: ModelMode = "train",
    label_mode: LabelMode | None = None,
    task_type: TaskType = "multiclass",
    categorical_relaxation: CategoricalRelaxation = "softmax",
    cat_init: CatInit = "small",
    l2_weight: float = 1e-4,
    lr_schedule: LrSchedule | None = None,
    compute_mode: ComputeMode = "gradients",
    meta_lr: float = 0.01,
    num_local_epochs: int = 1,
    local_batch_size: int | None = None,
    loss_reduction: LossReduction = "mean",
    evaluator: TabularEvaluator | None = None,
) -> Attack:
    """Gradient inversion attack for tabular data with mixed continuous/categorical features.

    Threat model: The attacker sees the client's gradient and knows the model; no other observation field.

    References:
        - Vero et al., "TabLeak: Tabular Data Leakage in Federated Learning", ICML 2023 (softmax relaxation of
          categorical features, tolerance-matched accuracy).
        - Geiping et al., "Inverting Gradients - How easy is it to break privacy in federated learning?",
          NeurIPS 2020 (cosine gradient matching).

    Deviations:
        - Mixed continuous/categorical reconstruction with a softmax relaxation of categorical groups, cosine plus L2
          matching and iDLG labels by default; no TabLeak pooling over restarts or entropy-based assessment.
        - The options below switch to raw one-hot slots, cosine-only matching and other label modes;
          ``tabular_raw_onehot`` bundles one such configuration.

    Args:
        continuous_cols: Indices of continuous feature columns in the F-dim feature vector.
        categorical_col_groups: One entry per categorical feature as
            ``(start_index, num_categories)``.
        num_features: Total number of columns F in the feature vector (after one-hot expansion).
        num_classes: Number of output classes (used when ``infer_labels=False``).
        feature_names: Optional per-column display names, used to label the
            evaluator's per-column accuracy metrics.
        tol: Continuous-feature match tolerance, as a multiple of each feature's
            std (see ``TabularEvaluator``).
        infer_labels: If ``True`` (default), use iDLG label inference before optimisation.
            If ``False``, jointly optimise labels alongside the reconstruction.
        lr: Learning rate for Adam.
        max_iter: Maximum optimisation iterations.
        model_mode: ``"train"`` (default) or ``"eval"`` — controls model dropout/BN behaviour.
        col_means: Optional mean per continuous column (by column index), stored on the schema as metadata;
            the attack does not read it.
        col_stds: Optional std per continuous column, stored like ``col_means``.

    Further options (the defaults keep this preset's configuration; see also ``tabular_raw_onehot``):
        label_mode: Overrides ``infer_labels`` when given. ``"idlg"``, ``"joint"``,
            ``"known"`` (pass the true labels as ``target=`` to ``run_attack``) or
            ``"zeros"`` (label-unknown mode: all labels zero).
        task_type: ``"multiclass"`` (cross-entropy), ``"binary"`` (single-logit
            BCE-with-logits) or ``"regression"`` (MSE).
        categorical_relaxation: ``"softmax"`` or ``"raw"`` (feed one-hot slots as-is).
        cat_init: ``"small"`` N(0, 0.1) or ``"randn"`` N(0, 1) categorical init.
        l2_weight: Weight of the extra L2 gradient term; ``0`` gives pure cosine.
        lr_schedule: ``None`` (constant) or ``"linear"`` (decay lr to 10%).
        compute_mode: ``"gradients"`` (FedSGD) or ``"updates"`` (match a FedAvg update via MetaSGD).
        meta_lr: Client learning rate for ``"updates"`` mode.
        num_local_epochs: Local epochs for ``"updates"`` mode.
        local_batch_size: Local mini-batch size for ``"updates"`` mode.
        loss_reduction: ``"mean"`` or ``"sum"`` for binary/regression tasks.
        evaluator: Custom ``TabularEvaluator`` (e.g. with ``reference_rows`` for the baselines).

    """
    if label_mode is None:
        label_mode = "idlg" if infer_labels else "joint"
    if label_mode not in ("idlg", "joint", "known", "zeros"):
        raise ValueError("label_mode must be 'idlg', 'joint', 'known' or 'zeros'.")
    if task_type not in _OBJECTIVES:
        raise ValueError(f"task_type must be one of {sorted(_OBJECTIVES)}.")
    if lr_schedule not in (None, "linear"):
        raise ValueError("lr_schedule must be None or 'linear'.")
    if label_mode == "idlg" and task_type != "multiclass":
        raise ValueError("iDLG label inference only applies to multiclass tasks.")

    losses = {"cosine": (CosineGradientLoss(), 1.0)}
    if l2_weight > 0:
        losses["l2"] = (L2GradientLoss(), l2_weight)

    target_source = JointTarget(num_classes=1 if task_type != "multiclass" else num_classes) \
        if label_mode == "joint" else FixedTarget()
    target_inference = {
        "idlg": IDLGInference(),
        "zeros": ZeroLabelInference(float_labels=task_type != "multiclass"),
    }.get(label_mode)

    objective_kwargs = {} if task_type == "multiclass" else {"reduction": loss_reduction}
    if compute_mode == "gradients":
        protocol = FedSGD()
    elif compute_mode == "updates":
        protocol = FedAvg(local_epochs=num_local_epochs, batch_size=local_batch_size, lr=meta_lr)
    else:
        raise ValueError(f"compute_mode must be 'gradients' or 'updates', got {compute_mode!r}.")
    task = Task(_OBJECTIVES[task_type](**objective_kwargs), protocol, model_mode=model_mode)

    schema = TabularSchema(
        num_features=num_features,
        continuous_cols=continuous_cols,
        categorical_col_groups=categorical_col_groups,
        col_means=col_means or {},
        col_stds=col_stds or {},
        feature_names=feature_names,
    )

    return Attack(
        modality=TabularModality(schema, categorical_relaxation=categorical_relaxation),
        evaluator=evaluator if evaluator is not None else TabularEvaluator(schema, tol=tol),
        # Plain gradient interception: the attack reads nothing but the gradient (or update).
        threat_model=ThreatModel(),
        task=task,
        stages=[
            OptimStage(
                losses=losses,
                rep=TabularRepresentation(schema, cat_init=cat_init),
                driver=GradientDriver(
                    lr=lr,
                    max_iter=max_iter,
                    return_best=True,
                    optimizer_cls=torch.optim.Adam,
                    schedule=LinearDecay() if lr_schedule == "linear" else None,
                ),
                target_source=target_source,
                aggregator=BestLossAggregator(),
            )
        ],
        target_inference=target_inference,
    )


def tabular_raw_onehot(
    continuous_cols: list[int],
    categorical_col_groups: list[tuple[int, int]],
    num_features: int,
    num_classes: int = 1,
    *,
    task_type: TaskType = "binary",
    label_known: bool = True,
    lr: float = 0.06,
    max_iter: int = 10_000,
    **kwargs: object,
) -> Attack:
    """Tabular attack on raw one-hot slots with cosine-only matching.

    Cosine-only gradient matching, raw one-hot slots fed to the model as they are (no softmax), N(0, 1) init on
    every column, Adam with linear lr decay to 10%, the model in eval mode, and known labels by default (or
    all-zero labels when ``label_known=False``). Other keyword arguments are passed to ``tabular``.

    Threat model: The attacker sees the client's gradient and knows the model; no other observation field.

    References:
        - Geiping et al., "Inverting Gradients - How easy is it to break privacy in federated learning?",
          NeurIPS 2020 (cosine gradient matching).

    Deviations:
        - Categorical columns are optimised as unconstrained one-hot slots rather than through a relaxation, and
          scored after an argmax per block.

    """
    params = {
        "label_mode": "known" if label_known else "zeros",
        "task_type": task_type,
        "categorical_relaxation": "raw",
        "cat_init": "randn",
        "l2_weight": 0.0,
        "lr_schedule": "linear",
        "model_mode": "eval",
    }
    params.update(kwargs)
    return tabular(
        continuous_cols, categorical_col_groups, num_features, num_classes,
        lr=lr, max_iter=max_iter, **params,
    )
