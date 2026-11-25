#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Options of the tabular attack and the ``tabular_raw_onehot`` preset: labels, preset, evaluator, modality.

The binary and regression objectives are tested in ``test_objectives.py``.
"""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from leakpro.attacks.gia_attacks.modular.components.drivers.schedules import LinearDecay
from leakpro.attacks.gia_attacks.modular.components.evaluators import TabularEvaluator
from leakpro.attacks.gia_attacks.modular.components.modalities.tabular import TabularModality
from leakpro.attacks.gia_attacks.modular.components.objectives import BinaryCrossEntropy, CrossEntropy
from leakpro.attacks.gia_attacks.modular.components.tabular.schema import TabularSchema
from leakpro.attacks.gia_attacks.modular.components.target_inference import ZeroLabelInference
from leakpro.attacks.gia_attacks.modular.core.attack import run_attack
from leakpro.attacks.gia_attacks.modular.core.state import TargetInfo
from leakpro.attacks.gia_attacks.modular.presets import tabular, tabular_raw_onehot

KW = dict(continuous_cols=[0, 1], categorical_col_groups=[(2, 3)], col_means={0: 0.0, 1: 0.0}, col_stds={0: 1.0, 1: 1.0},
          num_features=5, num_classes=2)


def _schema():
    return TabularSchema(num_features=5, continuous_cols=[0, 1], categorical_col_groups=[(2, 3)])


# ── labels ───────────────────────────────────────────────────────────────────

def test_zero_label_inference_dtypes():
    ctx = SimpleNamespace(input_shape=(4, 3), device=torch.device("cpu"))
    assert ZeroLabelInference().infer(ctx).value.dtype == torch.long
    z = ZeroLabelInference(float_labels=True).infer(ctx).value
    assert z.dtype == torch.float32 and z.shape == (4,) and not z.any()


# ── preset ───────────────────────────────────────────────────────────────────

def test_default_preset_unchanged():
    atk = tabular(**KW)
    assert isinstance(atk.task.objective, CrossEntropy)
    assert atk.modality.categorical_relaxation == "softmax"
    assert set(atk.stages[0].losses) == {"cosine", "l2"}
    assert atk.stages[0].driver.schedule is None
    assert atk.target_inference is not None


def test_raw_onehot_preset_options():
    atk = tabular_raw_onehot(**KW)
    assert isinstance(atk.task.objective, BinaryCrossEntropy)
    assert atk.modality.categorical_relaxation == "raw"
    assert set(atk.stages[0].losses) == {"cosine"}
    assert isinstance(atk.stages[0].driver.schedule, LinearDecay)
    assert atk.stages[0].driver.lr == 0.06 and atk.stages[0].driver.max_iter == 10_000
    assert atk.target_inference is None  # known labels are passed as target=
    assert tabular_raw_onehot(**KW, label_known=False).target_inference is not None


def test_preset_validation():
    with pytest.raises(ValueError):
        tabular(**KW, label_mode="bogus")
    with pytest.raises(ValueError):
        tabular(**KW, task_type="binary", label_mode="idlg")


def test_linear_decay_lr_reaches_ten_percent():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    sched = LinearDecay().build(opt, 10)
    for _ in range(10):
        opt.step()
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1)


def test_modality_raw_relaxation_is_identity():
    x = torch.randn(3, 5)
    assert torch.equal(TabularModality(_schema(), categorical_relaxation="raw").to_model_input(x), x)
    soft = TabularModality(_schema()).to_model_input(x)
    assert torch.allclose(soft[:, 2:5].sum(-1), torch.ones(3))


@pytest.mark.parametrize("label_mode", ["known", "zeros", "joint"])
def test_raw_onehot_attack_runs_end_to_end(label_mode):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.GELU(), torch.nn.Linear(8, 1)).eval()
    x = torch.cat([torch.randn(1, 2), torch.eye(3)[:1]], dim=1)
    y = torch.ones(1)
    atk = tabular_raw_onehot(**KW, label_mode=label_mode, max_iter=30)
    obs = atk.task.victim_signal(model, x, TargetInfo(value=y))
    state, _ = run_attack(atk, model=model, observation=obs, input_shape=tuple(x.shape), device=torch.device("cpu"),
                          target=TargetInfo(value=y) if label_mode == "known" else None)
    assert state.input.shape == x.shape


# ── evaluator ────────────────────────────────────────────────────────────────

def _rows(n=6):
    g = torch.Generator().manual_seed(0)
    cont = torch.randn(n, 2, generator=g)
    cat = F.one_hot(torch.randint(0, 3, (n,), generator=g), 3).float()
    return torch.cat([cont, cat], dim=1)


def test_unit_std_source_gives_nonzero_tolerance_for_single_row():
    x = _rows(1)
    recon = x.clone()
    recon[0, 0] += 0.1  # within 0.319 standardised units, but batch std of one row is 0
    batch = TabularEvaluator(_schema()).evaluate(recon, x).metrics
    unit = TabularEvaluator(_schema(), std_source="unit").evaluate(recon, x).metrics
    assert batch["continuous_accuracy"] == 0.5
    assert unit["continuous_accuracy"] == 1.0


def test_l1_matching_recovers_permutation_and_emr_thresholds():
    x = _rows(6)
    perm = torch.tensor([3, 1, 5, 0, 2, 4])
    m = TabularEvaluator(_schema(), matching="l1", std_source="unit").evaluate(x[perm], x).metrics
    assert m["reconstruction_accuracy"] == 100.0 and m["emr_90"] == m["emr_80"] == m["emr_60"] == 1.0


def test_emr_thresholds_partial_rows():
    x = _rows(4)
    recon = x.clone()
    recon[:, 0] += 10.0  # first continuous column wrong in every row -> 4/3 hits... 2 of 3 features right
    m = TabularEvaluator(_schema(), std_source="unit").evaluate(recon, x).metrics
    assert m["emr_60"] == 1.0 and m["emr_80"] == 0.0 and m["exact_row_recovery"] == 0.0


def test_prior_and_random_baselines_need_reference_and_behave():
    ref = _rows(200)
    x = ref[:8]
    assert "prior_accuracy" not in TabularEvaluator(_schema(), std_source="unit").evaluate(x, x).metrics
    m = TabularEvaluator(_schema(), std_source="unit", reference_rows=ref, random_trials=16).evaluate(x, x).metrics
    assert m["reconstruction_accuracy"] == 100.0
    assert 0.0 <= m["random_accuracy"] < m["prior_accuracy"] + 0.5
    assert m["gain_over_prior"] == pytest.approx(1.0 - m["prior_accuracy"])
    assert m["gain_over_random"] == pytest.approx(1.0 - m["random_accuracy"])
    assert "prior_categorical_accuracy" in m and "gain_continuous_over_prior" in m


def test_reference_std_source_requires_rows():
    with pytest.raises(ValueError):
        TabularEvaluator(_schema(), std_source="reference")


def test_round_categorical_scalar_codes():
    schema = TabularSchema(num_features=2, continuous_cols=[0], categorical_scalar_cols=[1])
    true = torch.tensor([[0.0, 2.0]])
    recon = torch.tensor([[0.0, 2.0001]])
    assert TabularEvaluator(schema, std_source="unit").evaluate(recon, true).metrics["categorical_accuracy"] == 0.0
    assert TabularEvaluator(schema, std_source="unit", round_categorical=True).evaluate(recon, true).metrics["categorical_accuracy"] == 1.0
