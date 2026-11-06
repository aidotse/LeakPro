# Modular gradient inversion

## What this is

An attack is a `Task` (what the client trains and how it reports back), a modality (how data space maps to the
model input), and an ordered list of `OptimStage`s. Each stage combines weighted loss terms, a representation of the
reconstruction, a driver that optimises it, a target source (labels) and an aggregator. The server sees a
`ClientObservation` (gradients or a weight update, optionally BN statistics and local hyperparameters), and
`run_attack` turns it into a reconstruction held in `state.input`.

## Install

```bash
pip install leakpro[federated]
```

## Run a preset

```python
from leakpro.attacks.gia_attacks.modular import run_attack, simulate_client
from leakpro.attacks.gia_attacks.modular.presets import dlg

attack = dlg(mean=[0.5] * 3, std=[0.25] * 3, device=device, num_classes=10)
observation = simulate_client(model, images, target, task=attack.task, modality=attack.modality,
                              input_shape=tuple(images.shape))
state, ctx = run_attack(attack, model, observation, tuple(images.shape), device)
reconstruction = state.input
```

## Compose your own attack

An attack is a `Task` (what the client trains), a `modality` (how data space maps to model input), and a list of
`OptimStage`s (loss terms, a representation, a driver, a target source, an aggregator). Below, a cosine
gradient-matching attack with a total-variation prior recovers two images from one FedSGD step.

```python
import torch
from torch import nn

from leakpro.attacks.gia_attacks.modular import (
    Attack,
    FedSGD,
    OptimStage,
    TargetInfo,
    Task,
    ThreatModel,
    run_attack,
    simulate_client,
)
from leakpro.attacks.gia_attacks.modular.components.aggregators import BestLossAggregator
from leakpro.attacks.gia_attacks.modular.components.drivers import GradientDriver
from leakpro.attacks.gia_attacks.modular.components.evaluators import ImageEvaluator
from leakpro.attacks.gia_attacks.modular.components.losses import CosineGradientLoss, TVLoss
from leakpro.attacks.gia_attacks.modular.components.modalities import ImageNormalizationModality
from leakpro.attacks.gia_attacks.modular.components.objectives import CrossEntropy
from leakpro.attacks.gia_attacks.modular.components.representations import PixelRepresentation
from leakpro.attacks.gia_attacks.modular.components.targets import FixedTarget

device = torch.device("cpu")
model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 8 * 8, 5))
images, labels = torch.rand(2, 3, 8, 8), torch.tensor([1, 3])

attack = Attack(
    modality=ImageNormalizationModality(mean=[0.5] * 3, std=[0.25] * 3, device=device),
    task=Task(CrossEntropy(), FedSGD(), model_mode="eval"),
    threat_model=ThreatModel(),  # the attacker sees only the gradients; None would grant every observation field
    evaluator=ImageEvaluator(),
    stages=[
        OptimStage(
            losses={"gradient": (CosineGradientLoss(), 1.0), "tv": (TVLoss(), 1e-2)},
            rep=PixelRepresentation(min_val=0.0, max_val=1.0),
            driver=GradientDriver(lr=0.1, max_iter=20),
            target_source=FixedTarget(),
            aggregator=BestLossAggregator(),
        )
    ],
)

# What the server intercepts when the client trains on (images, labels).
target = TargetInfo(value=labels)
observation = simulate_client(model, images, target, task=attack.task, modality=attack.modality,
                              input_shape=tuple(images.shape))

# Reconstruct. `state.input` holds the recovered images in data space.
state, ctx = run_attack(attack, model, observation, tuple(images.shape), device, target=target)
print(state.input.shape)  # torch.Size([2, 3, 8, 8])
```

What to swap:

| To change | Replace | With (from `leakpro.attacks.gia_attacks.modular.components...`) |
|---|---|---|
| Loss terms | the `losses` dict of a stage | `losses`: `CosineGradientLoss`, `L2GradientLoss`, `TVLoss`, `BNStatsLoss`, `L2ImageNormLoss`, `GroupConsistencyLoss`, `EpochOrderInvariantPrior` |
| Representation | `rep` | `representations`: `PixelRepresentation`, `FixedRepresentation` |
| Driver | `driver` | `drivers`: `GradientDriver` (Adam, L-BFGS, SGD) |
| Target source | `target_source` | `targets`: `FixedTarget` (labels passed to `run_attack(target=...)`), `JointTarget` (labels optimised with the image), `NullTarget` |
| Label inference | `Attack(target_inference=...)` | `target_inference`: `IDLGInference`, `DimitrovInference`, `GengInference` |
| Aggregator | `aggregator` | `aggregators`: `BestLossAggregator`, `MeanAggregator`, `EpochMatchingAggregator`, `ChainAggregator` |
| Protocol | `Task(..., FedSGD())` | `FedSGD` (one gradient) or `FedAvg(local_epochs=..., batch_size=..., lr=...)` (a weight update) |
| Objective | `Task(CrossEntropy(), ...)` | `objectives`: `CrossEntropy`, `BinaryCrossEntropy`, `MeanSquaredError` |
| Modality | `modality` | `modalities`: `ImageNormalizationModality` |

## Threat models

`ThreatModel` declares which observation fields the attacker may use:

| Flag | Grants |
|---|---|
| `has_bn_statistics` | `bn_stats` and `pre_bn_stats`, the BN running statistics after and before local training |
| `has_local_hyperparameters` | `batch_size` and `loss_value` of the client's local batch |

`Attack(threat_model=...)` sets it. Every preset declares one. An `Attack` built with `threat_model=None` (the
dataclass default) applies no sanitization, so its components see every field of the observation you pass, including
BN statistics and the batch size; pass `ThreatModel()` for a gradients-only attacker.

`run_attack` sanitizes a copy of the observation (fields without a granted flag become `None`), then checks the
requirements each loss term and label inference declares through `requirements()` (`gradients`, `bn_stats`,
`pre_bn_stats`, `batch_size`, `loss_value`, `proxy_dataloader`). An unmet requirement raises a `ValueError` before
any optimisation, naming the component and the fix.

## Presets

Import from `leakpro.attacks.gia_attacks.modular.presets`; each docstring lists the paper and every deviation from it.

| Factory | Paper | What differs |
|---|---|---|
| `dlg` | Zhu et al. 2019 | L2 matching, L-BFGS, jointly optimised soft labels |
| `idlg` | Zhao et al. 2020 | analytical label inference, L2 matching, L-BFGS |
| `inverting_gradients` | Geiping et al. 2020 | cosine matching with signed gradients, TV, step decay; one restart |
| `huang` | Huang et al. 2021 | adds a BN-statistics term matching the client's running statistics |
| `yin`, `see_through_gradients` | Yin et al. 2021 | parallel seeds, BN and group-consistency priors, seed averaging |
| `dimitrov` | Dimitrov et al. 2022 | FedAvg updates, epoch axis, Hungarian epoch alignment |
| `combined` | LeakPro | multi-seed restarts combined with multi-epoch updates |
| `gia_running`, `gia_estimate` | LeakPro | cosine matching with TV and a BN term; BN statistics from the observation (running) or a proxy dataset (estimate) |

## Extending

Implement the abstract methods of an interface and pass the object to a stage or attack. The interfaces, the
`LossInputs` a loss term receives, the parallel axes (`Axis`, `AxisManager`) and state checkpointing
(`save_state`, `load_state`) are importable from one module:

```python
from leakpro.attacks.gia_attacks.modular.core import Axis, AxisManager, LossInputs, LossTerm, load_state, save_state
```

| Interface | Abstract methods |
|---|---|
| `LossTerm` | `compute(inputs)`; optionally `requirements()`, `bind(ctx)`, `release()` |
| `Driver` | `run(stage, ctx, *, callbacks, stage_idx)` |
| `Representation` | `bind(state, ctx)`, `parameters()`, `decode()`, `encode(data)`, `project(ctx)`, `to_observable()`, `artifacts()` |
| `Modality` | `to_model_input(data)`, `strip_model_input(model_input)`; optionally `client_to_model_input(data)` |
| `Evaluator` | `evaluate(...)` |
| `TargetSource` | `current()`, `slice(...)`; optionally `bind(state, ctx)`, `parameters()`, `reduce(...)` |
| `TargetInference` | `infer(ctx)`; optionally `requirements()` |
| `Aggregator` | `aggregate(result, ctx, stage)` |

## Optional dependencies

All are part of the `federated` extra. Each feature raises an `ImportError` naming the package when it is missing.

| Package | Needed for |
|---|---|
| `lpips` | LPIPS image metrics |
| `torchmetrics` | SSIM in `ImageEvaluator` |
| `pytorch-ignite` | the dataloader SSIM helpers in `leakpro.fl_utils.similarity_measurements` |
