#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
r"""Record reference behaviour of the previous modular framework as fixtures for the parity tests.

The fixtures were recorded from the previous modular framework (branch feat/gia-generative-attacks-and-refactor
plus unpublished fixes). Run this script from a checkout that still contains that package (not collected by pytest):

    PYTHONPATH=<that checkout> python leakpro/tests/gia/parity/record_v1_fixtures.py \\
        --out leakpro/tests/gia/fixtures/v1

Self-contained: imports only torch and that checkout's v1 package, and pickles everything the
parity tests need (model, data, observations), so the two sides never share construction code.

Objective values and iterates are captured from v1's own loss evaluation
(``ComposableOptimizer._compute_loss``) during real ``run_attack`` calls, so the fixtures
include all of v1's per-run setup rather than a re-implementation of it. v1 optimizes the
normalized input ``z = (x - mean) / std``; every iterate is stored in that space and in v1's
``[E, N, G, C, H, W]`` layout (epoch, image, seed). ``target_labels`` are the labels v1 optimized
against (after its own label inference, in its ``[E, N]`` layout); iDLG, for example, recovers a
batch's label multiset but not which image carries which label.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
from collections.abc import Callable
from pathlib import Path

# v1 picks CUDA internally when it is available; the fixtures are CPU float32 by definition.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402
from torch import nn  # noqa: E402

from leakpro.attacks.gia_attacks.modular import presets as v1_presets  # noqa: E402
from leakpro.attacks.gia_attacks.modular.components.composable_optimizer import ComposableOptimizer  # noqa: E402
from leakpro.attacks.gia_attacks.modular.components.initialization import FixedInitialization  # noqa: E402
from leakpro.attacks.gia_attacks.modular.components.label_inference import (  # noqa: E402
    DimitrovLabelInference,
    GengLabelInference,
    IDLGLabelInference,
)
from leakpro.attacks.gia_attacks.modular.components.optimization_building_blocks.training_simulator import (  # noqa: E402
    MultiEpochTrainingSimulation,
    TrainingSettings,
)
from leakpro.attacks.gia_attacks.modular.config.builder import AttackBuilder  # noqa: E402
from leakpro.attacks.gia_attacks.modular.core.state import RunContext  # noqa: E402
from leakpro.fl_utils.fl_client_simulator import FLClientSimulator  # noqa: E402

MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2470, 0.2435, 0.2616]
NUM_CLASSES = 10
SHAPE = (3, 8, 8)
TRAJECTORY_ITERATIONS = 50
RECORDED_ITERATIONS = (0, 1, 2, 5, 10, 25, 49)
CPU = torch.device("cpu")

# preset name -> (v1 factory, FLClientSimulator threat string, factory overrides)
OBJECTIVE_PRESETS: dict[str, tuple[Callable, str, dict]] = {
    "inverting_gradients": (v1_presets.inverting_gradients_attack, "gradients_only", {}),
    "idlg": (v1_presets.idlg_attack, "gradients_only", {}),
    "huang": (v1_presets.huang_attack, "huang", {}),
    "gia_running": (v1_presets.gia_running_attack, "gia_running", {}),
    # A loss point is one image batch, so the multi-seed objective is evaluated with a single seed.
    "see_through_gradients": (v1_presets.see_through_gradients_attack, "huang", {"num_seeds": 1}),
}
TRAJECTORY_PRESETS: dict[str, tuple[Callable, str, dict]] = {
    **OBJECTIVE_PRESETS,
    # Gradient noise draws from the global RNG inside v1's step; recorded without it.
    "see_through_gradients": (v1_presets.see_through_gradients_attack, "huang", {"gradient_noise_std": 0.0}),
    "dimitrov": (v1_presets.dimitrov_fedavg_attack, "gradients_only", {"epochs": 2}),
}


def tiny_bn_cnn(seed: int = 0) -> nn.Sequential:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(),
        nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(32, NUM_CLASSES),
    )


def batch(n: int, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(n, *SHAPE, generator=gen), torch.randperm(NUM_CLASSES, generator=gen)[:n]


def norm(x: torch.Tensor) -> torch.Tensor:
    return (x - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)


def as_list(grads: dict[str, torch.Tensor] | list[torch.Tensor], model: nn.Module) -> list[torch.Tensor]:
    if isinstance(grads, dict):
        return [grads[n].detach().clone() for n, _ in model.named_parameters() if n in grads]
    return [g.detach().clone() for g in grads]


def observe(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, *, epochs: int, compute_mode: str,
            model_mode: str, batch_size: int | None, threat: str):  # noqa: ANN201
    """v1's client simulation. It shuffles mini-batches, so it is seeded; parity replays the recorded result."""
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(norm(images), labels), batch_size=len(images))
    sim = FLClientSimulator(loader, torch.tensor(MEAN).view(3, 1, 1), torch.tensor(STD).view(3, 1, 1), NUM_CLASSES, CPU)
    settings = TrainingSettings(epochs=epochs, optimizer_type="sgd", training_batch_size=batch_size,
                                compute_mode=compute_mode, model_mode=model_mode, shuffle_mode="attack")
    torch.manual_seed(0)
    return sim.train_and_observe(copy.deepcopy(model), settings, nn.CrossEntropyLoss(), threat_model=threat,
                                 send_labels_to_server=True)


def obs_dict(obs, model: nn.Module) -> dict:  # noqa: ANN001
    return {"gradients": as_list(obs.gradients, model), "post_bn_stats": obs.post_bn_stats,
            "pre_bn_stats": obs.pre_bn_stats, "num_images": obs.num_images}


class GeneratedInitialization(FixedInitialization):
    """FixedInitialization whose tensor is built for whatever shape the orchestrator requests."""

    def __init__(self, make: Callable[[tuple[int, ...]], torch.Tensor]) -> None:
        super().__init__(torch.zeros(()))
        self._make = make

    def initialize(self, shape: tuple[int, ...], device: torch.device, dtype: torch.dtype = torch.float32):  # noqa: ANN201
        self.fixed_tensor = self._make(tuple(shape)).detach().clone()
        return super().initialize(shape, device, dtype)


class CaptureObjective:
    """Records v1's own objective evaluations by wrapping ``ComposableOptimizer._compute_loss``.

    L-BFGS evaluates the objective several times per step; the first evaluation of each
    iteration is the one at the iterate the step starts from, and is the one kept.
    """

    def __init__(self, *, with_grad: bool = False) -> None:
        self.with_grad = with_grad
        self.calls: dict[int, dict] = {}

    def __enter__(self) -> CaptureObjective:
        self._original = ComposableOptimizer._compute_loss
        capture = self

        def wrapped(optimizer, state, target_gradients, ctx):  # noqa: ANN001, ANN202
            total, parts = capture._original(optimizer, state, target_gradients, ctx)
            if state.iteration not in capture.calls:
                record = {"z": state.optimizable_tensor.detach().clone(), "labels": state.labels.detach().clone(),
                          "total": float(total), "terms": {name: float(value) for name, value in parts.items()}}
                if capture.with_grad:
                    (grad,) = torch.autograd.grad(total, state.optimizable_tensor, retain_graph=True)
                    record["grad_z"] = grad.detach().clone()
                capture.calls[state.iteration] = record
            return total, parts

        ComposableOptimizer._compute_loss = wrapped
        return self

    def __exit__(self, *exc: object) -> None:
        ComposableOptimizer._compute_loss = self._original


def run_v1(cfg, obs, model: nn.Module, images: torch.Tensor, init: GeneratedInitialization) -> None:  # noqa: ANN001
    orchestrator = AttackBuilder.build(cfg, client_observations=obs)
    orchestrator.initialization = init
    orchestrator.run_attack(copy.deepcopy(model), obs, device=CPU, input_shape=tuple(images.shape), seed=0)


def preset_observation(factory: Callable, threat: str, overrides: dict, model: nn.Module,
                       images: torch.Tensor, labels: torch.Tensor):  # noqa: ANN201
    ts = factory(**overrides).training_simulator
    return observe(model, images, labels, epochs=ts.epochs, compute_mode=ts.compute_mode, model_mode=ts.model_mode,
                   batch_size=ts.batch_size, threat=threat)


def record_signals(out: dict) -> None:
    """The victim's observable update for each training setting."""
    for case, (mode, mm, epochs, bs) in {
        "grad-eval": ("gradients", "eval", 1, None), "grad-train": ("gradients", "train", 1, None),
        "upd-1ep": ("updates", "train", 1, None), "upd-3ep": ("updates", "train", 3, None),
        "upd-3ep-bs1": ("updates", "train", 3, 1), "upd-2ep-bs2-eval": ("updates", "eval", 2, 2),
    }.items():
        model, (images, labels) = tiny_bn_cnn(), batch(4)
        victim = copy.deepcopy(model).train(mm == "train")
        sim = MultiEpochTrainingSimulation(epochs=epochs, optimizer_type="sgd", batch_size=bs, compute_mode=mode,
                                           model_mode=mm, shuffle_mode="attack")
        grads = sim.simulate_training(model=victim, input_data=norm(images), labels=labels, loss_fn=nn.CrossEntropyLoss())
        out[f"signal-{case}"] = {"kind": "signal", "model": model, "images": images, "labels": labels,
                                 "settings": {"compute_mode": mode, "model_mode": mm, "epochs": epochs, "batch_size": bs,
                                              "meta_lr": sim.meta_learning_rate},
                                 "gradients": as_list(grads, victim)}


def record_labels(out: dict) -> None:
    """Labels inferred from the same observation by each v1 label-inference method."""
    for method, make in {"idlg": IDLGLabelInference, "geng": GengLabelInference,
                         "dimitrov": lambda: DimitrovLabelInference(epochs=1, batch_size=4)}.items():
        for mode in ("gradients", "updates"):
            model, (images, labels) = tiny_bn_cnn(), batch(4)
            obs = observe(model, images, labels, epochs=1, compute_mode=mode, model_mode="train", batch_size=None,
                          threat="gradients_only")
            ctx = RunContext(target_model=copy.deepcopy(model), client_observations=obs,
                             training_simulator=MultiEpochTrainingSimulation(epochs=1, compute_mode=mode),
                             loss_fn=nn.CrossEntropyLoss(), seed=0)
            torch.manual_seed(0)
            inferred = make().infer(ctx).labels
            out[f"labels-{method}-{mode}"] = {"kind": "labels", "model": model, "images": images, "labels": labels,
                                              "settings": {"method": method, "compute_mode": mode},
                                              "gradients": as_list(obs.gradients, model), "inferred": inferred}


def record_loss_points(out: dict) -> None:
    """Each preset's objective value, per-term values and gradient at fixed data-space points."""
    std = torch.tensor(STD).view(3, 1, 1)
    for name, (factory, threat, overrides) in OBJECTIVE_PRESETS.items():
        model, (images, labels) = tiny_bn_cnn(), batch(2)
        obs = preset_observation(factory, threat, overrides, model, images, labels)
        # Point 0 is the client's own batch, where gradient matching must vanish; the rest are random.
        points = torch.cat([images[None], torch.rand(2, *images.shape, generator=torch.Generator().manual_seed(100))])
        totals, terms, grads, used_labels = [], [], [], []
        for point in points:
            cfg = factory(**overrides)
            cfg.stages[0].max_iterations = 1
            z = norm(point)
            with CaptureObjective(with_grad=True) as capture:
                run_v1(cfg, obs, model, images, GeneratedInitialization(lambda shape, z=z: z.reshape(shape)))
            first = capture.calls[0]
            totals.append(first["total"])
            terms.append(first["terms"])
            grads.append(first["grad_z"].reshape(images.shape) / std)  # chain rule: dL/dx = dL/dz / std
            used_labels.append(first["labels"])
        out[f"loss-{name}"] = {"kind": "loss_points", "model": model, "images": images, "labels": labels,
                               "settings": {"preset": name, "overrides": overrides,
                                            "config": json.loads(factory(**overrides).model_dump_json()),
                                            "preset_defaults": json.loads(factory().model_dump_json())},
                               "observation": obs_dict(obs, model), "target_labels": used_labels[0],
                               "points": points,
                               "total": torch.tensor(totals), "terms": terms, "grad_wrt_data": torch.stack(grads)}


def record_trajectories(out: dict) -> None:
    """One TRAJECTORY_ITERATIONS-step v1 run per preset from a fixed z0: every loss, and z_t at RECORDED_ITERATIONS.

    z_t is the iterate at which loss L_t is evaluated, so iterates and losses are aligned by construction.
    """
    for name, (factory, threat, overrides) in TRAJECTORY_PRESETS.items():
        model, (images, labels) = tiny_bn_cnn(), batch(2)
        obs = preset_observation(factory, threat, overrides, model, images, labels)
        cfg = factory(**overrides)
        stage = cfg.stages[0]
        stage.max_iterations = TRAJECTORY_ITERATIONS
        stage.patience = 10**9
        gen = torch.Generator().manual_seed(7)
        init = GeneratedInitialization(lambda shape, gen=gen: torch.randn(shape, generator=gen))
        with CaptureObjective() as capture:
            run_v1(cfg, obs, model, images, init)
        if sorted(capture.calls) != list(range(TRAJECTORY_ITERATIONS)):
            raise RuntimeError(f"{name}: v1 stopped after {len(capture.calls)} of {TRAJECTORY_ITERATIONS} iterations")
        out[f"traj-{name}"] = {"kind": "trajectory", "model": model, "images": images, "labels": labels,
                               "settings": {"preset": name, "overrides": overrides,
                                            "config": json.loads(cfg.model_dump_json()),
                                            "preset_defaults": json.loads(factory().model_dump_json())},
                               "observation": obs_dict(obs, model), "target_labels": capture.calls[0]["labels"],
                               "z0": init.fixed_tensor,
                               "steps": list(RECORDED_ITERATIONS),
                               "iterates": [capture.calls[t]["z"] for t in RECORDED_ITERATIONS],
                               "losses": [capture.calls[t]["total"] for t in range(TRAJECTORY_ITERATIONS)],
                               "terms": [capture.calls[t]["terms"] for t in range(TRAJECTORY_ITERATIONS)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True, exist_ok=True)
    cases: dict[str, dict] = {}
    for recorder in (record_signals, record_labels, record_loss_points, record_trajectories):
        recorder(cases)
    manifest = {
        "v1_source": "previous modular framework (branch feat/gia-generative-attacks-and-refactor plus unpublished fixes)",
        "torch": torch.__version__, "python": platform.python_version(), "mean": MEAN, "std": STD, "cases": {},
    }
    for case, payload in cases.items():
        payload.update({"case": case, "mean": MEAN, "std": STD})
        torch.save(payload, args.out / f"{case}.pt")
        manifest["cases"][case] = {"kind": payload["kind"], "settings": payload["settings"], "file": f"{case}.pt"}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    print(f"wrote {len(cases)} fixtures to {args.out}")  # noqa: T201


if __name__ == "__main__":
    main()
