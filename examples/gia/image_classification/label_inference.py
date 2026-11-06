#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""How many labels does each label-inference method recover, as the client's batch grows?"""
from __future__ import annotations

import torch

from examples.gia.common import cli, data, models, results, runner, viz
from leakpro.attacks.gia_attacks.modular.components.evaluators import LabelEvaluator
from leakpro.attacks.gia_attacks.modular.components.target_inference import DimitrovInference, GengInference, IDLGInference
from leakpro.attacks.gia_attacks.modular.presets import dimitrov, inverting_gradients

CLIENT_LR = 0.01


def add_args(p) -> None:
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--trials", type=int, default=None, help="Client batches per batch size (default 20, quick 2).")
    p.add_argument("--protocol", choices=("fedsgd", "fedavg"), default="fedavg")
    p.add_argument("--local-epochs", type=int, default=3)


def _plot(out, sizes: list[int], rows: dict) -> None:
    methods = sorted({k.split(" / ", 1)[1] for k in rows})
    nan = float("nan")  # iDLG has no point where the batch outgrows the class count
    series = {m: [results.mean(rows[f"bs={b} / {m}"]["multiset_accuracy"]) if f"bs={b} / {m}" in rows else nan for b in sizes]
              for m in methods}
    viz.plot_curves(sizes, series, "client batch size", "label accuracy", out / "label_accuracy.png")


def main(argv: list[str] | None = None) -> dict:
    """Infer the labels of a client's update with every method; score the multiset overlap with the true labels."""
    a = cli.parse(__doc__, argv, extra=add_args)
    if a.replot:
        saved = results.load(a.out)
        _plot(a.out, saved["config"]["batch_sizes"], saved["rows"])
        return {}
    device = cli.device_for(a)
    trials = a.trials or (2 if a.quick else 20)
    epochs = a.local_epochs if a.protocol == "fedavg" else 1
    sizes = a.batch_sizes[:3] if a.quick else a.batch_sizes
    rows: dict = {}
    for bs in sizes:
        client_bs = max(1, bs // 2) if a.protocol == "fedavg" else bs
        for trial in range(trials):
            torch.manual_seed(a.seed + trial)
            batch = data.image_batch(a.quick, bs, a.data_dir, device, start_idx=trial * bs)
            if a.quick:  # synthetic labels cycle through the classes; draw them at random instead
                batch.labels = torch.randint(0, batch.num_classes, (bs,), device=device)
            model = (models.tiny_cnn(batch.num_classes) if a.quick else models.cifar_resnet(batch.num_classes)).to(device).eval()
            args = (batch.mean, batch.std, device)
            if a.protocol == "fedavg":
                base = dimitrov(*args, num_epochs=epochs, client_batch_size=client_bs, client_lr=CLIENT_LR, max_iter=1)
            else:
                base = inverting_gradients(*args, max_iter=1)
            methods = {
                "iDLG": IDLGInference(),
                "Geng": GengInference(num_dummy_samples=10),
                "Dimitrov (start)": DimitrovInference(epochs, client_bs, CLIENT_LR, num_dummy_samples=10, strategy="start"),
                "Dimitrov (end)": DimitrovInference(epochs, client_bs, CLIENT_LR, num_dummy_samples=10, strategy="end"),
                "Dimitrov (avg)": DimitrovInference(epochs, client_bs, CLIENT_LR, num_dummy_samples=10, strategy="avg"),
            }
            for name, inference in [("Oracle", None), *methods.items()]:
                if name == "iDLG" and bs > batch.num_classes:  # iDLG reads one label per class: it cannot label a larger batch
                    continue
                base.target_inference = inference
                _, state, _ = runner.run(base, model, batch, device, oracle=inference is None, seed=a.seed + trial)
                state.artifacts[LabelEvaluator.GT_KEY] = batch.labels
                metrics = LabelEvaluator().evaluate(state.input, batch.images, state=state).metrics
                cell = rows.setdefault(f"bs={bs} / {name}", {"accuracy": [], "multiset_accuracy": []})
                for m in cell:
                    cell[m].append(metrics[m])
        print(f"batch size {bs}: " + ", ".join(f"{k.split(' / ')[1]} {sum(v['multiset_accuracy']) / len(v['multiset_accuracy']):.2f}"
                                                for k, v in rows.items() if k.startswith(f"bs={bs} ")))
    a.out.mkdir(parents=True, exist_ok=True)
    _plot(a.out, sizes, rows)
    return runner.finish(a, rows, batch_sizes=sizes, trials=trials, protocol=a.protocol)


if __name__ == "__main__":
    main()
