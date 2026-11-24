#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""The shared example helpers: CLI contract, results files and README rendering."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from examples.gia.common import cli, data, report, results


def test_cli_defaults_and_quick_device():
    a = cli.parse("d", ["--quick"])
    assert a.quick and a.seed == 1234 and cli.device_for(a).type == "cpu"
    assert a.out.parent.name == "outputs" and a.data_dir.name == "data"
    assert a.command.endswith("--quick")


def test_cli_extra_flags():
    a = cli.parse("d", ["--seed", "3"], extra=lambda p: p.add_argument("--trials", type=int, default=7))
    assert a.seed == 3 and a.trials == 7 and not a.quick


def test_quick_image_batch_is_synthetic_and_in_data_space(tmp_path: Path):
    b = data.image_batch(True, 3, tmp_path, torch.device("cpu"))
    assert b.images.shape == (3, 3, 16, 16) and b.labels.shape == (3,) and 0 <= b.images.min() <= b.images.max() <= 1


def test_results_roundtrip_and_summary(tmp_path: Path, capsys):
    rows = {"IG": {"psnr": [10.0, 12.0], "ssim": [0.2, 0.4]}}
    results.save(tmp_path, rows, {"command": "python -m x"})
    loaded = results.load(tmp_path)
    assert loaded["rows"]["IG"]["psnr"]["mean"] == 11.0
    assert set(loaded["env"]) == {"date", "commit", "torch", "device", "python", "deterministic"}
    results.emit(rows)
    line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("RESULT "))
    assert json.loads(line[7:])["IG.psnr"] == 11.0


def test_report_fills_marked_block(tmp_path: Path):
    (tmp_path / "ex").mkdir()
    results.save(tmp_path / "ex", {"IG": {"psnr": [10.0, 12.0]}}, {"command": "python -m x"})
    readme = tmp_path / "README.md"
    readme.write_text("a\n<!-- results:ex -->\nold\n<!-- /results -->\nb\n")
    report.render(readme, tmp_path)
    text = readme.read_text()
    assert "old" not in text and "11.00 ± 1.00 (n=2)" in text and text.startswith("a\n") and text.endswith("b\n")
    assert "Command: `python -m x`" in text


def test_report_leaves_a_block_without_results_unchanged(tmp_path: Path):
    (tmp_path / "ex").mkdir()
    results.save(tmp_path / "ex", {"IG": {"psnr": [10.0, 12.0]}}, {"command": "python -m x"})
    readme = tmp_path / "README.md"
    readme.write_text("<!-- results:ex -->\nold\n<!-- /results -->\n<!-- results:missing -->\nkeep me\n<!-- /results -->\n")
    with pytest.warns(UserWarning, match="missing"):
        report.render(readme, tmp_path)
    text = readme.read_text()
    assert "11.00 ± 1.00 (n=2)" in text and "old" not in text
    assert "<!-- results:missing -->\nkeep me\n<!-- /results -->" in text


def test_cli_without_plots_has_no_replot_flag():
    assert cli.parse("d", ["--replot"]).replot
    assert not hasattr(cli.parse("d", [], plots=False), "replot")
    with pytest.raises(SystemExit):
        cli.parse("d", ["--replot"], plots=False)


def test_report_renders_relative_paths_and_no_commit(tmp_path: Path):
    (tmp_path / "ex").mkdir()
    command = f"python -m x --out {cli.REPO}/examples/gia/outputs/ex --data-dir /somewhere/else/data"
    results.save(tmp_path / "ex", {"IG": {"psnr": [10.0]}}, {"command": command, "seed": 3, "values": [1, 2]})
    readme = tmp_path / "README.md"
    readme.write_text("<!-- results:ex -->\n<!-- /results -->\n")
    report.render(readme, tmp_path)
    text = readme.read_text()
    assert str(cli.REPO) not in text and "/somewhere" not in text
    assert "--out examples/gia/outputs/ex --data-dir data" in text
    saved = results.load(tmp_path / "ex")["env"]
    assert saved["date"] in text and f"torch `{saved['torch']}`" in text and "seed=3" in text
    assert "commit" not in text


def test_readme_result_blocks_name_registered_runs():
    import re

    from examples.gia.common import run_all

    for readme in cli.REPO.glob("examples/gia/**/README.md"):
        for name in re.findall(r"<!-- results:(\w[\w-]*) -->", readme.read_text()):
            assert name in run_all.FULL, f"{readme}: results block {name!r} is not a run_all entry"


def test_quick_images_are_not_drawn_from_the_attacks_seeded_stream(tmp_path: Path):
    # The attack seeds the global generator; images from that stream would equal its uniform starting point.
    batch = data.image_batch(True, 2, tmp_path, torch.device("cpu"))
    torch.manual_seed(1234)
    assert not torch.equal(batch.images, torch.rand(2, 3, 16, 16))


def test_quick_proxy_images_are_disjoint_from_the_private_batch(tmp_path: Path):
    private = data.image_batch(True, 8, tmp_path, torch.device("cpu"))
    proxy = data.image_batch(True, 64, tmp_path, torch.device("cpu"), start_idx=10_000)
    assert not any(torch.equal(p, q) for p in private.images for q in proxy.images)
    assert not torch.equal(private.images, proxy.images[:8])


# ── run_all ──────────────────────────────────────────────────────────────────────────────────────

def test_run_all_command_and_gpu_classification(tmp_path: Path):
    from examples.gia.common import run_all

    cmd = run_all.command("compare_attacks", tmp_path)
    assert cmd[:3] == ["nice", "-n", "10"] and cmd[4:6] == ["-m", "examples.gia.image_classification.compare_attacks"]
    assert cmd[cmd.index("--out") + 1] == str(tmp_path / "compare_attacks")
    assert run_all.needs_gpu(run_all.FULL["compare_attacks"][1])


def test_run_all_skip_logic(tmp_path: Path):
    from examples.gia.common import run_all

    assert not run_all.is_done(tmp_path, "abc")
    (tmp_path / "results.json").write_text(json.dumps({"env": {"commit": "abc"}}))
    assert run_all.is_done(tmp_path, "abc")
    assert not run_all.is_done(tmp_path, "def")
    assert run_all.is_done(tmp_path, "def", any_commit=True)
    (tmp_path / "results.json").write_text("not json")
    assert not run_all.is_done(tmp_path, "abc") and run_all.is_done(tmp_path, "abc", any_commit=True)


def test_run_all_does_not_count_stale_results_when_the_latest_attempt_failed(tmp_path: Path):
    from examples.gia.common import run_all

    (tmp_path / "results.json").write_text(json.dumps({"env": {"commit": "old"}}))
    (tmp_path / "run.status").write_text("failed (exit 1)\n")
    assert not run_all.is_done(tmp_path, "new", any_commit=True)
    (tmp_path / "run.status").write_text("ok\n")
    assert run_all.is_done(tmp_path, "new", any_commit=True)


def test_run_all_main_skips_waits_and_continues(tmp_path: Path, monkeypatch, capsys):
    from examples.gia.common import run_all

    (tmp_path / "quickstart").mkdir()
    (tmp_path / "quickstart/results.json").write_text(json.dumps({"env": {"commit": "HEAD0"}}))
    monkeypatch.setattr(run_all, "head_commit", lambda: "HEAD0")
    monkeypatch.setitem(run_all.FULL, "cpu_only", ("examples.gia.getting_started.quickstart", ["--device", "cpu"]))
    ran, waited = [], []
    monkeypatch.setattr(run_all, "wait_for_gpu", lambda gb, **k: waited.append(gb) or True)
    monkeypatch.setattr(run_all, "run_one", lambda name, root, t: ran.append(name) or (("failed (exit 1)", 1.0) if name == "fedavg" else ("ok", 1.0)))
    rc = run_all.main(["--only", "quickstart", "fedavg", "cpu_only", "--out-root", str(tmp_path)])
    assert ran == ["fedavg", "cpu_only"] and waited == [10.0]  # CPU-only entries skip the GPU wait
    assert rc == 1 and "skipped (done)" in capsys.readouterr().out


def test_run_all_gpu_guard(monkeypatch):
    from examples.gia.common import run_all

    monkeypatch.setattr(run_all, "free_gpu_gb", lambda: 3.0)
    monkeypatch.setattr(run_all.time, "sleep", lambda s: None)
    assert not run_all.wait_for_gpu(10.0, poll_s=60, max_wait_s=120)
    monkeypatch.setattr(run_all, "free_gpu_gb", lambda: None)
    assert run_all.wait_for_gpu(10.0)


def test_run_all_dry_run_and_unknown(capsys, tmp_path: Path):
    from examples.gia.common import run_all

    assert run_all.main(["--only", "quickstart", "--dry-run", "--out-root", str(tmp_path)]) == 0
    assert capsys.readouterr().out.count("\n") == 1
    with pytest.raises(SystemExit):
        run_all.main(["--only", "nope"])


AGPL_EXAMPLES = "examples.gia.coco_yolo_1_image."  # AGPL-licensed example folder


@pytest.mark.parametrize("name", [*__import__("examples.gia.common.run_all", fromlist=["FULL"]).FULL,
                                  *__import__("examples.gia.common.run_all", fromlist=["FULL"]).OPTIONAL])
def test_run_all_flags_parse_with_each_example(name: str, tmp_path: Path, monkeypatch):
    """Every full-run command line is accepted by the example's own argument parser."""
    import argparse
    import os
    import runpy
    import sys

    from examples.gia.common import run_all

    monkeypatch.setattr(run_all, "registry", lambda: {**run_all.FULL, **{n: v[:2] for n, v in run_all.OPTIONAL.items()}})
    cmd = run_all.command(name, tmp_path)
    module, argv = cmd[cmd.index("-m") + 1], cmd[cmd.index("-m") + 2:]
    if module.startswith(AGPL_EXAMPLES):  # parse in a child process so the AGPL script never loads into this one
        import subprocess

        code = ("import argparse, runpy, sys\n"
                "real = argparse.ArgumentParser.parse_args\n"
                "def parse_only(self, args=None, namespace=None):\n"
                "    real(self, args, namespace)\n"
                "    sys.exit(0)\n"
                "argparse.ArgumentParser.parse_args = parse_only\n"
                f"sys.argv = {[module, *argv]!r}\n"
                f"runpy.run_module({module!r}, run_name='__main__', alter_sys=True)\n"
                "sys.exit(3)  # the script returned without parsing its arguments\n")
        root = Path(__file__).resolve().parents[4]
        env = {**os.environ, "PYTHONPATH": str(root), "CUDA_VISIBLE_DEVICES": ""}
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root, env=env,
                             timeout=300, check=False)
        assert out.returncode == 0, out.stderr[-2000:]
        return

    class Parsed(Exception):
        pass

    real = argparse.ArgumentParser.parse_args

    def parse_only(self, args=None, namespace=None):
        real(self, args, namespace)
        raise Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_only)
    monkeypatch.setattr(sys, "argv", [module, *argv])
    with pytest.raises(Parsed):
        runpy.run_module(module, run_name="__main__", alter_sys=True)


def test_deterministic_flag_sets_torch_and_is_recorded(tmp_path: Path, monkeypatch):
    import os

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    before = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    try:
        assert not cli.parse("d", []).deterministic
        a = cli.parse("d", ["--deterministic", "--seed", "5"])
        assert a.deterministic and torch.are_deterministic_algorithms_enabled(), "flag turns deterministic algorithms on"
        assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        first = torch.rand(3)
        cli.set_deterministic(5)
        assert torch.equal(first, torch.rand(3)), "seeded"
        assert results.environment()["deterministic"] is True
    finally:
        torch.use_deterministic_algorithms(before[0])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = before[1], before[2]
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    assert results.environment()["deterministic"] is (before[0] and before[1]), "restored"


def test_run_all_commands_are_deterministic(tmp_path: Path):
    from examples.gia.common import run_all

    assert "--deterministic" in run_all.command("fedavg", tmp_path)
