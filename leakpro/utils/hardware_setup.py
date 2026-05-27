#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Hardware detection and dependency-install helper.

Run interactively via ``python -m leakpro.utils.hardware_setup`` (use
``--install`` to actually install the detected platform's extras, or
``--print-only`` for a dry-run listing).

This module is intentionally torch-free at import time. It probes the
filesystem for the Habana plugin's pinned torch version and reports actionable
install commands even when torch itself can't be imported (the common case
when a previous ``pip install`` pulled in a torch incompatible with the
installed Habana stack).
"""
from __future__ import annotations

import argparse
import os
import shutil
import site
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class PlatformProfile:
    """Required pip packages for a detected hardware platform."""

    name: str
    description: str
    pip_packages: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


CUDA_PROFILE = PlatformProfile(
    name="cuda",
    description="NVIDIA CUDA GPU detected.",
    pip_packages=[],
    notes=[
        "PyTorch's CUDA build is expected to be installed already; if not, "
        "follow https://pytorch.org/get-started/locally/ for the correct "
        "command for your CUDA version.",
    ],
)

CPU_PROFILE = PlatformProfile(
    name="cpu",
    description="No accelerator detected — running on CPU.",
    pip_packages=[],
    notes=["LeakPro will run on CPU. Expect significantly longer runtimes."],
)


def _find_habana_required_torch() -> Optional[Tuple[str, Path]]:
    """Return ``(version_str, source_path)`` for the Habana plugin's torch pin, if found."""
    candidates = [Path(p) for p in site.getsitepackages()]
    candidates.append(Path(site.getusersitepackages()))
    for sp in candidates:
        path = sp / "habana_frameworks" / "torch" / "required_version.txt"
        if path.exists():
            try:
                return path.read_text().strip(), path
            except OSError:
                continue
    return None


def _torch_status() -> Tuple[Optional[str], Optional[str]]:
    """Return ``(version, import_error)`` describing the local torch install."""
    try:
        import torch  # type: ignore[import-not-found]  # noqa: PLC0415

        return getattr(torch, "__version__", "?"), None
    except Exception as exc:  # noqa: BLE001 - torch failures show up in many flavors
        return None, f"{type(exc).__name__}: {exc}"


def _build_hpu_profile() -> PlatformProfile:
    """Build the HPU profile, parameterized by the on-disk Habana plugin's torch pin."""
    required = _find_habana_required_torch()
    pip_packages = ["lightning-habana"]
    notes: List[str] = []

    if required is not None:
        version_pin, source = required
        try:
            major, minor = version_pin.split(".")[:2]
            spec = f"torch>={major}.{minor},<{major}.{int(minor) + 1}"
        except (ValueError, IndexError):
            spec = f"torch=={version_pin}"
        notes.append(
            f"Habana plugin pins torch {version_pin} (per {source}). "
            f"Install with: `pip install '{spec}'` (upstream CPU build) "
            "and set `PT_HPU_LAZY_MODE=0` to enable HPU in eager mode."
        )
        notes.append(
            "For lazy-mode (best performance) you need Habana's torch fork "
            "from https://docs.habana.ai/en/latest/Installation_Guide/index.html"
        )
    else:
        notes.append(
            "habana-torch-plugin and the matching SynapseAI runtime must be "
            "installed from Habana's index; see "
            "https://docs.habana.ai/en/latest/Installation_Guide/index.html"
        )

    notes.append(
        "Once torch is installed, leakpro.utils.device.mark_step() is wired "
        "into LeakPro's built-in attack loops after optimizer.step()."
    )

    return PlatformProfile(
        name="hpu",
        description="Habana Gaudi (HPU) detected.",
        pip_packages=pip_packages,
        notes=notes,
    )


def detect_profile() -> PlatformProfile:
    """Return the install profile for the detected platform.

    Detection order: HPU (via filesystem check + ``habana_frameworks.torch.hpu``) → CUDA → CPU.
    Falls back to the HPU profile (with broken-state notes) if the Habana plugin is on disk
    but ``import torch`` is failing — that's the case we most want to diagnose.
    """
    habana_required = _find_habana_required_torch()
    torch_version, torch_err = _torch_status()

    if habana_required is not None and torch_err is not None:
        profile = _build_hpu_profile()
        broken_notes = [
            f"Torch import is currently failing: {torch_err}",
            "This typically means the installed torch doesn't match the Habana "
            "plugin's required version (see note above).",
        ]
        return PlatformProfile(
            name=profile.name,
            description="Habana Gaudi (HPU) stack on disk, but torch is broken/missing.",
            pip_packages=profile.pip_packages,
            notes=broken_notes + profile.notes,
        )

    if torch_err is None:
        try:
            from leakpro.utils.device import get_device  # noqa: PLC0415 - keep torch dep out of module load

            device = get_device()
        except Exception as exc:  # noqa: BLE001
            return PlatformProfile(
                name="cpu",
                description=f"Device detection failed: {exc}",
                notes=["Falling back to CPU profile."],
            )
        if device.type == "hpu":
            return _build_hpu_profile()
        if device.type == "cuda":
            return CUDA_PROFILE
        return CPU_PROFILE

    # No Habana plugin AND torch is broken/missing -> generic CPU profile + an FYI note.
    notes = list(CPU_PROFILE.notes)
    if torch_err is not None:
        notes = [f"torch is not importable: {torch_err}", *notes]
    return PlatformProfile(
        name="cpu",
        description=CPU_PROFILE.description,
        pip_packages=CPU_PROFILE.pip_packages,
        notes=notes,
    )


def format_report(profile: PlatformProfile) -> str:
    """Render a human-readable summary of detection + recommended actions."""
    lines = [f"[LeakPro] {profile.description}"]
    torch_version, _ = _torch_status()
    if torch_version is not None:
        lines.append(f"  torch: {torch_version}")
    if profile.pip_packages:
        lines.append("  Recommended pip packages:")
        lines.extend(f"    - {pkg}" for pkg in profile.pip_packages)
    for note in profile.notes:
        lines.append(f"  Note: {note}")
    return "\n".join(lines)


def install_profile(profile: PlatformProfile, pip_executable: Optional[str] = None) -> int:
    """Install the pip packages for ``profile``. Returns the pip exit code (0 on success or no-op)."""
    if not profile.pip_packages:
        sys.stdout.write(f"No extra pip packages required for platform {profile.name}.\n")
        return 0
    pip = pip_executable or shutil.which("pip") or shutil.which("pip3")
    if pip is None:
        sys.stderr.write("Could not locate pip executable; aborting install.\n")
        return 1
    cmd = [pip, "install", *profile.pip_packages]
    sys.stdout.write("Running: " + " ".join(cmd) + "\n")
    return subprocess.call(cmd)  # noqa: S603 - args are constructed from a fixed allow-list


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m leakpro.utils.hardware_setup",
        description="Detect the local accelerator and report or install its required packages.",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Actually run `pip install` for the detected platform.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Just print the detection report (default behaviour).",
    )
    args = parser.parse_args(argv)

    profile = detect_profile()
    sys.stdout.write(format_report(profile) + "\n")

    if args.install and not args.print_only:
        return install_profile(profile)
    return 0


if __name__ == "__main__":
    # Avoid triggering torch's autoload hook when imported as a script on a
    # mismatched Habana env -- the detection logic below does its own probing.
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    raise SystemExit(main())
