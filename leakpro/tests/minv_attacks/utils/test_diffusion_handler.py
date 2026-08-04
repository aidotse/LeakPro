#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for Diff-MI diffusion component loading."""

from pathlib import Path
from typing import NoReturn

import pytest

from leakpro.attacks.utils.diff_mi.setup import PreTrainConfig
from leakpro.attacks.utils.diffusion_handler import DiffMiHandler
from leakpro.input_handler.user_imports import get_class_from_module, import_module_from_file


def _handler(module_path: str, model_class: str = "UNet", diffusion_class: str = "SpacedDiffusion") -> DiffMiHandler:
    handler = DiffMiHandler.__new__(DiffMiHandler)
    handler.diffusion_path = module_path
    handler.diffusion_model_class = model_class
    handler.gaussian_diffusion_class = diffusion_class
    return handler


def test_resolved_module_path_imports_configured_components(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A resolvable module_path should be used instead of the built-in factory path."""
    module_path = tmp_path / "diffusion_components.py"
    module_path.write_text(
        "class UNet:\n"
        "    def __init__(self, **kwargs):\n"
        "        self.kwargs = kwargs\n\n"
        "class SpacedDiffusion:\n"
        "    def __init__(self, **kwargs):\n"
        "        self.kwargs = kwargs\n"
    )

    def fail_factory(**_kwargs: object) -> NoReturn:
        raise AssertionError("resolved module_path should not use built-in factories")

    monkeypatch.setattr("leakpro.attacks.utils.diffusion_handler.create_model", fail_factory)
    monkeypatch.setattr("leakpro.attacks.utils.diffusion_handler.create_gaussian_diffusion", fail_factory)

    handler = _handler(str(module_path))
    handler._init_model(PreTrainConfig())

    assert handler.diffusion_model.__class__.__name__ == "UNet"
    assert handler.diffusion.__class__.__name__ == "SpacedDiffusion"


def test_missing_builtin_module_path_falls_back_to_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing paths may fall back only when the configured names are built-in aliases."""
    calls = {"model": None, "diffusion": None}

    def fake_create_model(**kwargs: object) -> str:
        calls["model"] = kwargs
        return "model"

    def fake_create_gaussian_diffusion(**kwargs: object) -> str:
        calls["diffusion"] = kwargs
        return "diffusion"

    monkeypatch.setattr("leakpro.attacks.utils.diffusion_handler.create_model", fake_create_model)
    monkeypatch.setattr("leakpro.attacks.utils.diffusion_handler.create_gaussian_diffusion", fake_create_gaussian_diffusion)

    handler = _handler("./does/not/exist.py")
    handler._init_model(PreTrainConfig())

    assert handler.diffusion_model == "model"
    assert handler.diffusion == "diffusion"
    assert calls["model"] is not None
    assert calls["diffusion"] is not None


def test_resolved_module_path_with_missing_class_raises(tmp_path: Path) -> None:
    """A resolvable module_path should not silently fall back when classes are missing."""
    module_path = tmp_path / "diffusion_components.py"
    module_path.write_text("class SomethingElse:\n    pass\n")

    handler = _handler(str(module_path))

    with pytest.raises(ValueError, match="Failed to import custom diffusion components"):
        handler._init_model(PreTrainConfig())


def test_internal_diffusion_module_exposes_builtin_aliases() -> None:
    """The package diffusion.py path can be used as a module_path for built-in names."""
    module = import_module_from_file("leakpro/attacks/utils/diff_mi/diffusion.py")

    assert get_class_from_module(module, "UNet").__name__ == "UNet"
    assert get_class_from_module(module, "SpacedDiffusion").__name__ == "SpacedDiffusion"
