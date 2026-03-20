"""Tests for input handler import utilities."""

from pathlib import Path
import sys

import pytest

from leakpro.input_handler.user_imports import (
    get_class_from_module,
    get_criterion_mapping,
    get_optimizer_mapping,
    import_module_from_file,
)


def _write_module(tmp_path: Path, filename: str, source: str) -> Path:
    path = tmp_path / filename
    path.write_text(source, encoding="utf-8")
    return path


def test_import_module_from_file_missing_path() -> None:
    with pytest.raises(FileNotFoundError):
        import_module_from_file("/tmp/does_not_exist_abc123.py")


def test_import_module_registers_and_reuses_same_module(tmp_path: Path) -> None:
    module_path = _write_module(
        tmp_path,
        "user_imports_unique_mod.py",
        "VALUE = 7\n\nclass DemoClass:\n    pass\n",
    )

    module_a = import_module_from_file(str(module_path))
    module_b = import_module_from_file(str(module_path))

    assert module_a is module_b
    assert module_a.__name__ in sys.modules
    assert sys.modules[module_a.__name__] is module_a
    assert get_class_from_module(module_a, "DemoClass") is module_a.DemoClass


def test_import_module_same_name_different_path_replaces_sys_modules(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    path_a = _write_module(
        dir_a,
        "duplicate_name_mod.py",
        "VALUE = 1\n\nclass DemoClass:\n    pass\n",
    )
    path_b = _write_module(
        dir_b,
        "duplicate_name_mod.py",
        "VALUE = 2\n\nclass DemoClass:\n    pass\n",
    )

    module_a = import_module_from_file(str(path_a))
    module_b = import_module_from_file(str(path_b))

    assert module_a is not module_b
    assert module_a.__name__ == module_b.__name__ == "duplicate_name_mod"
    assert module_a.VALUE == 1
    assert module_b.VALUE == 2
    assert sys.modules["duplicate_name_mod"] is module_b


def test_get_class_from_module_missing_class_raises(tmp_path: Path) -> None:
    module_path = _write_module(tmp_path, "class_lookup_mod.py", "class Existing:\n    pass\n")
    module = import_module_from_file(str(module_path))

    with pytest.raises(ValueError, match="Class Missing not found"):
        get_class_from_module(module, "Missing")


def test_mappings_contain_common_entries() -> None:
    optimizer_mapping = get_optimizer_mapping()
    criterion_mapping = get_criterion_mapping()

    assert "sgd" in optimizer_mapping
    assert "adam" in optimizer_mapping
    assert "crossentropyloss" in criterion_mapping
    assert "mseloss" in criterion_mapping
