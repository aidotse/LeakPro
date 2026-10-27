#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for building a recipe from an architecture file (CPU, tiny models)."""

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from leakpro.input_handler.user_imports import import_module_from_file
from leakpro.optimization.adapters import detect_binary, find_model_class, recipe_from_module

MULTICLASS = """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(6, num_classes)
    def forward(self, x):
        return self.fc(x)
"""

BINARY = """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6, 1)
    def forward(self, x):
        return self.fc(x)
"""

TWO_MODELS = MULTICLASS + """
class Other(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6, 2)
    def forward(self, x):
        return self.fc(x)
"""


@pytest.fixture()
def arch(tmp_path):  # noqa: ANN001, ANN201
    def write(source: str):  # noqa: ANN202
        path = tmp_path / "arch.py"
        path.write_text(source)
        return path
    return write


def _xy(n: int = 200, dim: int = 6, classes: int = 3, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    x = torch.tensor(rng.normal(size=(n, dim)), dtype=torch.float32)
    y = torch.tensor(rng.integers(0, classes, size=n))
    return x, y


def _dataset(classes: int = 3, n: int = 200) -> TensorDataset:
    x, y = _xy(n=n, classes=classes)
    return TensorDataset(x, y.float() if classes == 2 else y.long())


class TestModelDiscovery:
    def test_single_class_is_found_without_naming_it(self, arch):  # noqa: ANN001, ANN201
        assert find_model_class(import_module_from_file(str(arch(MULTICLASS)))).__name__ == "Net"

    def test_ambiguous_module_takes_the_first_defined_unless_named(self, arch, caplog):  # noqa: ANN001, ANN201
        module = import_module_from_file(str(arch(TWO_MODELS)))
        assert find_model_class(module).__name__ == "Net"
        assert "using the first" in caplog.text
        assert find_model_class(module, "Other").__name__ == "Other"

    def test_recorded_init_params_pick_the_class_that_accepts_them(self, arch):  # noqa: ANN001, ANN201
        module = import_module_from_file(str(arch(TWO_MODELS)))
        assert find_model_class(module, init_params={"num_classes": 10}).__name__ == "Net"

    def test_imported_classes_are_never_candidates(self, tmp_path):  # noqa: ANN001, ANN201
        """Issue #484: an imported block (torchvision's BasicBlock) is not the user's model."""
        path = tmp_path / "arch_imports.py"
        path.write_text("from torch.nn import Linear\nimport torch.nn as nn\n"
                        "class Net(nn.Module):\n    def __init__(self): super().__init__(); self.fc = Linear(2, 2)\n")
        assert find_model_class(import_module_from_file(str(path))).__name__ == "Net"

    def test_unknown_class_name_rejected(self, arch):  # noqa: ANN001, ANN201
        # user_imports.get_class_from_module raises ValueError, not KeyError.
        with pytest.raises(ValueError, match="not found in module"):
            find_model_class(import_module_from_file(str(arch(MULTICLASS))), "Missing")

    def test_missing_file_rejected(self, tmp_path):  # noqa: ANN001, ANN201
        with pytest.raises(FileNotFoundError):
            import_module_from_file(str(tmp_path / "nope.py"))

    def test_repeated_loads_reuse_the_cached_module(self, arch):  # noqa: ANN001, ANN201
        # user_imports.import_module_from_file caches by module name and checks
        # the path, so the same arch loaded twice is one import, not two.
        path = arch(MULTICLASS)
        assert import_module_from_file(str(path)) is import_module_from_file(str(path))


class TestDetectBinary:
    def test_single_logit_head(self, arch):  # noqa: ANN001, ANN201
        x, _ = _xy()
        assert detect_binary(arch(BINARY), x) is True

    def test_multiclass_head(self, arch):  # noqa: ANN001, ANN201
        x, _ = _xy()
        assert detect_binary(arch(MULTICLASS), x) is False


class TestRecipeFromModule:
    def test_multiclass_recipe_shape(self, arch):  # noqa: ANN001, ANN201
        recipe = recipe_from_module(arch(MULTICLASS), _dataset(), epochs=1)
        assert recipe.output_kind == "logits"
        assert isinstance(recipe.criterion, torch.nn.CrossEntropyLoss)

    def test_binary_recipe_uses_bce_on_raw_logits(self, arch):  # noqa: ANN001, ANN201
        # A single-logit head is still a logit: BCEWithLogitsLoss applies the
        # sigmoid itself, so output_kind stays "logits" (PETRecipe rejects the
        # old "binary_logits" spelling).
        recipe = recipe_from_module(arch(BINARY), _dataset(classes=2), epochs=1, binary=True)
        assert recipe.output_kind == "logits"
        assert isinstance(recipe.criterion, torch.nn.BCEWithLogitsLoss)

    def test_each_call_returns_an_untrained_model(self, arch):  # noqa: ANN001, ANN201
        # A search that resumed a trained model would silently measure the
        # wrong thing, so make_model must never hand back the same object.
        recipe = recipe_from_module(arch(MULTICLASS), _dataset(), epochs=1)
        assert recipe.make_model({}) is not recipe.make_model({})

    def test_loader_honours_sampled_batch_size_and_indices(self, arch):  # noqa: ANN001, ANN201
        recipe = recipe_from_module(arch(MULTICLASS), _dataset(), epochs=1)
        loader = recipe.make_loader(np.arange(64), {"batch_size": 16})
        assert next(iter(loader))[0].shape[0] == 16
        assert len(loader.dataset) == 64

    def test_loader_reads_through_the_dataset_getitem(self, arch):  # noqa: ANN001, ANN201
        # The recipe must not reach into raw arrays: a dataset that normalises
        # in __getitem__ (the webapp's ArrayDataset does) has to feed training
        # the same tensors the audit will see.
        class Scaled(TensorDataset):
            def __getitem__(self, idx):  # noqa: ANN001, ANN204
                x, y = super().__getitem__(idx)
                return x * 0.0 + 7.0, y

        x, y = _xy()
        recipe = recipe_from_module(arch(MULTICLASS), Scaled(x, y.long()), epochs=1)
        batch, _ = next(iter(recipe.make_loader(np.arange(8), {"batch_size": 8})))
        assert torch.all(batch == 7.0)

    def test_optimizer_honours_sampled_learning_rate(self, arch):  # noqa: ANN001, ANN201
        recipe = recipe_from_module(arch(MULTICLASS), _dataset(), epochs=1, optimizer_name="sgd")
        opt = recipe.make_optimizer(recipe.make_model({}).parameters(), {"learning_rate": 0.07})
        assert isinstance(opt, torch.optim.SGD)
        assert opt.param_groups[0]["lr"] == pytest.approx(0.07)

    def test_init_params_reach_the_constructor(self, arch):  # noqa: ANN001, ANN201
        recipe = recipe_from_module(arch(MULTICLASS), _dataset(), epochs=1, init_params={"num_classes": 5})
        assert recipe.make_model({}).fc.out_features == 5

    def test_unknown_optimizer_rejected(self, arch):  # noqa: ANN001, ANN201
        # Every torch.optim optimizer is valid now that the lookup comes from
        # user_imports.get_optimizer_mapping, so "rmsprop" is accepted; only a
        # name torch does not have is rejected.
        assert recipe_from_module(arch(MULTICLASS), _dataset(), epochs=1, optimizer_name="rmsprop")
        with pytest.raises(ValueError, match="Unknown optimizer"):
            recipe_from_module(arch(MULTICLASS), _dataset(), epochs=1, optimizer_name="not_an_optimizer")
