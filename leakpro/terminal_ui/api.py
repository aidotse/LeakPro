"""API layer for the LeakPro terminal UI."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _ensure_examples_in_path() -> None:
    """Ensure examples directory is in sys.path for imports."""
    project_root = Path(__file__).parent.parent.parent.resolve()
    examples_dir = project_root / "examples"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Support older pickles that reference cifar_handler without package prefix.
    try:
        from examples.mia.cifar import cifar_handler as _cifar_handler

        if "cifar_handler" not in sys.modules:
            sys.modules["cifar_handler"] = _cifar_handler
    except Exception:
        return


@dataclass
class ConfigPaths:
    train_config: Path
    audit_config: Path


class CifarMIAApi:
    """Programmatic API for the CIFAR MIA guided flow."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

    def resolve_path(self, path: str | Path) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path
        return (self.base_dir / path).resolve()

    def load_yaml(self, path: str | Path) -> dict:
        resolved = self.resolve_path(path)
        with resolved.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def set_random_seed(self, seed: int) -> None:
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            return

    def prepare_population_dataset(self, train_config: dict, save_if_missing: bool = True) -> dict:
        import pickle
        import torch
        from torchvision.datasets import CIFAR10, CIFAR100

        _ensure_examples_in_path()
        from examples.mia.cifar.cifar_handler import CifarInputHandler

        dataset_name = train_config["data"]["dataset"]
        root = self.resolve_path(train_config["data"]["data_dir"])
        root.mkdir(parents=True, exist_ok=True)

        if dataset_name == "cifar10":
            trainset = CIFAR10(root=root, train=True, download=True)
            testset = CIFAR10(root=root, train=False, download=True)
        elif dataset_name == "cifar100":
            trainset = CIFAR100(root=root, train=True, download=True)
            testset = CIFAR100(root=root, train=False, download=True)
        else:
            raise ValueError("Unknown dataset type")

        train_data = torch.tensor(trainset.data).permute(0, 3, 1, 2).float() / 255
        test_data = torch.tensor(testset.data).permute(0, 3, 1, 2).float() / 255

        data = torch.cat([train_data.clone().detach(), test_data.clone().detach()], dim=0)
        targets = torch.cat([torch.tensor(trainset.targets), torch.tensor(testset.targets)], dim=0)

        population_dataset = CifarInputHandler.UserDataset(data, targets)

        dataset_path = self.resolve_path(Path("data") / f"{dataset_name}.pkl")
        dataset_path.parent.mkdir(parents=True, exist_ok=True)

        if save_if_missing and not dataset_path.exists():
            with dataset_path.open("wb") as handle:
                pickle.dump(population_dataset, handle)

        return {
            "population_dataset": population_dataset,
            "data": data,
            "targets": targets,
            "dataset_name": dataset_name,
            "dataset_path": dataset_path,
        }

    def load_population_dataset(self, dataset_path: Path):
        import joblib

        _ensure_examples_in_path()
        with dataset_path.open("rb") as handle:
            return joblib.load(handle)

    def split_train_test(self, train_config: dict, data: Any, targets: Any) -> dict:
        import numpy as np
        from sklearn.model_selection import train_test_split
        from torch.utils.data import DataLoader

        _ensure_examples_in_path()
        from examples.mia.cifar.cifar_handler import CifarInputHandler

        train_fraction = train_config["data"]["f_train"]
        test_fraction = train_config["data"]["f_test"]
        batch_size = train_config["train"]["batch_size"]

        dataset_size = len(data)
        train_size = int(train_fraction * dataset_size)
        test_size = int(test_fraction * dataset_size)

        selected_index = np.random.choice(np.arange(dataset_size), train_size + test_size, replace=False)
        train_indices, test_indices = train_test_split(selected_index, test_size=test_size)

        train_subset = CifarInputHandler.UserDataset(data[train_indices], targets[train_indices])
        test_subset = CifarInputHandler.UserDataset(data[test_indices], targets[test_indices], **train_subset.return_params())

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        return {
            "train_loader": train_loader,
            "test_loader": test_loader,
            "train_indices": train_indices,
            "test_indices": test_indices,
            "train_subset": train_subset,
            "test_subset": test_subset,
        }

    def _load_target_model_class(self, module_path: Path, class_name: str):
        import importlib.util
        import sys

        module_path = self.resolve_path(module_path)
        spec = importlib.util.spec_from_file_location("leakpro_target_model", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if not hasattr(module, class_name):
            raise AttributeError(f"Class {class_name} not found in {module_path}")
        return getattr(module, class_name)

    def train_target_model(
        self,
        train_config: dict,
        audit_config: dict,
        train_loader,
        test_loader,
        dataset_name: str,
    ) -> dict:
        import pickle

        import torch
        from torch import nn, optim

        _ensure_examples_in_path()
        from examples.mia.cifar.cifar_handler import CifarInputHandler

        model_class = audit_config["target"]["model_class"]
        module_path = audit_config["target"]["module_path"]
        model_cls = self._load_target_model_class(module_path, model_class)

        if dataset_name == "cifar10":
            num_classes = 10
        elif dataset_name == "cifar100":
            num_classes = 100
        else:
            raise ValueError("Invalid dataset name")

        if model_class == "WideResNet":
            model = model_cls(depth=28, num_classes=num_classes, widen_factor=2)
        else:
            model = model_cls(num_classes=num_classes)

        lr = train_config["train"]["learning_rate"]
        weight_decay = train_config["train"]["weight_decay"]
        epochs = train_config["train"]["epochs"]
        optimizer_name = train_config["train"].get("optimizer", "SGD").upper()

        criterion = nn.CrossEntropyLoss()
        if optimizer_name == "ADAM":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "SGD":
            momentum = train_config["train"].get("momentum", 0.0)
            nesterov = train_config["train"].get("nesterov", False)
            optimizer = optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=nesterov,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        handler = CifarInputHandler()
        train_result = handler.train(
            dataloader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epochs=epochs,
        )
        test_result = handler.eval(test_loader, train_result.model, criterion)

        log_dir = self.resolve_path(train_config["run"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        target_path = log_dir / "target_model.pkl"
        with target_path.open("wb") as handle:
            torch.save(train_result.model.state_dict(), handle)

        return {
            "model": train_result.model,
            "train_result": train_result,
            "test_result": test_result,
            "optimizer": optimizer,
            "criterion": criterion,
            "epochs": epochs,
            "target_path": target_path,
        }

    def create_metadata(
        self,
        train_result,
        optimizer,
        criterion,
        train_loader,
        test_result,
        epochs: int,
        train_indices,
        test_indices,
        dataset_name: str,
        output_dir: Path,
    ) -> dict:
        import pickle

        from leakpro import LeakPro

        meta_data = LeakPro.make_mia_metadata(
            train_result=train_result,
            optimizer=optimizer,
            loss_fn=criterion,
            dataloader=train_loader,
            test_result=test_result,
            epochs=epochs,
            train_indices=train_indices,
            test_indices=test_indices,
            dataset_name=dataset_name,
        )

        output_dir = self.resolve_path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = output_dir / "model_metadata.pkl"
        with metadata_path.open("wb") as handle:
            pickle.dump(meta_data, handle)

        return {"metadata": meta_data, "metadata_path": metadata_path}

    def run_audit(self, audit_config_path: Path, create_pdf: bool = True) -> list[Any]:
        import yaml

        from leakpro import LeakPro

        _ensure_examples_in_path()
        from examples.mia.cifar.cifar_handler import CifarInputHandler

        resolved_audit_path = self.resolve_path(audit_config_path)

        with resolved_audit_path.open("r") as f:
            audit_config = yaml.safe_load(f)

        target_config = audit_config.get("target", {})
        if "module_path" in target_config:
            target_config["module_path"] = str(self.resolve_path(target_config["module_path"]))
        if "target_folder" in target_config:
            target_config["target_folder"] = str(self.resolve_path(target_config["target_folder"]))
        if "data_path" in target_config:
            target_config["data_path"] = str(self.resolve_path(target_config["data_path"]))

        with resolved_audit_path.open("w") as f:
            yaml.dump(audit_config, f)

        try:
            leakpro = LeakPro(CifarInputHandler, str(resolved_audit_path))
            return leakpro.run_audit(create_pdf=create_pdf)
        finally:
            with resolved_audit_path.open("w") as f:
                yaml.dump(audit_config, f)
