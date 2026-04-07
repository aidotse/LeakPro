"""Minimal real end-to-end tests for all MIA attacks in AttackFactoryMIA."""

from __future__ import annotations

import pickle
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader

from leakpro import LeakPro
from leakpro.attacks.mia_attacks.attack_factory_mia import AttackFactoryMIA
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.schemas import EvalOutput, TrainingOutput

N_SAMPLES = 72
TRAIN_FRACTION = 0.45
TEST_FRACTION = 0.45
N_TRAIN = int(round(N_SAMPLES * TRAIN_FRACTION))
N_TEST = int(round(N_SAMPLES * TEST_FRACTION))
if N_TRAIN + N_TEST >= N_SAMPLES:
    N_TEST = N_SAMPLES - N_TRAIN - 1
TRAIN_INDICES = list(range(0, N_TRAIN))
TEST_INDICES = list(range(N_TRAIN, N_TRAIN + N_TEST))
UNUSED_INDICES = list(range(N_TRAIN + N_TEST, N_SAMPLES))
# RMIA samples z-points from the full population and indexes cached target logits
# by those population indices. The E2E target split covers the full tiny
# population to keep population indices aligned with cached logits.
RMIA_N_TRAIN = N_SAMPLES // 2
RMIA_TRAIN_INDICES = list(range(0, RMIA_N_TRAIN))
RMIA_TEST_INDICES = list(range(RMIA_N_TRAIN, N_SAMPLES))
REPO_ROOT = Path(__file__).resolve().parents[4]


class TinyImageTargetModel(nn.Module):
    """Tiny image classifier for E2E tests."""

    def __init__(self, in_channels: int = 3, image_size: int = 8, num_classes: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_channels * image_size * image_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.flatten(x))


class TinyTimeSeriesTargetModel(nn.Module):
    """Tiny forecaster-like model for DTS E2E tests."""

    def __init__(self, input_size: int = 2, horizon: int = 2) -> None:
        super().__init__()
        self.input_size = input_size
        self.horizon = horizon
        self.linear = nn.Linear(input_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last_step = x[:, -1:, :].repeat(1, self.horizon, 1)
        return self.linear(last_step)


class TinyImageInputHandler(AbstractInputHandler):
    """Minimal CIFAR-like handler for image attacks."""

    def train(
        self,
        dataloader: DataLoader,
        model: nn.Module = None,
        criterion: nn.Module = None,
        optimizer: optim.Optimizer = None,
        epochs: int = None,
    ) -> TrainingOutput:
        if epochs is None:
            epochs = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for _ in range(epochs):
            for data, target in dataloader:
                data = data.to(device, non_blocking=True)
                target = target.long().to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(target)
                total_correct += (output.argmax(dim=1) == target).sum().item()
                total_samples += len(target)
        model.to("cpu")
        metrics = EvalOutput(accuracy=total_correct / max(total_samples, 1), loss=total_loss / max(total_samples, 1))
        return TrainingOutput(model=model, metrics=metrics)

    def eval(self, dataloader: DataLoader, model: nn.Module, criterion: nn.Module) -> EvalOutput:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for data, target in dataloader:
                data = data.to(device)
                target = target.long().to(device)
                output = model(data)
                total_loss += criterion(output, target).item() * len(target)
                total_correct += (output.argmax(dim=1) == target).sum().item()
                total_samples += len(target)
        model.to("cpu")
        return EvalOutput(accuracy=total_correct / max(total_samples, 1), loss=total_loss / max(total_samples, 1))

    class UserDataset(AbstractInputHandler.UserDataset):
        """Tiny image dataset with CIFAR-style mean/std attributes."""

        def __init__(self, data: torch.Tensor, targets: torch.Tensor, **kwargs: dict) -> None:
            self.data = data.float()
            self.targets = targets.long()
            self.augment_strength = kwargs.pop("augment_strength", "none")
            self.augment = kwargs.pop("augment", None)
            self.erase_post_norm = kwargs.pop("erase_post_norm", None)

            for key, value in kwargs.items():
                setattr(self, key, value)

            if not hasattr(self, "mean"):
                self.mean = self.data.mean(dim=(0, 2, 3)).view(-1, 1, 1)
            if not hasattr(self, "std"):
                self.std = self.data.std(dim=(0, 2, 3)).clamp_min(1e-6).view(-1, 1, 1)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            x = self.data[idx]
            y = self.targets[idx]
            if self.augment is not None:
                x = self.augment(x)
            x = (x - self.mean) / self.std
            if self.erase_post_norm is not None:
                x = self.erase_post_norm(x)
            return x, y

        def __len__(self) -> int:
            return len(self.targets)


class TinyTimeSeriesInputHandler(AbstractInputHandler):
    """Minimal time-series handler for DTS attack."""

    def train(
        self,
        dataloader: DataLoader,
        model: nn.Module = None,
        criterion: nn.Module = None,
        optimizer: optim.Optimizer = None,
        epochs: int = None,
    ) -> TrainingOutput:
        if epochs is None:
            epochs = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.train()
        total_loss = 0.0
        total_samples = 0
        for _ in range(epochs):
            for data, target in dataloader:
                data = data.float().to(device, non_blocking=True)
                target = target.float().to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(target)
                total_samples += len(target)
        model.to("cpu")
        metrics = EvalOutput(accuracy=0.5, loss=total_loss / max(total_samples, 1))
        return TrainingOutput(model=model, metrics=metrics)

    def eval(self, dataloader: DataLoader, model: nn.Module, criterion: nn.Module) -> EvalOutput:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for data, target in dataloader:
                data = data.float().to(device)
                target = target.float().to(device)
                output = model(data)
                total_loss += criterion(output, target).item() * len(target)
                total_samples += len(target)
        model.to("cpu")
        return EvalOutput(accuracy=0.5, loss=total_loss / max(total_samples, 1))

    class UserDataset(AbstractInputHandler.UserDataset):
        """Tiny time-series dataset."""

        def __init__(self, data: torch.Tensor, targets: torch.Tensor, **kwargs: dict) -> None:
            self.data = data.float()
            self.targets = targets.float()
            for key, value in kwargs.items():
                setattr(self, key, value)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            return self.data[idx], self.targets[idx]

        def __len__(self) -> int:
            return len(self.targets)


def _set_seed(seed: int = 1234) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _clear_global_attack_cache() -> None:
    cache_dirs = [
        Path("leakpro_output") / "attack_cache",
        REPO_ROOT / "leakpro_output" / "attack_cache",
    ]
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)


def _snapshot_optuna_metadata() -> dict[str, dict[str, Any]]:
    """Capture AttackConfig field optuna metadata for all MIA attacks.

    Some attack setup paths mutate pydantic field metadata (json_schema_extra)
    at class scope. Snapshot/restore keeps tests order-independent.
    """
    snapshot: dict[str, dict[str, Any]] = {}
    for attack_name, attack_cls in AttackFactoryMIA.attack_classes.items():
        snapshot[attack_name] = {
            field_name: deepcopy(field.json_schema_extra)
            for field_name, field in attack_cls.AttackConfig.model_fields.items()
        }
    return snapshot


def _restore_optuna_metadata(snapshot: dict[str, dict[str, Any]]) -> None:
    """Restore AttackConfig field optuna metadata for all MIA attacks."""
    for attack_name, field_map in snapshot.items():
        attack_cls = AttackFactoryMIA.attack_classes[attack_name]
        for field_name, extra in field_map.items():
            attack_cls.AttackConfig.model_fields[field_name].json_schema_extra = deepcopy(extra)


def _build_tiny_image_population() -> TinyImageInputHandler.UserDataset:
    data = torch.rand(N_SAMPLES, 3, 8, 8)
    targets = (torch.arange(N_SAMPLES) % 3).long()
    return TinyImageInputHandler.UserDataset(data, targets)


def _build_tiny_timeseries_population() -> TinyTimeSeriesInputHandler.UserDataset:
    data = torch.rand(N_SAMPLES, 4, 2)
    targets = torch.rand(N_SAMPLES, 2, 2)
    return TinyTimeSeriesInputHandler.UserDataset(data, targets)


def _get_schema_floor(field_schema: dict[str, Any], default_value: Any) -> Any:
    minimum = field_schema.get("minimum")
    exclusive_minimum = field_schema.get("exclusiveMinimum")

    if minimum is None and exclusive_minimum is None:
        if isinstance(default_value, int):
            return 1
        if isinstance(default_value, float):
            return 0.0
        return default_value

    if isinstance(default_value, int):
        min_value = minimum if minimum is not None else exclusive_minimum + 1
        return int(min_value)

    if isinstance(default_value, float):
        min_value = minimum if minimum is not None else exclusive_minimum + 1e-6
        return float(min_value)

    return default_value


def _minimal_field_value(field_name: str, default_value: Any, field_schema: dict[str, Any]) -> Any:
    enum_values = field_schema.get("enum")
    if enum_values:
        if isinstance(default_value, str):
            if "none" in enum_values:
                return "none"
            return enum_values[0]
        return enum_values[0]

    if isinstance(default_value, bool):
        if field_name in {"online", "verbose"}:
            return False
        return default_value

    if default_value is None:
        if field_schema.get("type") == "object":
            return {}
        for schema_option in field_schema.get("anyOf", []):
            if schema_option.get("type") == "object":
                return {}
        if field_schema.get("type") == "array":
            return []
        return default_value

    if isinstance(default_value, int):
        floor = _get_schema_floor(field_schema, default_value)
        if field_name == "num_shadow_models":
            return max(2, floor)
        if "epoch" in field_name:
            return max(1, floor)
        if "batch_size" in field_name:
            return max(1, floor)
        if field_name == "max_num_evals":
            return max(2, floor)
        if field_name in {"initial_num_evals", "num_iterations"}:
            return max(1, floor)
        if field_name in {"num_transforms", "n_ops"}:
            return max(0, floor)
        return floor

    if isinstance(default_value, float):
        floor = _get_schema_floor(field_schema, default_value)
        if "fraction" in field_name:
            return max(floor, 0.5)
        if field_name == "gamma":
            return max(floor, 1.0)
        return floor

    if isinstance(default_value, list):
        if field_name == "quantiles":
            return [0.5]
        if len(default_value) > 1:
            return default_value[:1]
        return default_value

    return default_value


def _build_attack_config(attack_name: str) -> dict:
    attack_class = AttackFactoryMIA.attack_classes[attack_name]
    config_cls = attack_class.AttackConfig

    default_config = config_cls().model_dump()
    try:
        schema = config_cls.model_json_schema().get("properties", {})
    except Exception:
        schema = {}

    candidate = {}
    for field_name, default_value in default_config.items():
        field_schema = schema.get(field_name, {})
        candidate[field_name] = _minimal_field_value(field_name, default_value, field_schema)

    if "online" in candidate:
        candidate["online"] = False

    if "attack_data_fraction" in candidate:
        candidate["attack_data_fraction"] = max(candidate["attack_data_fraction"], 0.2)
    if "batch_size" in candidate:
        candidate["batch_size"] = max(candidate["batch_size"], 1)
    if "temperature" in candidate:
        candidate["temperature"] = max(candidate["temperature"], 1e-3)
    if "max_num_evals" in candidate and "initial_num_evals" in candidate:
        if candidate["max_num_evals"] <= candidate["initial_num_evals"]:
            candidate["max_num_evals"] = candidate["initial_num_evals"] + 1
    if "num_audit" in candidate:
        candidate["num_audit"] = max(candidate["num_audit"], 10)

    if attack_name == "HSJ":
        candidate["attack_data_fraction"] = max(candidate.get("attack_data_fraction", 0.5), 0.5)
        candidate["batch_size"] = max(candidate.get("batch_size", 2), 2)
        candidate["initial_num_evals"] = max(candidate.get("initial_num_evals", 1), 1)
        candidate["max_num_evals"] = max(candidate.get("max_num_evals", 2), candidate["initial_num_evals"] + 1)
        candidate["num_iterations"] = max(candidate.get("num_iterations", 1), 1)
        candidate["constraint"] = 2
        candidate["verbose"] = False

    if attack_name == "loss_traj":
        candidate["train_mia_batch_size"] = max(candidate.get("train_mia_batch_size", 2), 2)
        candidate["mia_classifier_epochs"] = max(candidate.get("mia_classifier_epochs", 1), 1)

    if attack_name == "seqmia":
        # SeqMIA builds 5 metrics per trajectory step; keep feature width aligned with model input.
        candidate["input_size"] = max(candidate.get("input_size", 5), 5)
        candidate["train_mia_batch_size"] = max(candidate.get("train_mia_batch_size", 2), 2)

    if attack_name == "oslo":
        # Keep a valid positive threshold range and an even post-audit pool size
        # to satisfy current balanced shadow assignment assumptions.
        candidate["min_threshold"] = max(candidate.get("min_threshold", 1e-4), 1e-4)
        candidate["max_threshold"] = max(candidate.get("max_threshold", 1.0), 1.0)
        candidate["n_audits"] = max(candidate.get("n_audits", 2), 2)
        if candidate["n_audits"] % 2 != 0:
            candidate["n_audits"] += 1

    validated_config = config_cls(**candidate)
    return validated_config.model_dump()


def _create_target_artifacts(
    run_dir: Path,
    population: AbstractInputHandler.UserDataset,
    target_model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    dataset_name: str,
    train_indices: list[int] | None = None,
    test_indices: list[int] | None = None,
) -> tuple[Path, Path]:
    target_dir = run_dir / "target"
    target_dir.mkdir(parents=True, exist_ok=True)

    data_path = run_dir / f"{dataset_name}.pkl"
    joblib.dump(population, data_path)

    train_result = TrainingOutput(model=target_model, metrics=EvalOutput(accuracy=0.5, loss=1.0))
    test_result = EvalOutput(accuracy=0.5, loss=1.0)
    train_loader = DataLoader(population, batch_size=4, shuffle=False)
    train_indices = TRAIN_INDICES if train_indices is None else train_indices
    test_indices = TEST_INDICES if test_indices is None else test_indices
    metadata = LeakPro.make_mia_metadata(
        train_result=train_result,
        optimizer=optimizer,
        loss_fn=criterion,
        dataloader=train_loader,
        test_result=test_result,
        epochs=1,
        train_indices=train_indices,
        test_indices=test_indices,
        dataset_name=dataset_name,
    )

    with open(target_dir / "target_model.pkl", "wb") as model_file:
        torch.save(target_model.state_dict(), model_file)
    with open(target_dir / "model_metadata.pkl", "wb") as metadata_file:
        pickle.dump(metadata, metadata_file)

    return data_path, target_dir


def _create_audit_yaml(
    run_dir: Path,
    attack_name: str,
    data_modality: str,
    model_class: str,
    data_path: Path,
    target_dir: Path,
) -> Path:
    config = {
        "audit": {
            "random_seed": 1234,
            "attack_type": "mia",
            "attack_list": [{"attack": attack_name, **_build_attack_config(attack_name)}],
            "data_modality": data_modality,
            "output_dir": str(run_dir / "leakpro_output"),
        },
        "target": {
            "module_path": __file__,
            "model_class": model_class,
            "target_folder": str(target_dir),
            "data_path": str(data_path),
        },
        "shadow_model": None,
        "distillation_model": None,
    }

    config_path = run_dir / "audit.yaml"
    with open(config_path, "w") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    return config_path


def _create_image_e2e_config(run_dir: Path, attack_name: str) -> Path:
    population = _build_tiny_image_population()
    target_model = TinyImageTargetModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(target_model.parameters(), lr=0.01)
    if attack_name == "rmia":
        train_indices = RMIA_TRAIN_INDICES
        test_indices = RMIA_TEST_INDICES
    else:
        train_indices = TRAIN_INDICES
        test_indices = TEST_INDICES
    data_path, target_dir = _create_target_artifacts(
        run_dir=run_dir,
        population=population,
        target_model=target_model,
        criterion=criterion,
        optimizer=optimizer,
        dataset_name="tiny_cifar",
        train_indices=train_indices,
        test_indices=test_indices,
    )
    return _create_audit_yaml(run_dir, attack_name, "image", "TinyImageTargetModel", data_path, target_dir)


def _create_timeseries_e2e_config(run_dir: Path, attack_name: str) -> Path:
    population = _build_tiny_timeseries_population()
    target_model = TinyTimeSeriesTargetModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(target_model.parameters(), lr=0.01)
    data_path, target_dir = _create_target_artifacts(
        run_dir=run_dir,
        population=population,
        target_model=target_model,
        criterion=criterion,
        optimizer=optimizer,
        dataset_name="tiny_timeseries",
    )
    return _create_audit_yaml(run_dir, attack_name, "timeseries", "TinyTimeSeriesTargetModel", data_path, target_dir)


ALL_MIA_ATTACKS = list(AttackFactoryMIA.attack_classes.keys())
DISABLED_ATTACKS: set[str] = set()
# Only add entries here for attacks that cannot run a real E2E path yet.
# Keys are attack names and values are non-empty blocker reasons, e.g.
# {"seqmia": "requires a patch because ..."}.
PATCHED_ATTACK_BLOCKERS: dict[str, str] = {}
PATCHED_ATTACKS: set[str] = set(PATCHED_ATTACK_BLOCKERS.keys())

ENABLED_MIA_ATTACKS = [attack_name for attack_name in ALL_MIA_ATTACKS if attack_name not in DISABLED_ATTACKS]
E2E_MODE_BY_ATTACK = {
    attack_name: ("patched" if attack_name in PATCHED_ATTACKS else "real")
    for attack_name in ENABLED_MIA_ATTACKS
}

ATTACK_PARAMETERS = [
    pytest.param(attack_name, id=f"{attack_name}-{E2E_MODE_BY_ATTACK[attack_name]}")
    for attack_name in ENABLED_MIA_ATTACKS
]


def test_attack_factory_attack_list_is_covered() -> None:
    """Ensure this test suite follows the current factory attack list."""
    assert ALL_MIA_ATTACKS == list(AttackFactoryMIA.attack_classes.keys())
    assert set(ENABLED_MIA_ATTACKS).union(DISABLED_ATTACKS) == set(ALL_MIA_ATTACKS)
    assert PATCHED_ATTACKS.issubset(set(ENABLED_MIA_ATTACKS))


def test_target_split_keeps_aux_population() -> None:
    """Ensure metadata split leaves non-audit points for auxiliary attack data."""
    assert set(TRAIN_INDICES).isdisjoint(TEST_INDICES)
    assert len(TRAIN_INDICES) > 0
    assert len(TEST_INDICES) > 0
    assert len(UNUSED_INDICES) > 0
    assert len(TRAIN_INDICES) + len(TEST_INDICES) + len(UNUSED_INDICES) == N_SAMPLES
    assert len(RMIA_TRAIN_INDICES) + len(RMIA_TEST_INDICES) == N_SAMPLES


def test_attack_e2e_modes_are_explicit() -> None:
    """Ensure each enabled attack is labeled as real or patched E2E."""
    assert set(E2E_MODE_BY_ATTACK.keys()) == set(ENABLED_MIA_ATTACKS)
    assert set(E2E_MODE_BY_ATTACK.values()).issubset({"real", "patched"})
    assert "real" in set(E2E_MODE_BY_ATTACK.values())
    assert set(PATCHED_ATTACK_BLOCKERS.keys()) == PATCHED_ATTACKS
    assert all(len(reason) > 0 for reason in PATCHED_ATTACK_BLOCKERS.values())


@pytest.mark.parametrize("attack_name", ATTACK_PARAMETERS)
def test_all_attacks_end_to_end(attack_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run a minimal real E2E flow for each MIA attack in AttackFactoryMIA."""
    _set_seed()
    run_mode = E2E_MODE_BY_ATTACK[attack_name]
    optuna_metadata_snapshot = _snapshot_optuna_metadata()
    temp_root = None
    try:
        with tempfile.TemporaryDirectory(prefix=f"mia_e2e_{attack_name}_") as temp_dir:
            temp_root = Path(temp_dir)
            monkeypatch.chdir(temp_root)
            _clear_global_attack_cache()

            run_dir = temp_root / attack_name
            run_dir.mkdir(parents=True, exist_ok=True)

            if attack_name == "dts":
                config_path = _create_timeseries_e2e_config(run_dir, attack_name)
                user_handler = TinyTimeSeriesInputHandler
            else:
                config_path = _create_image_e2e_config(run_dir, attack_name)
                user_handler = TinyImageInputHandler

            leakpro = LeakPro(user_handler, str(config_path))
            results = leakpro.run_audit(create_pdf=False, use_optuna=False)

            assert len(results) == 1
            result = results[0]
            assert result is not None
            assert isinstance(result, MIAResult)
            assert hasattr(result, "id")
            assert isinstance(result.id, str)
            assert result.id

            output_dir = Path(leakpro.handler.configs.audit.output_dir)
            result_file = output_dir / "results" / result.id / "result.txt"
            mode_file = output_dir / "results" / result.id / "e2e_mode.txt"
            blocker_file = output_dir / "results" / result.id / "e2e_blocker.txt"
            with open(mode_file, "w") as file:
                file.write(run_mode)
            if run_mode == "patched":
                with open(blocker_file, "w") as file:
                    file.write(PATCHED_ATTACK_BLOCKERS[attack_name])
            assert result_file.exists()
            assert mode_file.exists()
            if run_mode == "patched":
                assert blocker_file.exists()
            assert (output_dir / "data_objects").exists()

        assert temp_root is not None
        assert not temp_root.exists()
    finally:
        _restore_optuna_metadata(optuna_metadata_snapshot)
