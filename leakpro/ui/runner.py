"""Backend runner: wraps the LeakPro pipeline for the Streamlit UI.

Handles data preparation, model training (standard + DP-SGD), and audit execution.
Log messages are streamed to the UI via a custom logging handler.
"""

from __future__ import annotations

import contextlib
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch import cat, nn, optim, tensor
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100

# Ensure project root is on sys.path when the UI is launched directly
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_CIFAR_DIR = _PROJECT_ROOT / "examples" / "mia" / "cifar"

# Also put the CIFAR dir on sys.path so that LeakPro's internal re-imports
# of the handler (e.g. during shadow model training) can find 'cifar_handler'
# by its simple module name — matching how the notebooks import it.
if str(_CIFAR_DIR) not in sys.path:
    sys.path.insert(0, str(_CIFAR_DIR))

# Lazy imports from leakpro (available once project root is on path)
from leakpro import LeakPro  # noqa: E402
from leakpro.utils.logger import logger as leakpro_logger  # noqa: E402


# ---------------------------------------------------------------------------
# Streamlit log handler
# ---------------------------------------------------------------------------

class StreamlitLogHandler(logging.Handler):
    """Captures leakpro log messages and writes them to a Streamlit container.

    Attach before a long-running call; detach afterwards.
    """

    def __init__(self, container) -> None:
        super().__init__()
        self.container = container
        self._lines: list[str] = []
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self._lines.append(msg)
        # Keep only the most recent 30 lines to avoid giant text areas
        self.container.code("\n".join(self._lines[-30:]), language=None)


@contextlib.contextmanager
def _stream_logs(container):
    """Context manager that attaches/detaches a StreamlitLogHandler."""
    handler = StreamlitLogHandler(container)
    leakpro_logger.addHandler(handler)
    try:
        yield handler
    finally:
        leakpro_logger.removeHandler(handler)


@contextlib.contextmanager
def _in_cifar_dir():
    """Temporarily change the working directory to the CIFAR example folder.

    This is required because audit.yaml uses relative paths (./target, ./data/…).
    """
    original = os.getcwd()
    os.chdir(_CIFAR_DIR)
    try:
        yield
    finally:
        os.chdir(original)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

class LeakProRunner:
    """Orchestrates the full LeakPro auditing pipeline for the Streamlit UI."""

    # ------------------------------------------------------------------
    # Stage 2a – prepare dataset
    # ------------------------------------------------------------------

    def prepare_data(self, train_config: dict, log_container=None) -> dict:
        """Download CIFAR data, build population dataset, create train/test splits.

        Returns a dict with keys:
            data, targets, population_dataset, dataset_name,
            train_loader, test_loader, train_indices, test_indices
        """
        with _in_cifar_dir():
            dataset_name: str = train_config["data"]["dataset"]
            root: str = train_config["data"]["data_dir"]
            batch_size: int = train_config["train"]["batch_size"]
            f_train: float = train_config["data"]["f_train"]
            f_test: float = train_config["data"]["f_test"]
            random_seed: int = train_config["run"].get("random_seed", 42)

            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

            if log_container:
                log_container.info(f"Downloading / loading {dataset_name}…")

            # ---- load torchvision dataset --------------------------------
            cls = CIFAR10 if dataset_name == "cifar10" else CIFAR100
            trainset = cls(root=root, train=True, download=True)
            testset = cls(root=root, train=False, download=True)

            train_data = tensor(trainset.data).permute(0, 3, 1, 2).float() / 255
            test_data = tensor(testset.data).permute(0, 3, 1, 2).float() / 255
            data = cat([train_data, test_data], dim=0)
            targets = cat([tensor(trainset.targets), tensor(testset.targets)], dim=0)

            # ---- import UserDataset (defined inside the handler) ---------
            from cifar_handler import CifarInputHandler  # noqa: PLC0415

            population_dataset = CifarInputHandler.UserDataset(data, targets)

            # ---- save population pickle if missing -----------------------
            pkl_path = Path(f"data/{dataset_name}.pkl")
            pkl_path.parent.mkdir(exist_ok=True)
            if not pkl_path.exists():
                with open(pkl_path, "wb") as f:
                    pickle.dump(population_dataset, f)
                if log_container:
                    log_container.success(f"Population dataset saved to {pkl_path}")

            # ---- train / test split --------------------------------------
            dataset_size = len(population_dataset)
            train_size = int(f_train * dataset_size)
            test_size = int(f_test * dataset_size)
            selected = np.random.choice(np.arange(dataset_size), train_size + test_size, replace=False)
            train_indices, test_indices = train_test_split(selected, test_size=test_size)

            train_subset = CifarInputHandler.UserDataset(data[train_indices], targets[train_indices])
            test_subset = CifarInputHandler.UserDataset(
                data[test_indices], targets[test_indices], **train_subset.return_params()
            )

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

            if log_container:
                log_container.success(
                    f"Dataset ready — {train_size} train / {test_size} test samples."
                )

            return {
                "data": data,
                "targets": targets,
                "population_dataset": population_dataset,
                "dataset_name": dataset_name,
                "train_loader": train_loader,
                "test_loader": test_loader,
                "train_indices": train_indices,
                "test_indices": test_indices,
            }

    # ------------------------------------------------------------------
    # Stage 2b – standard training
    # ------------------------------------------------------------------

    def train_standard(self, train_config: dict, data_result: dict, log_container=None) -> dict:
        """Train target model without differential privacy.

        Returns a dict with keys:
            model, train_result, test_result, epochs, target_folder
        """
        with _in_cifar_dir():
            from cifar_handler import CifarInputHandler  # noqa: PLC0415
            from target_model_class import WideResNet  # noqa: PLC0415

            dataset_name: str = data_result["dataset_name"]
            num_classes = 10 if dataset_name == "cifar10" else 100
            epochs: int = train_config["train"]["epochs"]
            lr: float = train_config["train"]["learning_rate"]
            weight_decay: float = train_config["train"]["weight_decay"]
            target_folder = Path(train_config["run"]["log_dir"])
            target_folder.mkdir(parents=True, exist_ok=True)

            model = WideResNet(depth=28, num_classes=num_classes, widen_factor=2)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            if log_container:
                log_container.info("Starting standard training…")

            maybe_stream = _stream_logs(log_container) if log_container else contextlib.nullcontext()
            with maybe_stream:
                train_result = CifarInputHandler().train(
                    dataloader=data_result["train_loader"],
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    epochs=epochs,
                )

            test_result = CifarInputHandler().eval(
                data_result["test_loader"], train_result.model, criterion
            )

            # Save model
            model_path = target_folder / "target_model.pkl"
            torch.save(train_result.model.state_dict(), model_path)

            # Save metadata
            meta = LeakPro.make_mia_metadata(
                train_result=train_result,
                optimizer=optimizer,
                loss_fn=criterion,
                dataloader=data_result["train_loader"],
                test_result=test_result,
                epochs=epochs,
                train_indices=data_result["train_indices"],
                test_indices=data_result["test_indices"],
                dataset_name=data_result["dataset_name"],
            )
            with open(target_folder / "model_metadata.pkl", "wb") as f:
                pickle.dump(meta, f)

            if log_container:
                log_container.success(
                    f"Training complete — train acc: {train_result.metrics.accuracy:.3f}, "
                    f"test acc: {test_result.accuracy:.3f}"
                )

            return {
                "model": train_result.model,
                "train_result": train_result,
                "test_result": test_result,
                "epochs": epochs,
                "target_folder": str(target_folder),
                "optimizer": optimizer,
                "criterion": criterion,
            }

    # ------------------------------------------------------------------
    # Stage 2c – DP-SGD training
    # ------------------------------------------------------------------

    def train_dpsgd(
        self,
        train_config: dict,
        dpsgd_params: dict,
        data_result: dict,
        log_container=None,
    ) -> dict:
        """Train target model with differential privacy (DP-SGD via Opacus).

        dpsgd_params keys: target_epsilon, target_delta, max_grad_norm,
                           virtual_batch_size (optional, default 16)

        Returns same dict shape as train_standard, plus 'achieved_epsilon'.
        """
        with _in_cifar_dir():
            from cifar_handler_dpsgd import CifarInputHandlerDPsgd  # noqa: PLC0415
            from target_model_class import ResNet18_DPsgd  # noqa: PLC0415

            dataset_name: str = data_result["dataset_name"]
            num_classes = 10 if dataset_name == "cifar10" else 100
            epochs: int = train_config["train"]["epochs"]
            lr: float = train_config["train"]["learning_rate"]
            momentum: float = train_config["train"].get("momentum", 0.9)
            target_folder = Path("target_dpsgd")
            target_folder.mkdir(parents=True, exist_ok=True)

            virtual_batch_size: int = dpsgd_params.get("virtual_batch_size", 16)

            # Build and save dpsgd_dic.pkl that the handler reads
            sample_rate = 1.0 / len(data_result["train_loader"])
            privacy_dict = {
                "target_epsilon": dpsgd_params["target_epsilon"],
                "target_delta": dpsgd_params["target_delta"],
                "sample_rate": sample_rate,
                "epochs": epochs,
                "epsilon_tolerance": 0.01,
                "accountant": "prv",
                "eps_error": 0.01,
                "max_grad_norm": dpsgd_params["max_grad_norm"],
            }
            privacy_pkl_path = str(target_folder / "dpsgd_dic.pkl")
            with open(privacy_pkl_path, "wb") as f:
                pickle.dump(privacy_dict, f)

            model = ResNet18_DPsgd(num_classes=num_classes, dpsgd=True)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

            if log_container:
                log_container.info(
                    f"Starting DP-SGD training (ε={dpsgd_params['target_epsilon']}, "
                    f"δ={dpsgd_params['target_delta']})…"
                )

            maybe_stream = _stream_logs(log_container) if log_container else contextlib.nullcontext()
            with maybe_stream:
                train_result = CifarInputHandlerDPsgd().train(
                    dataloader=data_result["train_loader"],
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    epochs=epochs,
                    dpsgd_metadata_path=privacy_pkl_path,
                    virtual_batch_size=virtual_batch_size,
                )

            test_result = CifarInputHandlerDPsgd().eval(
                data_result["test_loader"], train_result.model, criterion
            )

            torch.save(train_result.model.state_dict(), target_folder / "target_model.pkl")

            meta = LeakPro.make_mia_metadata(
                train_result=train_result,
                optimizer=optimizer,
                loss_fn=criterion,
                dataloader=data_result["train_loader"],
                test_result=test_result,
                epochs=epochs,
                train_indices=data_result["train_indices"],
                test_indices=data_result["test_indices"],
                dataset_name=data_result["dataset_name"],
            )
            with open(target_folder / "model_metadata.pkl", "wb") as f:
                pickle.dump(meta, f)

            if log_container:
                log_container.success(
                    f"DP-SGD training complete — train acc: {train_result.metrics.accuracy:.3f}, "
                    f"test acc: {test_result.accuracy:.3f}"
                )

            return {
                "model": train_result.model,
                "train_result": train_result,
                "test_result": test_result,
                "epochs": epochs,
                "target_folder": str(target_folder),
                "optimizer": optimizer,
                "criterion": criterion,
                "dpsgd_params": privacy_dict,
            }

    # ------------------------------------------------------------------
    # Stage 3 – run audit
    # ------------------------------------------------------------------

    def run_audit(self, dpsgd: bool = False, log_container=None) -> list:
        """Run LeakPro MIA attacks and return a list of MIAResult objects."""
        with _in_cifar_dir():
            if dpsgd:
                from cifar_handler_dpsgd import CifarInputHandlerDPsgd  # noqa: PLC0415
                handler_cls = CifarInputHandlerDPsgd
                config_path = "audit_dpsgd.yaml"
            else:
                from cifar_handler import CifarInputHandler  # noqa: PLC0415
                handler_cls = CifarInputHandler
                config_path = "audit.yaml"

            if log_container:
                log_container.info(f"Initialising audit with config: {config_path}")

            maybe_stream = _stream_logs(log_container) if log_container else contextlib.nullcontext()
            with maybe_stream:
                leakpro = LeakPro(handler_cls, config_path)
                results = leakpro.run_audit(create_pdf=False)

            if log_container:
                log_container.success(f"Audit complete — {len(results)} attack(s) finished.")

            return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def load_audit_results_from_disk(output_dir: str) -> list:
        """Load previously saved MIAResult objects from a leakpro_output directory."""
        from leakpro.reporting.mia_result import MIAResult  # noqa: PLC0415

        data_objects_dir = Path(output_dir) / "data_objects"
        results = []
        if not data_objects_dir.exists():
            return results
        for json_file in sorted(data_objects_dir.glob("*.json")):
            try:
                results.append(MIAResult.load(str(json_file)))
            except Exception:  # noqa: BLE001
                pass
        return results

    @staticmethod
    def default_train_config() -> dict:
        config_path = _CIFAR_DIR / "train_config.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)

    @staticmethod
    def default_audit_config() -> dict:
        config_path = _CIFAR_DIR / "audit.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)
