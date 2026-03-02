"""Step definitions for the LeakPro terminal UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from leakpro.terminal_ui.api import CifarMIAApi, ConfigPaths
from leakpro.utils.import_helper import Any, Self


@dataclass
class AppContext:
    """Shared state across terminal UI steps."""

    base_dir: Path
    paths: ConfigPaths
    train_config: dict | None = None
    audit_config: dict | None = None
    population_dataset: object | None = None
    data: object | None = None
    targets: object | None = None
    dataset_name: str | None = None
    dataset_path: Path | None = None
    train_loader: object | None = None
    test_loader: object | None = None
    train_indices: object | None = None
    test_indices: object | None = None
    train_result: object | None = None
    test_result: object | None = None
    optimizer: object | None = None
    criterion: object | None = None
    epochs: int | None = None
    target_path: Path | None = None
    metadata: object | None = None
    metadata_path: Path | None = None
    audit_results: list | None = None


@dataclass
class StepResult:
    """Outcome of a single step execution."""

    status: str
    message: str


class Step:
    """Abstract step interface."""

    id: str = ""
    title: str = ""
    description: str = ""

    def run(self: Self, context: AppContext, api: CifarMIAApi, io: Any) -> StepResult:  # pragma: no cover - interface only
        """Execute the step."""
        raise NotImplementedError


class ConfigurePathsStep(Step):
    """Configure paths, configs, and attack selections."""

    id = "configure_paths"
    title = "Configure Paths & Audit"
    description = "Select configs, review/edit values, and choose audit attacks."

    def _get_nested_value(self: Self, config: dict, key: str) -> Any:
        """Return a nested config value using dot notation."""
        parts = key.split(".")
        val = config
        for p in parts:
            val = val.get(p, {})
        return val if val != config else config.get(key)

    def _set_nested_value(self: Self, config: dict, key: str, value: Any) -> None:
        """Set a nested config value using dot notation."""
        parts = key.split(".")
        d = config
        for p in parts[:-1]:
            if p not in d:
                d[p] = {}
            d = d[p]
        try:
            if isinstance(d.get(parts[-1]), int):
                value = int(value)
            elif isinstance(d.get(parts[-1]), float):
                value = float(value)
            elif isinstance(d.get(parts[-1]), bool):
                value = value.lower() in ("true", "yes", "1")
        except (ValueError, AttributeError):
            pass
        d[parts[-1]] = value

    def _print_config_section(self: Self, io: Any, title: str, config: dict, keys: list[tuple[str, str]]) -> None:
        """Print a section of configuration values."""
        io.print(f"\n{title}:")
        io.print("-" * 40)
        for key, label in keys:
            val = self._get_nested_value(config, key)
            io.print(f"  {label}: {val}")

    def _resolve_config_path(self: Self, io: Any, current: Path, filename: str) -> Path:
        """Resolve a config path, prompting when missing."""
        while True:
            if current.exists() and current.is_file():
                return current
            if current.exists() and current.is_dir():
                candidate = current / filename
                if candidate.exists():
                    return candidate

            io.warning(f"File not found: {current}")
            io.print("Use the path selector to find the correct file.")
            selected = io.ask_path(f"Find {filename}:", default=str(current.parent), must_exist=True)
            current = Path(selected)

    def _is_path_key(self: Self, key: str) -> bool:
        """Return True for path-like config keys."""
        return (
            key.endswith("_path")
            or key.endswith("_dir")
            or key.endswith("_folder")
            or key
            in {
                "run.log_dir",
                "data.data_dir",
                "audit.output_dir",
                "target.module_path",
                "target.data_path",
                "target.target_folder",
            }
        )

    def _path_must_exist(self: Self, key: str) -> bool:
        """Return True when a path must exist."""
        return key not in {"audit.output_dir", "run.log_dir", "data.data_dir", "target.data_path", "target.target_folder"}

    def _edit_config_loop(self: Self, io: Any, config: dict, keys: list[tuple[str, str]], config_name: str) -> dict:
        """Edit config values interactively."""
        while True:
            io.print(f"\n{'=' * 60}")
            io.heading(f"EDIT {config_name} CONFIG")
            io.print(f"{'=' * 60}")
            self._print_config_section(io, config_name, config, keys)

            io.print("\nAvailable keys to edit:")
            for i, (key, label) in enumerate(keys, 1):
                io.print(f"  {i}. {label} ({key})")
            io.print(f"  0. Done editing {config_name}")

            choice = io.ask("Select option", default="0")
            if choice == "0":
                break

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(keys):
                    key, label = keys[idx]
                    current_val = self._get_nested_value(config, key)
                    if self._is_path_key(key):
                        new_val = io.ask_path(
                            f"Select path for {label}:",
                            default=str(current_val),
                            must_exist=self._path_must_exist(key),
                        )
                    else:
                        new_val = io.ask(f"Enter new value for {label}", default=str(current_val))
                    self._set_nested_value(config, key, new_val)
                    io.success(f"Updated {label} to: {new_val}")
                else:
                    io.warning("Invalid option")
            except ValueError:
                io.warning("Please enter a valid number")

        return config

    def _print_attack_summary(self: Self, io: Any, attack: dict, index: int) -> None:
        """Print a summary of an attack configuration."""
        name = attack.get("attack", "unknown")
        io.heading(f"Attack {index}: {name}")
        has_fields = False
        for key, value in attack.items():
            if key == "attack":
                continue
            io.print(f"  {key}: {value}")
            has_fields = True
        if not has_fields:
            io.print("  <no default config>")

    def _get_attack_default_dict(self: Self, attack_cls: type) -> dict:
        """Return the default config for an attack class."""
        try:
            if hasattr(attack_cls, "AttackConfig"):
                return attack_cls.AttackConfig().model_dump()
            if hasattr(attack_cls, "Config"):
                return attack_cls.Config().model_dump()
            default_cfg = attack_cls.get_default_attack_config()
            return default_cfg.model_dump()
        except Exception:
            return {}

    def _parse_attack_selection(self: Self, selection: str, attack_names: list[str]) -> list[str]:
        """Parse attack selection input into attack names."""
        selection = selection.strip().lower()
        if selection in {"all", "*"}:
            return list(attack_names)

        tokens = [token.strip() for token in selection.split(",") if token.strip()]
        if not tokens:
            raise ValueError("No selections")

        resolved: list[str] = []
        for token in tokens:
            if token.isdigit():
                index = int(token)
                if index < 1 or index > len(attack_names):
                    raise ValueError("Selection out of range")
                resolved.append(attack_names[index - 1])
                continue
            if token not in attack_names:
                raise ValueError("Unknown attack")
            resolved.append(token)
        return resolved

    def _edit_attack_config(self: Self, io: Any, attack: dict) -> dict:  # noqa: C901
        keys = [(k, k) for k in attack if k != "attack"]
        if not keys:
            io.print("No editable fields for this attack.")
            return attack

        while True:
            io.print("\nAttack configuration:")
            for i, (key, _) in enumerate(keys, 1):
                io.print(f"  {i}. {key}: {attack.get(key)}")
            io.print("  0. Done")
            choice = io.ask("Select option", default="0")
            if choice == "0":
                break
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(keys):
                    key = keys[idx][0]
                    current_val = attack.get(key)
                    new_val = io.ask(f"Enter new value for {key}", default=str(current_val))
                    try:
                        if isinstance(current_val, int):
                            new_val = int(new_val)
                        elif isinstance(current_val, float):
                            new_val = float(new_val)
                        elif isinstance(current_val, bool):
                            new_val = new_val.lower() in ("true", "yes", "1")
                    except (ValueError, AttributeError):
                        pass
                    attack[key] = new_val
                    io.success(f"Updated {key} to: {new_val}")
                else:
                    io.warning("Invalid option")
            except ValueError:
                io.warning("Please enter a valid number")
        return attack

    def _configure_attacks(self: Self, io: Any, audit_config: dict) -> dict:  # noqa: C901, PLR0912
        """Select and edit attacks for the audit config."""
        from leakpro.attacks.mia_attacks.attack_factory_mia import AttackFactoryMIA

        available = AttackFactoryMIA.attack_classes
        configured_by_name: dict[str, list[dict]] = {}
        for entry in audit_config.get("audit", {}).get("attack_list", []):
            if not isinstance(entry, dict):
                continue
            name = entry.get("attack")
            if not name:
                continue
            configured_by_name.setdefault(name, []).append(entry)

        io.print("\nAvailable attacks:")
        attack_names = sorted(available.keys())
        for idx, name in enumerate(attack_names, 1):
            default_dict = self._get_attack_default_dict(available[name])
            suffix = "defaults" if default_dict else "no default config"
            io.print(f"  {idx}. {name} ({suffix})")

        io.print("\n" + "=" * 60)
        io.heading("ATTACK SELECTION")
        io.print("=" * 60)
        while True:
            selection = io.ask(
                "Select attacks by number or name (comma-separated, repeats allowed) or 'all'",
                default="all",
            )
            try:
                selected_names = self._parse_attack_selection(selection, attack_names)
                break
            except ValueError:
                io.warning("Invalid selection. Use numbers like 1,3,5 or names like lira,qmia.")

        updated_attacks = []
        for name in selected_names:
            default_dict = self._get_attack_default_dict(available[name])
            attack = {"attack": name, **default_dict}
            if configured_by_name.get(name):
                prior = configured_by_name[name].pop(0)
                attack.update({k: v for k, v in prior.items() if k != "attack"})
            self._print_attack_summary(io, attack, attack_names.index(name) + 1)
            if name == "lira":
                num_shadows = attack.get("num_shadow_models", 1)
                if isinstance(num_shadows, int) and num_shadows < 2:
                    io.warning("LiRA with <2 shadow models can produce NaN scores.")
                    if io.ask_yes_no("Edit LiRA configuration now?", default=True):
                        attack = self._edit_attack_config(io, dict(attack))
                        num_shadows = attack.get("num_shadow_models", num_shadows)
                        if isinstance(num_shadows, int) and num_shadows < 2:
                            io.warning("LiRA may still fail; consider num_shadow_models >= 2.")
            if io.ask_yes_no("Edit this attack's configuration?", default=False):
                attack = self._edit_attack_config(io, dict(attack))
            updated_attacks.append(attack)

        if not updated_attacks:
            io.warning("No attacks selected; keeping original attack list.")
            return audit_config

        audit_config["audit"]["attack_list"] = updated_attacks
        return audit_config

    def run(self: Self, context: AppContext, api: CifarMIAApi, io: Any) -> StepResult:  # noqa: C901
        """Run the configuration step."""
        io.print(f"Current training config: {context.paths.train_config}")
        io.print(f"Current audit config:    {context.paths.audit_config}")
        paths_changed = False

        if io.ask_yes_no("Edit config paths?", default=False):
            io.print("\n--- Configure Training Config ---")
            train_path_str = io.ask_path(
                "Select train_config.yaml location:", default=str(context.paths.train_config), must_exist=True
            )
            context.paths.train_config = Path(train_path_str)

            io.print("\n--- Configure Audit Config ---")
            audit_path_str = io.ask_path("Select audit.yaml location:", default=str(context.paths.audit_config), must_exist=True)
            context.paths.audit_config = Path(audit_path_str)
            paths_changed = True

        context.paths = ConfigPaths(
            self._resolve_config_path(io, context.paths.train_config, "train_config.yaml"),
            self._resolve_config_path(io, context.paths.audit_config, "audit.yaml"),
        )
        context.base_dir = context.paths.train_config.parent.resolve()
        io.print(f"\nWorking directory set to: {context.base_dir}")

        if paths_changed:
            context.train_config = None
            context.audit_config = None

        while True:
            if context.train_config is None:
                context.train_config = api.load_yaml(context.paths.train_config)
            if context.audit_config is None:
                context.audit_config = api.load_yaml(context.paths.audit_config)

            io.print("\n" + "=" * 60)
            io.heading("CONFIGURATION REVIEW")
            io.print("=" * 60)

            train_keys = [
                ("run.random_seed", "Random Seed"),
                ("run.log_dir", "Log Directory"),
                ("data.dataset", "Dataset"),
                ("data.data_dir", "Data Directory"),
                ("data.f_train", "Train Fraction"),
                ("data.f_test", "Test Fraction"),
                ("train.epochs", "Epochs"),
                ("train.batch_size", "Batch Size"),
                ("train.optimizer", "Optimizer"),
                ("train.learning_rate", "Learning Rate"),
            ]
            self._print_config_section(io, "TRAIN CONFIG", context.train_config, train_keys)

            audit_keys = [
                ("audit.random_seed", "Audit Random Seed"),
                ("audit.output_dir", "Output Directory"),
                ("target.model_class", "Model Class"),
                ("target.module_path", "Model Module"),
                ("target.data_path", "Data Path"),
                ("target.target_folder", "Target Folder"),
            ]
            self._print_config_section(io, "AUDIT CONFIG", context.audit_config, audit_keys)

            if io.ask_yes_no("Edit configuration values?", default=False):
                if io.ask_yes_no(f"Edit TRAIN CONFIG ({context.paths.train_config.name})?", default=True):
                    context.train_config = self._edit_config_loop(io, context.train_config, train_keys, "TRAIN")
                if io.ask_yes_no(f"Edit AUDIT CONFIG ({context.paths.audit_config.name})?", default=True):
                    context.audit_config = self._edit_config_loop(io, context.audit_config, audit_keys, "AUDIT")

            if io.ask_yes_no("Review and choose audit attacks?", default=True):
                context.audit_config = self._configure_attacks(io, context.audit_config)

            with context.paths.train_config.open("w") as f:
                import yaml

                yaml.dump(context.train_config, f)
            with context.paths.audit_config.open("w") as f:
                import yaml

                yaml.dump(context.audit_config, f)

            io.print("\n" + "=" * 60)
            if io.ask_yes_no("Is this configuration correct?", default=True):
                break
            io.warning("Let's review the configuration again.")

        train_seed = context.train_config["run"].get("random_seed")
        if train_seed is not None:
            api.set_random_seed(train_seed)

        return StepResult("done", "Configs loaded, reviewed, and updated")


class PreparePopulationStep(Step):
    """Prepare dataset and train/test loaders."""

    id = "prepare_population"
    title = "Prepare Dataset"
    description = "Select a dataset or provide a dataset file to use for the audit."

    def run(self: Self, context: AppContext, api: CifarMIAApi, io: Any) -> StepResult:  # noqa: ARG002
        """Prepare dataset based on config or user-provided file."""
        if context.train_config is None:
            raise RuntimeError("Train config not loaded")
        io.print("\nDataset options:")
        io.print("  1. Use predefined dataset from config (CIFAR10/100)")
        io.print("  2. Provide a dataset file (.pkl) path")
        choice = io.ask("Select option", default="1")
        if choice == "2":
            dataset_path = io.ask_path(
                "Select dataset file (.pkl):",
                default=str(context.base_dir / "data"),
                must_exist=True,
            )
            context.dataset_path = Path(dataset_path)
            dataset = api.load_population_dataset(context.dataset_path)
            if not hasattr(dataset, "data") or not hasattr(dataset, "targets"):
                raise RuntimeError("Dataset file must contain data and targets")
            context.population_dataset = dataset
            context.data = dataset.data
            context.targets = dataset.targets
            context.dataset_name = context.train_config["data"].get("dataset")
            io.success(f"Using dataset file: {context.dataset_path}")
        else:
            save_if_missing = io.ask_yes_no("Save population dataset if missing?", default=True)
            result = api.prepare_population_dataset(context.train_config, save_if_missing=save_if_missing)
            context.population_dataset = result["population_dataset"]
            context.data = result["data"]
            context.targets = result["targets"]
            context.dataset_name = result["dataset_name"]
            context.dataset_path = result["dataset_path"]

        if context.data is None or context.targets is None:
            raise RuntimeError("Population dataset not prepared")

        split = api.split_train_test(context.train_config, context.data, context.targets)
        context.train_loader = split["train_loader"]
        context.test_loader = split["test_loader"]
        context.train_indices = split["train_indices"]
        context.test_indices = split["test_indices"]
        return StepResult("done", "Dataset ready and train/test loaders created")


class SplitTrainTestStep(Step):
    """Validate pre-created train/test loaders."""

    id = "split_train_test"
    title = "Split Train/Test"
    description = "Create train/test loaders for the target model."

    def run(self: Self, context: AppContext, api: CifarMIAApi, io: Any) -> StepResult:  # noqa: ARG002
        """Confirm train/test loaders are ready."""
        if context.train_loader is None or context.test_loader is None:
            raise RuntimeError("Train/test loaders not prepared")
        return StepResult("done", "Train/test loaders already prepared")


class TrainTargetModelStep(Step):
    """Train the target model."""

    id = "train_target"
    title = "Train Target Model"
    description = "Train the target model and store weights to disk."

    def run(self, context: AppContext, api: CifarMIAApi, io: Any) -> StepResult:  # noqa: ARG002
        """Train the target model and record artifacts."""
        if context.train_config is None or context.audit_config is None:
            raise RuntimeError("Configs not loaded")
        if context.train_loader is None or context.test_loader is None:
            raise RuntimeError("Train/test loaders not prepared")
        if context.dataset_name is None:
            raise RuntimeError("Dataset name not set")
        result = api.train_target_model(
            context.train_config,
            context.audit_config,
            context.train_loader,
            context.test_loader,
            context.dataset_name,
        )
        context.train_result = result["train_result"]
        context.test_result = result["test_result"]
        context.optimizer = result["optimizer"]
        context.criterion = result["criterion"]
        context.epochs = result["epochs"]
        context.target_path = result["target_path"]
        return StepResult("done", f"Target model saved to {context.target_path}")


class CreateMetadataStep(Step):
    """Create MIA metadata."""

    id = "create_metadata"
    title = "Create Metadata"
    description = "Generate MIA metadata for LeakPro."

    def run(self: Self, context: AppContext, api: CifarMIAApi, io: Any) -> StepResult:  # noqa: ARG002
        """Generate and persist metadata for auditing."""
        if context.train_result is None or context.test_result is None:
            raise RuntimeError("Train/test results not available")
        if context.optimizer is None or context.criterion is None:
            raise RuntimeError("Training artifacts missing")
        if context.train_loader is None:
            raise RuntimeError("Train loader missing")
        if context.train_indices is None or context.test_indices is None:
            raise RuntimeError("Train/test indices missing")
        output_dir = context.paths.train_config.parent / "target"
        result = api.create_metadata(
            train_result=context.train_result,
            optimizer=context.optimizer,
            criterion=context.criterion,
            train_loader=context.train_loader,
            test_result=context.test_result,
            epochs=context.epochs or 0,
            train_indices=context.train_indices,
            test_indices=context.test_indices,
            dataset_name=context.dataset_name or "cifar",
            output_dir=output_dir,
        )
        context.metadata = result["metadata"]
        context.metadata_path = result["metadata_path"]
        return StepResult("done", f"Metadata saved to {context.metadata_path}")


class RunAuditStep(Step):
    """Run the LeakPro audit and report results."""

    id = "run_audit"
    title = "Run LeakPro Audit"
    description = "Execute the audit and generate the report."

    def run(self: Self, context: AppContext, api: CifarMIAApi, io: Any) -> StepResult:
        """Execute the audit and display result summary."""
        if context.dataset_path is not None:
            if context.audit_config is None:
                raise RuntimeError("Audit config not loaded")
            io.print(f"Using dataset: {context.dataset_path}")
            context.audit_config.setdefault("target", {})["data_path"] = str(context.dataset_path)
            with context.paths.audit_config.open("w") as f:
                import yaml

                yaml.dump(context.audit_config, f)

        create_pdf = io.ask_yes_no("Create PDF report?", default=True)
        results, report_dir = api.run_audit(context.paths.audit_config, create_pdf=create_pdf)
        context.audit_results = results
        summary_lines = [f"Results saved to: {report_dir}"]
        for res in results:
            summary_lines.append("")
            summary_lines.append(f"Attack: {res.result_name}")
            if res.fixed_fpr_table:
                summary_lines.append("  TPR at fixed FPR:")
                for key in ("TPR@1%FPR", "TPR@0.1%FPR", "TPR@0.01%FPR"):
                    summary_lines.append(f"    {key}: {res.fixed_fpr_table.get(key, 0.0)}")
            else:
                summary_lines.append("  TPR at fixed FPR: unavailable")
            roc_path = report_dir / "results" / res.id / "ROC.png"
            if roc_path.exists():
                summary_lines.append(f"ROC curve: {roc_path}")
                try:
                    from PIL import Image

                    img = Image.open(roc_path)
                    img.show()
                    io.ask("Close the ROC window and press Enter to continue", default="")
                except Exception:
                    io.warning("Could not display ROC image; open the file manually.")

        summary_path = report_dir / "audit_summary.txt"
        summary_path.write_text("\n".join(summary_lines))
        for line in summary_lines:
            io.print(line)
        io.print(f"\nSummary saved to: {summary_path}")
        return StepResult("done", "Audit completed")


@dataclass
class Flow:
    """Container for a named step flow."""

    name: str
    steps: list[Step] = field(default_factory=list)


def cifar_mia_flow() -> Flow:
    """Return the default CIFAR MIA flow."""
    return Flow(
        name="CIFAR MIA Guided Audit",
        steps=[
            ConfigurePathsStep(),
            PreparePopulationStep(),
            TrainTargetModelStep(),
            CreateMetadataStep(),
            RunAuditStep(),
        ],
    )
