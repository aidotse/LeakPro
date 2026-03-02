"""Main entry point for the LeakPro terminal UI."""

from __future__ import annotations

import argparse
import sys
import traceback
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Union

from leakpro.terminal_ui.api import CifarMIAApi, ConfigPaths
from leakpro.terminal_ui.io import TerminalIO
from leakpro.terminal_ui.steps import (
    AppContext,
    ConfigurePathsStep,
    CreateMetadataStep,
    Flow,
    PreparePopulationStep,
    RunAuditStep,
    SplitTrainTestStep,
    Step,
    TrainTargetModelStep,
    cifar_mia_flow,
)


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent.resolve()


def load_ascii_art() -> str:
    ascii_path = get_project_root() / "resources" / "leakpro_ascii.txt"
    if ascii_path.exists():
        return ascii_path.read_text(encoding="utf-8")
    return r"""


   _          __          __        _           _ 
  | |        /\ \        / /       | |         | |
  | |       /  \ \      / /        | |         | |
  | |      / /\ \ \    / /         | |         | |
  | |     / /  \ \ \  / /          | |         | |
  | |____/ /____\ \_\/_/           | |____     |_|
  |______/______\___/              |______|   (_)
                                              
  LeakPro - Privacy Auditing Framework
"""


class AutoIO:
    """Non-interactive IO for automated runs."""

    def __init__(self, real_io: TerminalIO) -> None:
        self.real_io = real_io
        self.use_color = False

    def print(self, text: str = "") -> None:
        self.real_io.print(text)

    def heading(self, text: str) -> None:
        self.real_io.print(text)

    def subtle(self, text: str) -> None:
        self.real_io.print(text)

    def success(self, text: str) -> None:
        self.real_io.success(text)

    def warning(self, text: str) -> None:
        self.real_io.warning(text)

    def error(self, text: str) -> None:
        self.real_io.error(text)

    def banner(self, art: str) -> None:
        self.real_io.banner(art)

    def set_overlay_art(self, art: str) -> None:
        self.real_io.set_overlay_art(art)

    def enable_overlay(self) -> None:
        self.real_io.enable_overlay()

    def ask(self, prompt: str, default: str | None = None, required: bool = False) -> str:
        if default is not None:
            self.real_io.print(f"{prompt} [{default}]: {default}")
            return default
        if required:
            raise RuntimeError(f"Required input {prompt} not provided in auto mode")
        return ""

    def ask_yes_no(self, prompt: str, default: bool | None = None) -> bool:
        if default is not None:
            self.real_io.print(f"{prompt} {'[Y/n]' if default else '[y/N]'}: {'y' if default else 'n'}")
            return default
        return True

    def ask_choice(self, prompt: str, choices: Any) -> str:
        choices_list = list(choices)
        self.real_io.print(f"{prompt}: {choices_list[0].key}")
        return choices_list[0].key


class TerminalApp:
    def __init__(self, base_dir: Path | None = None, auto_mode: bool = False) -> None:
        self.real_io = TerminalIO()
        self.auto_mode = auto_mode
        self.io: Union[TerminalIO, AutoIO] = self.real_io
        self.project_root = get_project_root()

        if base_dir is None:
            base_dir = self.project_root / "examples" / "mia" / "cifar"

        base_dir = base_dir.resolve()

        self.api = CifarMIAApi(base_dir)
        self.context = AppContext(
            base_dir=base_dir,
            paths=ConfigPaths(
                train_config=base_dir / "train_config.yaml",
                audit_config=base_dir / "audit.yaml",
            ),
        )
        self.flow = cifar_mia_flow()
        self.session_dir = Path.home() / ".leakpro_sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.session_id: str | None = None

    def enable_auto_mode(self) -> None:
        """Enable non-interactive auto mode with default answers."""
        self.auto_mode = True
        self.io = AutoIO(self.real_io)

    def print_banner(self) -> None:
        art = load_ascii_art()
        self.io.set_overlay_art(art)
        self.io.enable_overlay()
        self.io.print()
        self.io.success("Welcome to LeakPro Terminal UI")
        self.io.print()

    def print_step_header(self, step: Step, step_num: int, total: int) -> None:
        self.io.print()
        self.io.print("=" * 60)
        self.io.heading(f"Step {step_num}/{total}: {step.title}")
        self.io.subtle(step.description)
        self.io.print("=" * 60)

    def run_step(self, step: Step, step_num: int, total: int) -> bool:
        self.print_step_header(step, step_num, total)

        try:
            result = step.run(self.context, self.api, self.io)
            self.io.success(f"✓ {result.message}")
            return True
        except Exception as e:
            self.io.error(f"✗ Error: {e}")
            self.io.warning(f"Step failed.")
            self.io.print()
            traceback.print_exc()
            if self.auto_mode:
                return False
            return self.io.ask_yes_no("Retry this step?", default=True)

    def _render_menu_boxes(self, completed: set[int]) -> None:
        labels = {
            1: "Configure Paths & Audit",
            2: "Prepare Dataset",
            3: "Train Target + Metadata",
            4: "Run Audit",
        }
        active_ids = [1, 2, 3, 4]
        locked = {
            3: not completed.issuperset({1, 2}),
            4: not completed.issuperset({1, 2, 3}),
        }

        use_color = isinstance(self.io, TerminalIO)

        def build_box(index: int, label: str, done: bool, is_locked: bool) -> list[str]:
            text = f"{index}. {label}"
            width = max(len(text) + 2, 26)
            top = "+" + "-" * (width - 2) + "+"
            mid = "|" + text.ljust(width - 2) + "|"
            bot = "+" + "-" * (width - 2) + "+"
            if use_color:
                if done:
                    top = self.real_io._style(top, "32")
                    mid = self.real_io._style(mid, "32")
                    bot = self.real_io._style(bot, "32")
                elif is_locked:
                    top = self.real_io._style(top, "31")
                    mid = self.real_io._style(mid, "31")
                    bot = self.real_io._style(bot, "31")
            return [top, mid, bot]

        boxes = [build_box(idx, labels[idx], idx in completed, locked.get(idx, False)) for idx in active_ids]
        for row in range(3):
            line = "  ".join(box[row] for box in boxes)
            self.io.print(line)

    def _run_box_steps(self, steps: list[Step]) -> bool:
        total = len(steps)
        for i, step in enumerate(steps, 1):
            while True:
                ok = self.run_step(step, i, total)
                if ok:
                    break
                return False
        return True

    def run(self) -> None:
        self.print_banner()
        completed: set[int] = set()
        state = self._load_or_create_session()
        if state:
            completed = set(state.get("completed", []))
            self._apply_state(state)
        tasks: dict[int, list[Step]] = {
            1: [ConfigurePathsStep()],
            2: [PreparePopulationStep()],
            3: [TrainTargetModelStep(), CreateMetadataStep()],
        }
        audit_step = RunAuditStep()

        while True:
            self.io.print()
            self._render_menu_boxes(completed)
            self.io.print()
            if completed.issuperset({1, 2, 3}):
                prompt = "Select 1, 2, 3, or 4 (audit), or q to quit"
            else:
                prompt = "Select 1, 2, 3 to complete tasks, or q to quit"
            choice = self.io.ask(prompt, default="1").strip().lower()

            if choice in {"q", "quit", "exit"}:
                self.io.print("Aborted. Goodbye!")
                return

            if choice == "4":
                if not completed.issuperset({1, 2, 3}):
                    self.io.warning("Complete tasks 1-3 before running the audit.")
                    continue
                if self._run_box_steps([audit_step]):
                    completed.add(4)
                    self._save_state(completed)
                    self.io.print()
                    self._render_menu_boxes(completed)
                    break
                continue

            if choice.isdigit() and int(choice) in tasks:
                task_id = int(choice)
                if self._run_box_steps(tasks[task_id]):
                    completed.add(task_id)
                    self._save_state(completed)
                continue

            self.io.warning("Invalid selection.")

        self.io.print()
        self.io.success("=" * 60)
        self.io.heading("Audit Complete!")
        self.io.success("=" * 60)

        self.io.print("\nThank you for using LeakPro!")

    def _session_summary(self, state: dict) -> str:
        base_dir = state.get("base_dir", "")
        completed = state.get("completed", [])
        return f"{state.get('session_id', '')} | {base_dir} | completed: {completed}"

    def _load_or_create_session(self) -> dict | None:
        sessions = sorted(self.session_dir.glob("session_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if sessions:
            self.io.print("\nAvailable sessions (most recent first):")
            for idx, path in enumerate(sessions, 1):
                try:
                    data = json.loads(path.read_text())
                    self.io.print(f"  {idx}. {self._session_summary(data)}")
                except Exception:
                    self.io.print(f"  {idx}. {path.name} (corrupt)")
            choice = self.io.ask("Select a session number or 'new'", default="new").strip().lower()
            if choice != "new" and choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(sessions):
                    try:
                        data = json.loads(sessions[index].read_text())
                        self.session_id = data.get("session_id")
                        return data
                    except Exception:
                        self.io.warning("Failed to load session; starting new.")
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._save_state(set())
        return None

    def _save_state(self, completed: set[int]) -> None:
        state = {
            "session_id": self.session_id,
            "base_dir": str(self.context.base_dir),
            "paths": {
                "train_config": str(self.context.paths.train_config),
                "audit_config": str(self.context.paths.audit_config),
            },
            "train_config_snapshot": self.context.train_config,
            "audit_config_snapshot": self.context.audit_config,
            "dataset_path": str(self.context.dataset_path) if self.context.dataset_path else None,
            "completed": sorted(list(completed)),
        }
        path = self.session_dir / f"session_{self.session_id}.json"
        path.write_text(json.dumps(state, indent=2))

    def _apply_state(self, state: dict) -> None:
        self.context.base_dir = Path(state.get("base_dir", self.context.base_dir))
        paths = state.get("paths", {})
        self.context.paths.train_config = Path(paths.get("train_config", self.context.paths.train_config))
        self.context.paths.audit_config = Path(paths.get("audit_config", self.context.paths.audit_config))
        self.context.train_config = state.get("train_config_snapshot")
        self.context.audit_config = state.get("audit_config_snapshot")
        dataset_path = state.get("dataset_path")
        self.context.dataset_path = Path(dataset_path) if dataset_path else None

    def run_auto(self) -> bool:
        """Run the entire workflow non-interactively."""
        self.enable_auto_mode()
        self.io.set_overlay_art(load_ascii_art())
        self.io.enable_overlay()
        self.io.print("[AUTO MODE] Running leakpro workflow non-interactively")
        self.io.print()

        total = len(self.flow.steps)
        for i, step in enumerate(self.flow.steps, 1):
            self.io.print(f"[AUTO MODE] Running step {i}/{total}: {step.title}")
            while True:
                if self.run_step(step, i, total):
                    break

        self.io.print()
        self.io.success("=" * 60)
        self.io.heading("Audit Complete!")
        self.io.success("=" * 60)

        if self.context.audit_results:
            self.io.print("\nAttack Results Summary:")
            for result in self.context.audit_results:
                self.io.print(f"  - {result}")

        self.io.print("\nThank you for using LeakPro!")
        return True


def create_parser() -> argparse.ArgumentParser:
    project_root = get_project_root()
    default_cifar = project_root / "examples" / "mia" / "cifar"

    parser = argparse.ArgumentParser(
        prog="leakpro.terminal_ui",
        description="LeakPro Terminal UI - Interactive privacy auditing workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m leakpro.terminal_ui
  python -m leakpro.terminal_ui --base-dir ./examples/mia/cifar
  python -m leakpro.terminal_ui -b /path/to/configs --no-color
  python -m leakpro.terminal_ui --auto --base-dir ./examples/mia/cifar
        """,
    )
    parser.add_argument(
        "-b",
        "--base-dir",
        type=Path,
        default=default_cifar,
        help=f"Base directory containing train_config.yaml and audit.yaml (default: %(default)s)",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("-q", "--quiet", action="store_true", help="Skip welcome message")
    parser.add_argument(
        "-a",
        "--auto",
        action="store_true",
        help="Run in non-interactive mode with default answers (useful for testing/automation)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    base_dir: Path = args.base_dir
    if not base_dir.is_dir():
        print(f"Error: Base directory does not exist: {base_dir}", file=sys.stderr)
        return 1

    if not (base_dir / "train_config.yaml").exists():
        print(f"Error: train_config.yaml not found in {base_dir}", file=sys.stderr)
        return 1

    if not (base_dir / "audit.yaml").exists():
        print(f"Error: audit.yaml not found in {base_dir}", file=sys.stderr)
        return 1

    use_color = not args.no_color
    app = TerminalApp(base_dir=base_dir, auto_mode=args.auto)
    app.io.use_color = use_color and sys.stdout.isatty()

    if args.auto:
        try:
            app.run_auto()
        except Exception as e:
            print(f"Error during auto run: {e}", file=sys.stderr)
            traceback.print_exc()
            return 1
    elif not args.quiet:
        app.run()
    else:
        total = len(app.flow.steps)
        for i, step in enumerate(app.flow.steps, 1):
            while True:
                if app.run_step(step, i, total):
                    break

    return 0


if __name__ == "__main__":
    sys.exit(main())
