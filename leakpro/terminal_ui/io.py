"""Terminal input/output helpers for LeakPro terminal UI."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts import clear

from leakpro.utils.import_helper import Iterable, Self


@dataclass
class Choice:
    """Selection option for interactive prompts."""

    key: str
    label: str
    description: str | None = None


class TerminalIO:
    """Minimal terminal IO wrapper with light styling and validation."""

    def __init__(self: Self, use_color: bool = True) -> None:
        """Initialize terminal IO with optional color output."""
        self.use_color = use_color and sys.stdout.isatty()
        self.session = PromptSession()
        self.overlay_art: list[str] = []
        self.overlay_active = False
        self._buffer: list[str] = []
        self._overlay_width = 0

    def _style(self: Self, text: str, code: str) -> str:
        """Apply ANSI color styling if enabled."""
        if not self.use_color:
            return text
        return f"\033[{code}m{text}\033[0m"

    def heading(self: Self, text: str) -> None:
        """Print a bold heading."""
        self.print(self._style(text, "1"))

    def subtle(self: Self, text: str) -> None:
        """Print subtle, dimmed text."""
        self.print(self._style(text, "2"))

    def success(self: Self, text: str) -> None:
        """Print a success message."""
        self.print(self._style(text, "32"))

    def warning(self: Self, text: str) -> None:
        """Print a warning message."""
        self.print(self._style(text, "33"))

    def error(self: Self, text: str) -> None:
        """Print an error message."""
        self.print(self._style(text, "31"))

    def print(self: Self, text: str = "") -> None:
        """Print text, respecting overlay mode."""
        if self.overlay_active:
            self._buffer.append(text)
            self._render()
        else:
            sys.stdout.write(text + "\n")
            sys.stdout.flush()

    def banner(self: Self, art: str) -> None:
        """Print or render a banner block."""
        if self.overlay_active:
            self._buffer.append(art.rstrip("\n"))
            self._render()
        else:
            self.print(art.rstrip("\n"))

    def ask(self: Self, prompt: str, default: str | None = None, required: bool = False) -> str:
        """Prompt for a string value."""
        if default is not None:
            prompt = f"{prompt} [{default}]"
        prompt = f"{prompt}: "
        while True:
            try:
                response = self._prompt(prompt).strip()
            except EOFError as err:
                raise RuntimeError("Input stream closed. Cannot continue.") from err
            except KeyboardInterrupt as err:
                raise KeyboardInterrupt("Operation cancelled by user.") from err

            if response:
                return response
            if default is not None:
                return default
            if required:
                self.warning("This field is required. Please enter a value.")
            else:
                return ""

    def ask_yes_no(self: Self, prompt: str, default: bool | None = None) -> bool:
        """Prompt for a yes/no confirmation."""
        suffix = ""
        if default is True:
            suffix = " [Y/n]"
        elif default is False:
            suffix = " [y/N]"
        else:
            suffix = " [y/n]"
        while True:
            try:
                response = self._prompt(f"{prompt}{suffix}: ").strip().lower()
            except EOFError as err:
                raise RuntimeError("Input stream closed. Cannot continue.") from err
            except KeyboardInterrupt as err:
                raise KeyboardInterrupt("Operation cancelled by user.") from err

            if not response and default is not None:
                return default
            if response in {"y", "yes"}:
                return True
            if response in {"n", "no"}:
                return False
            self.warning("Please enter y or n.")

    def ask_choice(self: Self, prompt: str, choices: Iterable[Choice]) -> str:
        """Prompt for a choice by index or key."""
        self.print(prompt)
        choices_list = list(choices)
        for idx, choice in enumerate(choices_list, start=1):
            line = f"{idx}. {choice.label}"
            if choice.description:
                line += f" — {choice.description}"
            self.print(line)
        options = [c.key for c in choices_list]
        while True:
            try:
                response = self._prompt("Select option: ").strip()
            except EOFError as err:
                raise RuntimeError("Input stream closed. Cannot continue.") from err
            except KeyboardInterrupt as err:
                raise KeyboardInterrupt("Operation cancelled by user.") from err

            if response.isdigit():
                index = int(response) - 1
                if 0 <= index < len(options):
                    return options[index]
            if response in options:
                return response
            self.warning("Invalid selection.")

    def ask_path(self: Self, prompt: str, default: str | None = None, must_exist: bool = True) -> str:  # noqa: C901, PLR0912, PLR0915
        """Prompt for a path using a shell-like navigator."""
        current_path = Path(default) if default else Path.cwd()

        self.print(f"{prompt}")
        self.print("Commands: ls, cd <dir>, Add, pwd, q (quit)")
        self.print("-" * 40)

        while True:
            try:
                response = self._prompt(f"{current_path}> ").strip()
            except EOFError as err:
                raise RuntimeError("Input stream closed. Cannot continue.") from err
            except KeyboardInterrupt as err:
                raise KeyboardInterrupt("Operation cancelled by user.") from err

            if not response:
                continue

            parts = response.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("q", "quit", "exit"):
                if default:
                    return default
                return str(current_path)

            if cmd in ("p", "pwd"):
                self.print(str(current_path))

            elif cmd in ("ls", "dir"):
                try:
                    entries = list(current_path.iterdir())
                    for e in sorted(entries):
                        suffix = "/" if e.is_dir() else ""
                        marker = "*" if e.is_file() and e.suffix in (".yaml", ".yml", ".json") else ""
                        self.print(f"  {e.name}{suffix}{marker}")
                except PermissionError:
                    self.warning("Permission denied")

            elif cmd in ("cd"):
                if not arg:
                    current_path = Path.home()
                else:
                    new_path = Path(arg)
                    if not new_path.is_absolute():
                        new_path = current_path / new_path
                    new_path = new_path.resolve()
                    if new_path.exists() and new_path.is_dir():
                        current_path = new_path
                    else:
                        self.warning(f"Directory not found: {new_path}")

            elif cmd in ("a", "add"):
                target = Path(arg) if arg else current_path
                if not target.is_absolute():
                    target = current_path / target
                target = target.resolve()

                if must_exist and not target.exists():
                    self.warning(f"Path does not exist: {target}")
                    continue

                return str(target)

            elif cmd == "h":
                self.print("Commands:")
                self.print("  ls/dir    - List directory contents")
                self.print("  cd <dir>  - Change directory")
                self.print("  Add       - Use current directory")
                self.print("  Add <path>- Use specified path")
                self.print("  pwd       - Print working directory")
                self.print("  h         - Show this help")
                self.print("  q         - Quit and use default")

            else:
                self.warning(f"Unknown command: {cmd}. Type 'h' for help.")

    def set_overlay_art(self: Self, art: str) -> None:
        """Set the overlay ASCII art content."""
        raw_lines = [line.rstrip("\n") for line in art.splitlines()]
        if not raw_lines:
            self.overlay_art = []
            self._overlay_width = 0
            return
        min_indent = min((len(line) - len(line.lstrip(" ")) for line in raw_lines if line.strip()), default=0)
        normalized = [line[min_indent:] for line in raw_lines if line.strip()]
        self.overlay_art = normalized
        self._overlay_width = max((len(line) for line in normalized), default=0)

    def enable_overlay(self: Self) -> None:
        """Enable overlay rendering."""
        self.overlay_active = True
        self._render()

    def disable_overlay(self: Self) -> None:
        """Disable overlay rendering and clear buffer."""
        self.overlay_active = False
        self._buffer.clear()
        clear()

    def _prompt(self: Self, text: str) -> str:
        """Internal prompt wrapper that supports overlay mode."""
        if not self.overlay_active:
            return self.session.prompt(text)
        with patch_stdout():
            return self.session.prompt(text)

    def _render(self: Self) -> None:
        """Render the overlay and buffered output to the terminal."""
        clear()
        rows, cols = self._get_terminal_size()
        overlay_lines = self.overlay_art
        overlay_height = len(overlay_lines)
        content_height = max(rows - 1 - overlay_height, 0)
        buffer_lines = self._buffer[-content_height:] if content_height > 0 else []

        for i in range(rows - 1):
            if i < overlay_height:
                line = overlay_lines[i][:cols]
            else:
                idx = i - overlay_height
                line = buffer_lines[idx] if idx < len(buffer_lines) else ""
            sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def _get_terminal_size(self: Self) -> tuple[int, int]:
        """Return terminal (rows, cols) with a fallback default."""
        try:
            size = os.get_terminal_size()
            return size.lines, size.columns
        except Exception:
            return 24, 80

    def _render_overlay(self: Self, _cols: int) -> dict[int, str]:
        """Return a rendered overlay mapping (unused placeholder)."""
        return {}

    def _merge_lines(self: Self, base: str, _overlay: str, cols: int) -> str:
        """Merge overlay and base lines (unused placeholder)."""
        return base[:cols]
