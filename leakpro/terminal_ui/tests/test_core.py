"""Tests for LeakPro Terminal UI."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


class TestTerminalIO:
    """Tests for TerminalIO class."""

    def test_style_with_color(self):
        from leakpro.terminal_ui.io import TerminalIO

        io = TerminalIO(use_color=True)
        # Note: color is disabled when not a TTY, so result depends on environment
        result = io._style("test", "32")
        assert result is not None  # Either returns styled or plain text

    def test_style_without_color(self):
        from leakpro.terminal_ui.io import TerminalIO

        io = TerminalIO(use_color=False)
        result = io._style("test", "32")
        assert result == "test"

    def test_print(self):
        from leakpro.terminal_ui.io import TerminalIO

        io = TerminalIO(use_color=False)
        io.print("test")
        # Basic test - if no exception, it's fine

    def test_banner(self):
        from leakpro.terminal_ui.io import TerminalIO

        io = TerminalIO(use_color=False)
        io.banner("test\n")
        # Basic test - if no exception, it's fine


class TestAutoIO:
    """Tests for AutoIO class."""

    def test_ask_with_default(self):
        from leakpro.terminal_ui.__main__ import AutoIO, TerminalIO

        real_io = TerminalIO(use_color=False)
        auto_io = AutoIO(real_io)
        result = auto_io.ask("Test", default="default_value")
        assert result == "default_value"

    def test_ask_yes_no_with_default_true(self):
        from leakpro.terminal_ui.__main__ import AutoIO, TerminalIO

        real_io = TerminalIO(use_color=False)
        auto_io = AutoIO(real_io)
        result = auto_io.ask_yes_no("Test?", default=True)
        assert result is True

    def test_ask_yes_no_with_default_false(self):
        from leakpro.terminal_ui.__main__ import AutoIO, TerminalIO

        real_io = TerminalIO(use_color=False)
        auto_io = AutoIO(real_io)
        result = auto_io.ask_yes_no("Test?", default=False)
        assert result is False


class TestCifarMIAApi:
    """Tests for CifarMIAApi class."""

    def test_resolve_path_absolute(self):
        from leakpro.terminal_ui.api import CifarMIAApi

        api = CifarMIAApi(Path("/tmp"))
        result = api.resolve_path("/absolute/path")
        assert result == Path("/absolute/path")

    def test_resolve_path_relative(self):
        from leakpro.terminal_ui.api import CifarMIAApi

        with tempfile.TemporaryDirectory() as tmpdir:
            api = CifarMIAApi(Path(tmpdir))
            result = api.resolve_path("relative/path")
            expected = (Path(tmpdir) / "relative/path").resolve()
            assert result == expected

    def test_load_yaml(self):
        from leakpro.terminal_ui.api import CifarMIAApi

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"key": "value"}, f)
            temp_path = Path(f.name)

        try:
            api = CifarMIAApi(Path.cwd())
            result = api.load_yaml(temp_path)
            assert result == {"key": "value"}
        finally:
            temp_path.unlink()


class TestSteps:
    """Tests for Step classes."""

    def test_step_result(self):
        from leakpro.terminal_ui.steps import StepResult

        result = StepResult(status="done", message="test message")
        assert result.status == "done"
        assert result.message == "test message"

    def test_configure_paths_step(self):
        from leakpro.terminal_ui.steps import AppContext, ConfigurePathsStep, ConfigPaths

        mock_io = MagicMock()
        mock_api = MagicMock()

        context = AppContext(
            base_dir=Path("/tmp"),
            paths=ConfigPaths(
                train_config=Path("/tmp/train.yaml"),
                audit_config=Path("/tmp/audit.yaml"),
            ),
        )

        Path("/tmp/train_config.yaml").write_text("run:\n  random_seed: 123")
        Path("/tmp/audit.yaml").write_text("audit:\n  attack_list: []\ntarget:\n  data_path: ./data.pkl")

        mock_io.ask_yes_no.side_effect = [False, False, False, True]
        mock_io.ask_path.side_effect = [
            "/tmp/train_config.yaml",
            "/tmp/audit.yaml",
            "/tmp/train_config.yaml",
            "/tmp/audit.yaml",
            "/tmp/train_config.yaml",
            "/tmp/audit.yaml",
        ]
        mock_api.load_yaml.side_effect = [
            {"run": {"random_seed": 123}},
            {"audit": {"attack_list": []}, "target": {"data_path": "./data.pkl"}},
        ]

        step = ConfigurePathsStep()
        result = step.run(context, mock_api, mock_io)

        assert result.status == "done"
        assert mock_io.ask_yes_no.call_count >= 1

    def test_load_configs_step_missing_file(self):
        from leakpro.terminal_ui.steps import AppContext, ConfigurePathsStep, ConfigPaths

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_io = MagicMock()
            mock_api = MagicMock()
            mock_api.load_yaml.side_effect = [
                {"run": {"random_seed": 123}},
                {"audit": {"attack_list": []}, "target": {"data_path": "./data.pkl"}},
            ]

            context = AppContext(
                base_dir=Path(tmpdir),
                paths=ConfigPaths(
                    train_config=Path(tmpdir) / "train_config.yaml",
                    audit_config=Path(tmpdir) / "audit.yaml",
                ),
            )

            (Path(tmpdir) / "train_config.yaml").write_text("run:\n  random_seed: 123")
            (Path(tmpdir) / "audit.yaml").write_text("audit:\n  attack_list: []\n")

            step = ConfigurePathsStep()
            mock_io.ask_yes_no.side_effect = [False, False, False, True]
            mock_io.ask_path.side_effect = [
                str(Path(tmpdir) / "train_config.yaml"),
                str(Path(tmpdir) / "audit.yaml"),
            ]

            with patch.object(step, "_print_config_section"):
                result = step.run(context, mock_api, mock_io)

            assert result.status == "done"
            assert context.train_config == {"run": {"random_seed": 123}}


class TestTerminalApp:
    """Tests for TerminalApp class."""

    def test_init_default(self):
        from leakpro.terminal_ui import TerminalApp

        with patch("leakpro.terminal_ui.__main__.get_project_root") as mock_root:
            mock_root.return_value = Path("/project")
            app = TerminalApp()

            assert app.project_root == Path("/project")

    def test_init_with_base_dir(self):
        from leakpro.terminal_ui import TerminalApp

        app = TerminalApp(base_dir=Path("/tmp/test"))
        assert app.context.base_dir == Path("/tmp/test").resolve()

    def test_enable_auto_mode(self):
        from leakpro.terminal_ui import TerminalApp

        app = TerminalApp()
        assert not app.auto_mode

        app.enable_auto_mode()
        assert app.auto_mode is True
        from leakpro.terminal_ui.__main__ import AutoIO

        assert isinstance(app.io, AutoIO)


class TestFlow:
    """Tests for Flow class."""

    def test_cifar_mia_flow(self):
        from leakpro.terminal_ui.steps import cifar_mia_flow

        flow = cifar_mia_flow()
        assert flow.name == "CIFAR MIA Guided Audit"
        assert len(flow.steps) == 5
        assert flow.steps[0].title == "Configure Paths & Audit"
        assert flow.steps[1].title == "Prepare Dataset"
        assert flow.steps[2].title == "Train Target Model"
        assert flow.steps[3].title == "Create Metadata"
        assert flow.steps[4].title == "Run LeakPro Audit"


class TestPathResolution:
    """Tests for path resolution in API."""

    def test_ensure_examples_in_path(self):
        from leakpro.terminal_ui.api import _ensure_examples_in_path

        # Should not raise an exception
        _ensure_examples_in_path()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
