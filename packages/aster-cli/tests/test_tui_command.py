# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Terminal-chat command validation, configuration, and lazy startup."""

import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from aster_cli import _tui_launcher as launcher
from aster_cli import cli
from aster_cli.commands import tui
from click.testing import CliRunner


@pytest.fixture(autouse=True)
def clean_tui_environment(monkeypatch, tmp_path):
    for suffix in ("MODEL", "SANDBOX", "TUI_BIN"):
        monkeypatch.delenv(f"ASTER_{suffix}", raising=False)
    for name in ("ASTER_SETTINGS", "ASTER_LLM_CONFIG", "XDG_CONFIG_HOME"):
        monkeypatch.delenv(name, raising=False)
    for name, directory in (("ASTER_USER_DIR", "user"), ("ASTER_PROJECT_DIR", "project")):
        config_dir = tmp_path / directory
        config_dir.mkdir()
        monkeypatch.setenv(name, str(config_dir))


@pytest.fixture(params=[tui.command, cli], ids=["tui-subcommand", "aster"])
def terminal_command(request):
    return request.param


@pytest.fixture
def terminal_app(monkeypatch):
    calls = []

    def run_tui(**kwargs):
        calls.append(kwargs)
        return 0

    monkeypatch.setattr(launcher, "launch_tui", run_tui)
    original_find_spec = tui.importlib.util.find_spec
    monkeypatch.setattr(
        tui.importlib.util,
        "find_spec",
        lambda name: (
            SimpleNamespace(name=name) if name in {"acp", "aster_acp"} else original_find_spec(name)
        ),
    )
    monkeypatch.setattr(
        tui,
        "sys",
        SimpleNamespace(
            stdin=SimpleNamespace(isatty=lambda: True),
            stdout=SimpleNamespace(isatty=lambda: True),
        ),
    )
    return calls


def test_tui_help_and_discovery_do_not_import_runtime():
    script = """
import sys
from click.testing import CliRunner
from aster_cli import cli
result = CliRunner().invoke(cli, ['tui', '--help'])
assert result.exit_code == 0, result.output
assert '--resume' in result.output
assert '--sandbox' in result.output
assert '--dev' not in result.output
assert '--no-dev' not in result.output
assert '--working-dir' in result.output
assert '--version' not in result.output
for name in ('aster', 'aster_cli._tui_model', 'prompt_toolkit', 'rich', 'aster_acp.server'):
    assert name not in sys.modules, name
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_toolkit_commands_work_without_starting_a_terminal(terminal_app, monkeypatch):
    monkeypatch.setenv("ASTER_MODEL", "configured-model")
    monkeypatch.setenv("ASTER_SANDBOX", "true")

    result = CliRunner().invoke(cli, ["completion", "bash"])

    assert result.exit_code == 0, result.output
    assert "_ASTER_COMPLETE=bash_complete" in result.output
    assert terminal_app == []


@pytest.mark.parametrize(
    "arguments",
    [
        ["--model", "chat"],
        ["-w", "."],
        ["--client-type", "responses"],
        ["--sandbox"],
        ["--resume", "saved-session"],
        ["--continue"],
    ],
)
def test_terminal_options_with_a_subcommand_are_not_silently_ignored(terminal_app, arguments):
    result = CliRunner().invoke(cli, [*arguments, "completion", "bash"])

    assert result.exit_code == 2
    assert "Use terminal options without a subcommand" in result.output
    assert terminal_app == []


def test_unknown_subcommand_does_not_start_a_terminal(terminal_app):
    result = CliRunner().invoke(cli, ["nonexistent-command"])

    assert result.exit_code == 2
    assert "No such command" in result.output
    assert terminal_app == []


@pytest.mark.parametrize(
    ("entrypoint", "arguments", "expected"),
    [
        ("main", ["tui", "--help"], "Usage: aster tui [OPTIONS]"),
        ("main", ["--help"], "Usage: aster [OPTIONS]"),
        ("main", ["-h"], "Usage: aster [OPTIONS]"),
        ("main", ["--version"], "Aster, version "),
        ("aster_main", ["--help"], "Usage: aster [OPTIONS]"),
    ],
)
def test_console_entrypoints_keep_help_and_version_lazy(entrypoint, arguments, expected):
    script = """
import sys
import aster_cli
entrypoint = sys.argv.pop(1)
sys.argv[0] = 'aster'
try:
    getattr(aster_cli, entrypoint)()
except SystemExit as exc:
    assert exc.code == 0, exc.code
for name in (
    'aster', 'aster.secrets', 'aster_cli._tui_model', 'litellm',
    'prompt_toolkit', 'rich', 'aster_acp.server',
):
    assert name not in sys.modules, name
"""
    result = subprocess.run(
        [sys.executable, "-c", script, entrypoint, *arguments],
        capture_output=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert expected in result.stdout
    if "--version" not in arguments:
        assert "Open Aster" in result.stdout


@pytest.mark.parametrize("entrypoint", ["main", "aster_main"])
@pytest.mark.parametrize("model_source", ["argument", "settings", "singleton"])
def test_tui_launch_leaves_secrets_loading_to_agent_process(entrypoint, model_source, tmp_path):
    if model_source == "settings":
        (tmp_path / "project/settings.yaml").write_text(
            "tui:\n  default_model: configured-model\n", encoding="utf-8"
        )
    elif model_source == "singleton":
        (tmp_path / "project/llm_config.yaml").write_text(
            "models:\n  configured-model: {}\n", encoding="utf-8"
        )
    script = """
import sys
from pathlib import Path
from types import SimpleNamespace
import aster_cli
from aster_cli.commands import tui
from aster_cli import _tui_launcher as launcher
entrypoint, model_source = sys.argv[1:]
calls = []
read_config_names = []
original_read_text = Path.read_text

def read_text(path, *args, **kwargs):
    assert path.name != 'secrets.yaml', 'launcher must not read credentials'
    read_config_names.append(path.name)
    return original_read_text(path, *args, **kwargs)

Path.read_text = read_text

def run_tui(**kwargs):
    calls.append(kwargs)
    return 0
launcher.launch_tui = run_tui
tui.sys = SimpleNamespace(
    stdin=SimpleNamespace(isatty=lambda: True),
    stdout=SimpleNamespace(isatty=lambda: True),
)
tui.importlib.util.find_spec = lambda name: object()
sys.argv = ['aster'] if entrypoint == 'aster_main' else ['aster', 'tui']
if model_source == 'argument':
    sys.argv.extend(['--model', 'configured-model'])
try:
    getattr(aster_cli, entrypoint)()
except SystemExit as exc:
    assert exc.code == 0, exc.code
assert len(calls) == 1
assert calls[0]['model'] == 'configured-model'
for name in ('aster', 'aster.secrets', 'litellm', 'aster_acp.server'):
    assert name not in sys.modules, name
if model_source == 'argument':
    assert 'aster_cli._tui_model' not in sys.modules
else:
    expected_file = 'settings.yaml' if model_source == 'settings' else 'llm_config.yaml'
    assert expected_file in read_config_names
"""
    result = subprocess.run(
        [sys.executable, "-c", script, entrypoint, model_source],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_tui_requires_a_default_or_single_configured_model(terminal_command, terminal_app):
    result = CliRunner().invoke(terminal_command, [])

    assert result.exit_code == 2
    assert "--model" in result.output
    assert "tui.default_model" in result.output
    assert "ASTER_MODEL" in result.output
    assert terminal_app == []


def test_tui_bare_command_forwards_saved_default(terminal_command, terminal_app, tmp_path):
    (tmp_path / "project/settings.yaml").write_text(
        "tui:\n  default_model: selected-default\n", encoding="utf-8"
    )

    result = CliRunner().invoke(terminal_command, [])

    assert result.exit_code == 0, result.output
    assert terminal_app[0]["model"] == "selected-default"


def test_tui_bare_command_forwards_single_configured_alias(
    terminal_command, terminal_app, tmp_path
):
    (tmp_path / "user/llm_config.yaml").write_text("models:\n  only-chat: {}\n", encoding="utf-8")

    result = CliRunner().invoke(terminal_command, [])

    assert result.exit_code == 0, result.output
    assert terminal_app[0]["model"] == "only-chat"


def test_tui_ambiguous_models_explain_how_to_choose(terminal_command, terminal_app, tmp_path):
    (tmp_path / "project/llm_config.yaml").write_text(
        "models:\n  first-chat: {}\n  second-chat: {}\n", encoding="utf-8"
    )

    result = CliRunner().invoke(terminal_command, [])

    assert result.exit_code == 2
    assert "Multiple configured chat models" in result.output
    assert "first-chat" in result.output
    assert "second-chat" in result.output
    assert "tui.default_model" in result.output
    assert "--model" in result.output
    assert terminal_app == []


def test_tui_default_comes_from_selected_workspace_without_changing_cwd(
    terminal_command, terminal_app, tmp_path, monkeypatch
):
    monkeypatch.delenv("ASTER_PROJECT_DIR")
    selected = tmp_path / "selected workspace"
    workspace = selected / "src/nested"
    workspace.mkdir(parents=True)
    caller = tmp_path / "caller"
    caller.mkdir()
    for root, model in ((selected, "selected-default"), (caller, "caller-default")):
        (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        (root / ".aster").mkdir()
        (root / ".aster/settings.yaml").write_text(
            f"tui:\n  default_model: {model}\n", encoding="utf-8"
        )
    monkeypatch.chdir(caller)

    result = CliRunner().invoke(terminal_command, ["-w", str(workspace)])

    assert result.exit_code == 0, result.output
    assert terminal_app[0]["workspace"] == workspace.resolve()
    assert terminal_app[0]["model"] == "selected-default"
    assert Path.cwd() == caller


@pytest.mark.parametrize(
    ("arguments", "aster_model", "expected"),
    [
        (["--model", "explicit-model"], "aster-model", "explicit-model"),
        ([], "aster-model", "aster-model"),
    ],
)
def test_explicit_model_precedence_bypasses_malformed_configuration(
    terminal_command,
    terminal_app,
    tmp_path,
    monkeypatch,
    arguments,
    aster_model,
    expected,
):
    for filename in ("settings.yaml", "llm_config.yaml"):
        (tmp_path / "project" / filename).write_text("invalid: [unfinished\n", encoding="utf-8")
    monkeypatch.setenv("ASTER_MODEL", aster_model)

    result = CliRunner().invoke(terminal_command, arguments)

    assert result.exit_code == 0, result.output
    assert terminal_app[0]["model"] == expected


@pytest.mark.parametrize("filename", ["settings.yaml", "llm_config.yaml"])
def test_tui_default_parse_error_is_sanitized(terminal_command, terminal_app, tmp_path, filename):
    path = tmp_path / "project" / filename
    path.write_text("sensitive_test_value: [unfinished\n", encoding="utf-8")

    result = CliRunner().invoke(terminal_command, [])

    assert result.exit_code == 2
    assert str(path) in result.output
    assert "--model" in result.output
    assert "sensitive_test_value" not in result.output
    assert "Traceback" not in result.output
    assert terminal_app == []


def test_tui_forwards_workspace_and_configured_model(
    terminal_command, terminal_app, tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace with spaces"
    workspace.mkdir()
    monkeypatch.setenv("ASTER_MODEL", "my-model")
    monkeypatch.setenv("ASTER_SANDBOX", "true")

    result = CliRunner().invoke(
        terminal_command,
        ["-w", str(workspace), "--client-type", "responses", "--resume", "saved-123"],
    )

    assert result.exit_code == 0, result.output
    assert terminal_app == [
        {
            "workspace": workspace.resolve(),
            "model": "my-model",
            "client_type": "responses",
            "sandbox": True,
            "resume": "saved-123",
            "continue_last": False,
        }
    ]


def test_tui_defaults_to_current_workspace_and_native_execution(
    terminal_command, terminal_app, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(terminal_command, ["--model", "my-model"])

    assert result.exit_code == 0, result.output
    assert terminal_app[0]["workspace"] == tmp_path.resolve()
    assert terminal_app[0]["sandbox"] is False
    assert terminal_app[0]["resume"] is None
    assert terminal_app[0]["client_type"] is None


@pytest.mark.parametrize("option", ["--dev", "--no-dev"])
def test_tui_rejects_removed_developer_options(terminal_command, terminal_app, option):
    result = CliRunner().invoke(terminal_command, ["--model", "my-model", option])

    assert result.exit_code == 2
    assert "No such option" in result.output
    assert option in result.output
    assert terminal_app == []


@pytest.mark.parametrize("environment", ["true", "false", "1", "invalid"])
def test_tui_ignores_retired_developer_environment(
    terminal_command, terminal_app, monkeypatch, environment
):
    monkeypatch.setenv("ASTER_DEV", environment)
    result = CliRunner().invoke(terminal_command, ["--model", "my-model"])

    assert result.exit_code == 0, result.output
    assert terminal_app[0]["model"] == "my-model"
    assert "dev" not in terminal_app[0]


def test_tui_environment_can_disable_sandbox(terminal_command, terminal_app, monkeypatch):
    monkeypatch.setenv("ASTER_MODEL", "aster-model")
    monkeypatch.setenv("ASTER_SANDBOX", "false")

    result = CliRunner().invoke(terminal_command, [])

    assert result.exit_code == 0, result.output
    assert terminal_app[0]["model"] == "aster-model"
    assert terminal_app[0]["sandbox"] is False


def test_tui_native_flag_overrides_environment_and_continue_reaches_app(
    terminal_command, terminal_app, monkeypatch
):
    monkeypatch.setenv("ASTER_MODEL", "environment-model")
    monkeypatch.setenv("ASTER_SANDBOX", "true")

    result = CliRunner().invoke(
        terminal_command, ["--model", "selected-model", "--no-sandbox", "--continue"]
    )

    assert result.exit_code == 0, result.output
    assert terminal_app[0]["model"] == "selected-model"
    assert terminal_app[0]["sandbox"] is False
    assert terminal_app[0]["continue_last"] is True


@pytest.mark.parametrize(
    "arguments",
    [
        ["--resume", "saved-123", "--continue"],
        ["--resume", " "],
        ["--model", " "],
    ],
)
def test_tui_rejects_invalid_resume_or_model_options(terminal_command, terminal_app, arguments):
    result = CliRunner().invoke(terminal_command, ["--model", "my-model", *arguments])

    assert result.exit_code == 2
    assert terminal_app == []


def test_tui_rejects_file_as_workspace(terminal_command, terminal_app, tmp_path):
    file = tmp_path / "file.txt"
    file.write_text("not a directory")

    result = CliRunner().invoke(terminal_command, ["--model", "my-model", "-w", str(file)])

    assert result.exit_code == 2
    assert terminal_app == []


@pytest.mark.parametrize("stream", ["stdin", "stdout"])
def test_tui_rejects_redirected_stream_before_startup(
    terminal_command, terminal_app, monkeypatch, stream
):
    monkeypatch.setattr(tui.sys, stream, SimpleNamespace(isatty=lambda: False))

    result = CliRunner().invoke(terminal_command, ["--model", "my-model"])

    assert result.exit_code == 2
    assert "interactive terminal" in result.output
    assert terminal_app == []


@pytest.mark.parametrize("missing", ["acp", "aster_acp"])
def test_tui_explains_missing_optional_package(
    terminal_command, terminal_app, monkeypatch, missing
):
    original = tui.importlib.util.find_spec
    monkeypatch.setattr(
        tui.importlib.util, "find_spec", lambda name: None if name == missing else original(name)
    )

    result = CliRunner().invoke(terminal_command, ["--model", "my-model"])

    assert result.exit_code == 1
    assert "uv sync --extra tui" in result.output
    assert terminal_app == []


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("Agent process could not start"),
        OSError("Rust executable could not start"),
    ],
)
def test_tui_reports_startup_failure_without_traceback(
    terminal_command, terminal_app, monkeypatch, failure
):
    def run_tui(**kwargs):
        raise failure

    monkeypatch.setattr(launcher, "launch_tui", run_tui)
    result = CliRunner().invoke(terminal_command, ["--model", "my-model"])

    assert result.exit_code == 1
    assert str(failure) in result.output
    assert "Traceback" not in result.output


def test_tui_preserves_rust_exit_status(terminal_command, terminal_app, monkeypatch):
    monkeypatch.setattr(launcher, "launch_tui", lambda **kwargs: 7)
    result = CliRunner().invoke(
        terminal_command, ["--model", "my-model", "--resume", "saved-session"]
    )

    assert result.exit_code == 7
    assert "Traceback" not in result.output


def executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test executable")
    path.chmod(0o755)
    return path.resolve()


def binary_name(base: str = "aster") -> str:
    return f"{base}.exe" if launcher.os.name == "nt" else base


def python_console_launcher(path: Path, launcher_format: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    script = b"#!/usr/bin/env python3\nfrom foreign_cli import main\nraise SystemExit(main())\n"
    if launcher_format == "windows-uv":
        # uv uses a PE launcher with an appended ZIP containing __main__.py.
        path.write_bytes(b"MZ" + bytes(64))
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("__main__.py", script)
    elif launcher_format == "posix-shell":
        # Python console scripts use a shell trampoline for long interpreter paths.
        path.write_bytes(
            b"#!/bin/sh\n'''exec' \"/path with spaces/bin/python\" \"$0\" \"$@\"\n' '''\n"
            + script.split(b"\n", 1)[1]
        )
    else:
        path.write_bytes(script)
    path.chmod(0o755)
    return path.resolve()


def test_explicit_binary_path_wins_and_invalid_override_does_not_fall_back(tmp_path, monkeypatch):
    binary = executable(tmp_path / "path with spaces" / binary_name())
    variable = "ASTER_TUI_BIN"
    monkeypatch.setenv(variable, str(binary))
    assert launcher.find_tui_binary() == binary
    for invalid in ("", " ", str(tmp_path / "missing.exe")):
        monkeypatch.setenv(variable, invalid)
        with pytest.raises(RuntimeError, match=variable):
            launcher.find_tui_binary()


def test_source_checkout_uses_release_build_and_explains_missing_binary(tmp_path, monkeypatch):
    package = tmp_path / "packages/aster-cli/src/aster_cli"
    package.mkdir(parents=True)
    manifest = tmp_path / "packages/aster-tui/Cargo.toml"
    manifest.parent.mkdir()
    manifest.write_text("[package]\nname = 'aster-tui'\n")
    monkeypatch.setattr(launcher, "__file__", str(package / "_tui_launcher.py"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    with pytest.raises(RuntimeError, match="cargo build --release"):
        launcher.find_tui_binary()

    binary = executable(manifest.parent / "target/release" / binary_name())
    assert launcher.find_tui_binary() == binary
    bundled = executable(package / "bin" / binary_name())
    assert launcher.find_tui_binary() == bundled


def test_installed_package_uses_path(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "__file__", str(tmp_path / "installed/_tui_launcher.py"))
    binary = executable(tmp_path / "bin" / binary_name())
    monkeypatch.setenv("PATH", str(binary.parent))
    assert launcher.find_tui_binary() == binary


def test_path_skips_python_launcher_and_prefers_native_aster(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "__file__", str(tmp_path / "installed/_tui_launcher.py"))
    shim = executable(tmp_path / "scripts" / binary_name())
    monkeypatch.setattr(launcher.sysconfig, "get_path", lambda key: str(shim.parent))
    binary = executable(tmp_path / "native" / binary_name())
    monkeypatch.setenv(
        "PATH", launcher.os.pathsep.join(str(path.parent) for path in (shim, binary))
    )

    assert launcher.find_tui_binary() == binary


def test_path_skips_current_launcher_even_with_stripped_windows_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "__file__", str(tmp_path / "installed/_tui_launcher.py"))
    shim = executable(tmp_path / "entry-script" / binary_name())
    entry = shim.with_suffix("") if launcher.os.name == "nt" else shim
    monkeypatch.setattr(sys, "argv", [str(entry)])
    binary = executable(tmp_path / "native" / binary_name())
    monkeypatch.setenv("PATH", launcher.os.pathsep.join((str(shim.parent), str(binary.parent))))

    assert launcher.find_tui_binary() == binary


def test_python_launcher_on_path_reports_missing_binary(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "__file__", str(tmp_path / "installed/_tui_launcher.py"))
    shim = executable(tmp_path / "scripts" / binary_name())
    monkeypatch.setattr(launcher.sysconfig, "get_path", lambda key: str(shim.parent))
    monkeypatch.setenv("PATH", str(shim.parent))

    with pytest.raises(RuntimeError, match="native Aster executable was not found"):
        launcher.find_tui_binary()


def test_override_rejects_python_launcher(tmp_path, monkeypatch):
    variable = "ASTER_TUI_BIN"
    shim = executable(tmp_path / "scripts" / binary_name())
    monkeypatch.setattr(launcher.sysconfig, "get_path", lambda key: str(shim.parent))
    monkeypatch.setenv(variable, str(shim))

    with pytest.raises(RuntimeError, match=variable):
        launcher.find_tui_binary()


@pytest.mark.parametrize("launcher_format", ["windows-uv", "posix-script", "posix-shell"])
def test_path_skips_foreign_python_launcher_for_native_binary(
    tmp_path, monkeypatch, launcher_format
):
    monkeypatch.setattr(launcher, "__file__", str(tmp_path / "installed/_tui_launcher.py"))
    shim = python_console_launcher(tmp_path / "foreign-env" / binary_name(), launcher_format)
    binary = executable(tmp_path / "native" / binary_name())
    monkeypatch.setenv("PATH", launcher.os.pathsep.join((str(shim.parent), str(binary.parent))))

    assert launcher.find_tui_binary() == binary


@pytest.mark.parametrize("launcher_format", ["windows-uv", "posix-script", "posix-shell"])
def test_only_foreign_python_launcher_on_path_reports_missing_binary(
    tmp_path, monkeypatch, launcher_format
):
    monkeypatch.setattr(launcher, "__file__", str(tmp_path / "installed/_tui_launcher.py"))
    shim = python_console_launcher(tmp_path / "foreign-env" / binary_name(), launcher_format)
    monkeypatch.setenv("PATH", str(shim.parent))

    with pytest.raises(RuntimeError, match="native Aster executable was not found"):
        launcher.find_tui_binary()


@pytest.mark.parametrize("launcher_format", ["windows-uv", "posix-script", "posix-shell"])
def test_override_rejects_foreign_python_launcher(tmp_path, monkeypatch, launcher_format):
    variable = "ASTER_TUI_BIN"
    shim = python_console_launcher(tmp_path / "foreign-env" / binary_name(), launcher_format)
    monkeypatch.setenv(variable, str(shim))

    with pytest.raises(RuntimeError, match=variable):
        launcher.find_tui_binary()


def test_launcher_preserves_arguments_with_spaces_and_waits_through_parent_interrupt(
    tmp_path, monkeypatch
):
    binary = tmp_path / "bin with spaces" / binary_name()
    workspace = tmp_path / "工作目录 with spaces"
    monkeypatch.setattr(launcher, "find_tui_binary", lambda: binary)
    calls = []
    outcomes = iter([KeyboardInterrupt(), 0])

    def wait():
        result = next(outcomes)
        if isinstance(result, BaseException):
            raise result
        return result

    def popen(arguments):
        calls.append(arguments)
        return SimpleNamespace(wait=wait)

    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    assert (
        launcher.launch_tui(
            workspace=workspace,
            model="alias with space",
            sandbox=True,
            client_type="responses",
            resume="session-1",
        )
        == 0
    )
    assert calls == [
        [
            str(binary),
            "--workspace",
            str(workspace),
            "--python",
            sys.executable,
            "--model",
            "alias with space",
            "--client-type",
            "responses",
            "--sandbox",
            "--resume",
            "session-1",
        ]
    ]
