# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Discover and launch Aster without importing the agent runtime."""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
import zipfile
from pathlib import Path

_MAX_LAUNCHER_SCRIPT_BYTES = 16 * 1024


def _python_console_script(path: Path) -> bool:
    """Recognize Python CLI wrappers without running a discovered executable."""
    try:
        with path.open("rb") as source:
            header = source.read(2)
            if header == b"#!":
                script = source.read(_MAX_LAUNCHER_SCRIPT_BYTES)
                lines = script.split(b"\n", 3)
                shebang = lines[0].lower()
                shell_exec = len(lines) > 1 and lines[1].startswith(b"'''exec'")
                return (
                    b"python" in shebang
                    or (shell_exec and b"python" in lines[1].lower())
                    or b"aster_cli" in script
                    or b"console_scripts" in script
                )
            elif header == b"MZ":
                # Windows uv/distlib launchers append a ZIP to their PE image.
                with zipfile.ZipFile(source) as archive:
                    archive.getinfo("__main__.py")
                    return True
            else:
                return False
    except (OSError, KeyError, RuntimeError, zipfile.BadZipFile):
        return False


def _native_executable(path: Path, python_launchers: set[Path]) -> bool:
    return (
        path.is_file()
        and (os.name == "nt" or os.access(path, os.X_OK))
        and path.resolve() not in python_launchers
        and not _python_console_script(path)
    )


def _python_launchers(name: str) -> set[Path]:
    """Exclude Python entry scripts that share the native app's command name."""
    paths = {(Path(sysconfig.get_path("scripts")) / name).resolve()}
    if sys.argv and sys.argv[0]:
        entry = Path(sys.argv[0]).resolve()
        paths.add(entry)
        # Windows console entry scripts may remove their .exe suffix in argv[0].
        if os.name == "nt" and not entry.suffix:
            paths.add(entry.with_suffix(".exe"))
    return paths


def find_tui_binary() -> Path:
    """Prefer an explicit override, a bundled binary, a source build, then PATH."""
    name = "aster.exe" if os.name == "nt" else "aster"
    python_launchers = _python_launchers(name)
    override = os.environ.get("ASTER_TUI_BIN")
    if override is not None:
        candidate = Path(override).expanduser().resolve() if override.strip() else None
        if candidate is None or not _native_executable(candidate, python_launchers):
            raise RuntimeError("ASTER_TUI_BIN must point to an existing native Aster executable.")
        return candidate

    package = Path(__file__).resolve().parent
    bundled = package / "bin" / name
    if _native_executable(bundled, python_launchers):
        return bundled

    manifest = None
    for parent in package.parents:
        source = parent / "packages" / "aster-tui"
        if (source / "Cargo.toml").is_file():
            manifest = source / "Cargo.toml"
            binary = source / "target" / "release" / name
            if _native_executable(binary, python_launchers):
                return binary
            break

    # Search every PATH directory so a Python `aster` shim does not hide a native
    # Aster executable later on PATH.
    for directory in os.get_exec_path():
        candidate = Path(directory) / name
        if _native_executable(candidate, python_launchers):
            return candidate.resolve()

    if manifest is not None:
        raise RuntimeError(
            "Build Aster first: cargo build --release --manifest-path "
            f'"{manifest}". Or set ASTER_TUI_BIN to an existing native Aster executable.'
        )
    raise RuntimeError(
        "The native Aster executable was not found. Build packages/aster-tui with "
        "cargo build --release --manifest-path packages/aster-tui/Cargo.toml, then set "
        "ASTER_TUI_BIN to the built executable or place the native aster executable on PATH."
    )


def launch_tui(
    *,
    workspace: Path,
    model: str,
    client_type: str | None = None,
    sandbox: bool = False,
    resume: str | None = None,
    continue_last: bool = False,
) -> int:
    """Let Rust own the terminal and start ACP with this environment's Python."""
    arguments = [
        str(find_tui_binary()),
        "--workspace",
        str(workspace),
        "--python",
        sys.executable,
        "--model",
        model,
    ]
    if client_type is not None:
        arguments.extend(("--client-type", client_type))
    if sandbox:
        arguments.append("--sandbox")
    if resume is not None:
        arguments.extend(("--resume", resume))
    if continue_last:
        arguments.append("--continue")

    # Inherit the terminal directly. Shell quoting and redirected pipes would
    # interfere with raw keyboard input and paths containing spaces.
    child = subprocess.Popen(arguments)
    while True:
        try:
            returncode = child.wait()
            return returncode if returncode >= 0 else 128 - returncode
        except KeyboardInterrupt:
            # Console Ctrl+C is delivered to the child as well. Keep waiting
            # while it cancels work or restores the terminal; do not kill it.
            continue
