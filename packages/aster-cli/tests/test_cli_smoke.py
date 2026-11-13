# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke test: verify the aster CLI package is importable and the main entry point exists."""

import subprocess
import sys


def test_cli_importable():
    from aster_cli import main

    assert callable(main)


def test_commands_discoverable():
    from aster_cli.commands import discover_commands

    commands = list(discover_commands())
    assert len(commands) > 0
    names = [name for name, _ in commands]
    assert "start-dev" in names
    assert "eval" in names
    assert "config" in names


def test_cli_discovery_does_not_import_tracing_runtime():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import aster_cli; assert 'aster.tracing' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
