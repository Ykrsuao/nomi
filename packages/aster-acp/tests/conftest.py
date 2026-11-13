# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep protocol subprocesses on the checkout and configuration under test."""

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _protocol_subprocess_environment(monkeypatch):
    """Preserve test sources/config through ACP's sanitized subprocess environment."""
    import acp.transports

    original = acp.transports.default_environment
    root = Path(__file__).resolve().parents[3]
    sources = [root / "src"] + [
        root / "packages" / package / "src"
        for package in ("aster-cli", "aster-acp", "aster-memory", "aster-bench")
    ]

    def environment():
        values = original()
        values["PYTHONPATH"] = os.pathsep.join(str(path) for path in sources)
        for name in (
            "ASTER_USER_DIR",
            "ASTER_PROJECT_DIR",
            "ASTER_SETTINGS",
            "ASTER_SESSIONS_DIR",
            "ASTER_ACP_MCP_TRACE",
        ):
            if name in os.environ:
                values[name] = os.environ[name]
        return values

    monkeypatch.setattr(acp.transports, "default_environment", environment)
