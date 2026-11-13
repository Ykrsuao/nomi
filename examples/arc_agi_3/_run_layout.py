# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aster output names and read-only compatibility for earlier example runs."""

from __future__ import annotations

import re
from pathlib import Path

SOLVER = "aster_single_agent"
DEFAULT_GROUP = "aster_solver"
_LEGACY_TEAM = "nemo"
_RESULTS_GROUP_PATTERN = rf"(?:aster|{re.escape(_LEGACY_TEAM)})_solver"
_WORKSPACE_PATTERN = rf"team_(?:aster|{re.escape(_LEGACY_TEAM)})"


def canonical_team(team: str) -> str:
    return "aster" if team == _LEGACY_TEAM else team


def canonical_group(group: str) -> str:
    return DEFAULT_GROUP if group == f"{_LEGACY_TEAM}_solver" else group


def is_solver(value: object) -> bool:
    return value in (SOLVER, f"{_LEGACY_TEAM}_single_agent")


def _existing_path(current: Path, legacy: Path) -> Path:
    return legacy if not current.exists() and legacy.exists() else current


def read_workspace(run_dir: Path, *parts: str) -> Path:
    """Resolve an existing workspace artifact without creating or migrating it."""
    return _existing_path(
        run_dir.joinpath("team_aster", "shared", *parts),
        run_dir.joinpath(f"team_{_LEGACY_TEAM}", "shared", *parts),
    )


def read_log(run_dir: Path, *parts: str, role: str = "team_leader") -> Path:
    """Prefer each Aster log artifact, falling back only when it is absent."""
    return _existing_path(
        run_dir.joinpath("agent_logs", "aster", role, *parts),
        run_dir.joinpath("agent_logs", _LEGACY_TEAM, role, *parts),
    )


def fleet_event_paths(container: Path) -> list[Path]:
    """Return one event stream per memory run, without counting both layouts."""
    paths = []
    for run in sorted(container.glob("*/memory/*")):
        path = read_log(run, "events.jsonl")
        if path.is_file():
            paths.append(path)
    return paths
