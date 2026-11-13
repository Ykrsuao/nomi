# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Choose Aster's configured model without loading the framework or credentials."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from aster_cli._common import find_project_root


def _config_paths(workspace: Path, filename: str, env_var: str) -> list[Path]:
    """Mirror user/project/override layers as seen from the backend's workspace cwd."""
    user_override = os.environ.get("ASTER_USER_DIR")
    xdg = os.environ.get("XDG_CONFIG_HOME")
    config_home = Path(xdg) if xdg else Path.home() / ".config"
    user_dir = Path(user_override) if user_override else config_home / "aster"
    project_override = os.environ.get("ASTER_PROJECT_DIR")
    project_dir = (
        Path(project_override) if project_override else find_project_root(workspace) / ".aster"
    )
    candidates = [user_dir / filename, project_dir / filename]
    candidates.extend(
        Path(raw.strip()).expanduser()
        for raw in os.environ.get(env_var, "").split(",")
        if raw.strip()
    )

    paths: list[Path] = []
    for candidate in candidates:
        if not candidate.is_absolute():
            candidate = workspace / candidate
        try:
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
        except OSError:
            continue
        # The shared config loader keeps the last occurrence of a resolved path.
        if resolved in paths:
            paths.remove(resolved)
        paths.append(resolved)
    return paths


def _read_config(path: Path) -> dict[str, Any]:
    import yaml

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        # YAML exceptions can contain source lines, so keep config values out of
        # the user-facing diagnostic.
        raise ValueError(
            f"Cannot read model configuration in {path}. Fix it or pass --model."
        ) from exc
    return data if isinstance(data, dict) else {}


def resolve_tui_model(workspace: Path) -> str:
    """Return a settings default or the sole explicitly configured chat alias.

    Called only when the CLI and model environment variables supply no model.
    Bundled providers are not user preferences and are never loaded here.
    """
    workspace = workspace.resolve()
    default: object = None
    for path in _config_paths(workspace, "settings.yaml", "ASTER_SETTINGS"):
        settings = _read_config(path)
        if "tui" not in settings:
            continue
        tui = settings["tui"]
        if not isinstance(tui, dict):
            default = None
        elif "default_model" in tui:
            default = tui["default_model"]
    if default is not None:
        if not isinstance(default, str) or not default.strip():
            raise ValueError("tui.default_model must be a nonblank model name, or pass --model.")
        return default.strip()

    aliases: dict[str, bool] = {}
    for path in _config_paths(workspace, "llm_config.yaml", "ASTER_LLM_CONFIG"):
        models = _read_config(path).get("models")
        if not isinstance(models, dict):
            continue
        for name, config in models.items():
            if not isinstance(name, str) or not name.strip():
                continue
            if isinstance(config, dict):
                aliases[name] = config.get("client_type", "completion") in (
                    "completion",
                    "responses",
                )
            else:
                aliases.pop(name, None)
    candidates = sorted(name for name, eligible in aliases.items() if eligible)
    if len(candidates) == 1:
        return candidates[0]

    instruction = "Pass --model, set ASTER_MODEL, or set tui.default_model in .aster/settings.yaml."
    if not candidates:
        raise ValueError(f"No default or configured chat model was found. {instruction}")
    names = ", ".join(candidates[:5]) + (", ..." if len(candidates) > 5 else "")
    raise ValueError(f"Multiple configured chat models ({names}). {instruction}")
