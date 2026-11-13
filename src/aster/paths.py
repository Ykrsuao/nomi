# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Well-known filesystem paths for Aster.

Two root directories:

- **User dir** — ``~/.config/aster/`` on every platform (honors
  ``XDG_CONFIG_HOME`` if set). A single, predictable location so the
  installer and the runtime always agree. Override the whole base with
  ``ASTER_USER_DIR``.

- **Project dir** (``<project-root>/.aster/``) — local to the current
  project; config, optional trace files, library code. Override with
  ``ASTER_PROJECT_DIR``.

Usage::

    from aster.paths import get_user_dir, get_project_dir

    sessions = get_user_dir("sessions")          # ~/.config/aster/sessions
    config   = get_project_dir("settings.yaml")  # <root>/.aster/settings.yaml
"""

import os
from pathlib import Path

#: Directory name used for the project-local directory.
DIR_NAME = ".aster"


def find_project_root() -> Path:
    """Find the nearest project root by walking up from the working directory.

    Include the working directory itself and fall back to it when no
    ``pyproject.toml`` is found, independently of the library's installation path.
    """
    current = Path.cwd().resolve()
    for parent in (current, *current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return current


def get_user_dir(*parts: str) -> Path:
    """Return the user-global Aster directory, optionally joined with *parts*.

    Location is ``~/.config/aster/`` on every platform (honoring
    ``XDG_CONFIG_HOME`` when set), so the path is predictable. Override
    the entire base with ``ASTER_USER_DIR``.

    Examples::

        get_user_dir()              # ~/.config/aster/
        get_user_dir("sessions")    # ~/.config/aster/sessions/
        get_user_dir("traces.db")   # ~/.config/aster/traces.db
    """
    override = os.environ.get("ASTER_USER_DIR")
    if override:
        base = Path(override)
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME")
        config_home = Path(xdg) if xdg else Path.home() / ".config"
        base = config_home / "aster"
    return base.joinpath(*parts) if parts else base


def get_project_dir(*parts: str) -> Path:
    """Return the project-local Aster directory, optionally joined with *parts*.

    Defaults to ``<project-root>/.aster/`` where the project root is
    the nearest ancestor directory containing a ``pyproject.toml``.  Override
    with the ``ASTER_PROJECT_DIR`` environment variable.

    Examples::

        get_project_dir()                   # <root>/.aster/
        get_project_dir("settings.yaml")    # <root>/.aster/settings.yaml
        get_project_dir("traces")           # <root>/.aster/traces/
    """
    base_str = os.environ.get("ASTER_PROJECT_DIR")
    base = Path(base_str) if base_str else find_project_root() / DIR_NAME
    return base.joinpath(*parts) if parts else base
