# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Discover distributions for exposed modules without statting unrelated files."""

from __future__ import annotations

import csv
import importlib.metadata as metadata
import inspect
import sys
from typing import Protocol, cast


class _LocatedFile(Protocol):
    # PathDistribution.files requires exists(), but its path stub omits it.
    def exists(self) -> bool: ...


def _inferred_root(path: metadata.PackagePath) -> str | None:
    # Match the supported stdlib versions, including extensionless files in 3.13.
    if sys.version_info >= (3, 13):
        top, *rest = path.parts
        name = top if rest else inspect.getmodulename(path) or str(path)
    else:
        name = path.parts[0] if len(path.parts) > 1 else inspect.getmodulename(path)
    return name if name is not None and "." not in name else None


def _record_matches(
    distribution: metadata.PathDistribution, record: str, roots: set[str]
) -> set[str] | None:
    """Find one existing recorded file per root, or request the stdlib fallback."""
    found: set[str] = set()
    try:
        for row in csv.reader(record.splitlines()):
            # Leave malformed metadata to the standard reader, including its
            # errors. Validate even unrelated rows, as Distribution.files does.
            if not 1 <= len(row) <= 3:
                return None
            if len(row) == 3 and row[2]:
                int(row[2])
            path = metadata.PackagePath(row[0])
            root = _inferred_root(path)
            if root is not None and root in roots and root not in found:
                located = cast(_LocatedFile, distribution.locate_file(path))
                if located.exists():
                    found.add(root)
    except (csv.Error, ValueError):
        return None
    return found


def packages_distributions(roots: set[str]) -> dict[str, list[str]]:
    """Project importlib.metadata.packages_distributions() onto module roots.

    Read current metadata each time so installs, version changes and sys.path
    changes remain visible. The standard PathDistribution uses a targeted RECORD
    scan; custom providers and legacy egg manifests keep their files behavior.
    Every distribution is considered because namespaces can have many providers.
    """
    result: dict[str, list[str]] = {}
    if not roots:
        return result
    for distribution in metadata.distributions():
        declared = (distribution.read_text("top_level.txt") or "").split()
        matches: list[str] | set[str] | None
        if declared:
            # Declared roots take precedence, including duplicates and editables
            # whose sources are outside the installed distribution directory.
            matches = [root for root in declared if root in roots]
        else:
            matches = None
            if type(distribution) is metadata.PathDistribution:
                record = distribution.read_text("RECORD")
                if record:
                    matches = _record_matches(distribution, record, roots)
            if matches is None:
                matches = {
                    root
                    for path in distribution.files or ()
                    if (root := _inferred_root(path)) is not None and root in roots
                }
        if matches:
            name = distribution.metadata["Name"]
            for root in matches:
                result.setdefault(root, []).append(name)
    return result
