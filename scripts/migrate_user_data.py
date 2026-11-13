# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Copy saved Aster data to a new directory without modifying its source.

Run once for each project or user data directory:
    uv run --no-sync python scripts/migrate_user_data.py SOURCE DESTINATION

Both paths are explicit. DESTINATION must not exist. Configuration, messages,
drafts and attachments retain their exact contents; SQLite databases are copied
through SQLite's backup API so committed WAL data is included. Rebuildable
runtime caches and database sidecars are excluded.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

_REBUILDABLE_DIRECTORIES = frozenset({"acp-code-runtime-cache"})
_SQLITE_HEADER = b"SQLite format 3\x00"
_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


@dataclass(frozen=True)
class MigrationResult:
    files: int
    databases: int
    skipped: int


def _is_link(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def _inventory(source: Path) -> tuple[list[Path], list[Path], set[Path], int]:
    def walk_error(error: OSError) -> None:
        raise error

    directories: list[Path] = []
    files: list[Path] = []
    databases: set[Path] = set()
    skipped = 0
    for current, child_dirs, child_files in os.walk(source, followlinks=False, onerror=walk_error):
        parent = Path(current)
        for name in sorted(child_dirs):
            path = parent / name
            if parent == source and name in _REBUILDABLE_DIRECTORIES:
                child_dirs.remove(name)
                skipped += 1
                continue
            if _is_link(path):
                raise ValueError("Linked directories require a separate migration")
            directories.append(path.relative_to(source))
        for name in sorted(child_files):
            path = parent / name
            if _is_link(path) or not path.is_file():
                raise ValueError("Only regular files can be migrated")
            if path.name == ".lock" or path.suffix == ".lock":
                skipped += 1
                continue
            relative = path.relative_to(source)
            files.append(relative)
            # Windows locks active shared-memory files. Identify the owning
            # database without reading its sidecars; backup reads WAL safely.
            if name.endswith(_SQLITE_SIDECAR_SUFFIXES):
                continue
            with path.open("rb") as stream:
                if stream.read(len(_SQLITE_HEADER)) == _SQLITE_HEADER:
                    databases.add(relative)
    sidecars = {
        Path(f"{path}{suffix}") for path in databases for suffix in _SQLITE_SIDECAR_SUFFIXES
    }
    skipped += sum(path in sidecars for path in files)
    return directories, [path for path in files if path not in sidecars], databases, skipped


def _copy_database(source: Path, destination: Path) -> None:
    deadline = time.monotonic() + 30.0

    def progress(_status: int, _remaining: int, _total: int) -> None:
        if time.monotonic() > deadline:
            raise TimeoutError("Database backup exceeded its time limit")

    reader = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True, timeout=5.0)
    try:
        writer = sqlite3.connect(destination)
        try:
            reader.backup(writer, pages=512, progress=progress, sleep=0.05)
            if writer.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise ValueError("Copied database failed its integrity check")
            writer.execute("PRAGMA journal_mode=DELETE")
        finally:
            writer.close()
    finally:
        reader.close()
    shutil.copystat(source, destination)


def migrate_directory(source: str | Path, destination: str | Path) -> MigrationResult:
    """Copy one directory; preserve source contents and refuse existing targets.

    The checked copy is prepared beside the destination. Publication reserves
    the destination with an exclusive mkdir, so concurrent migrations cannot
    overwrite one another. A failed publication may leave a partial destination;
    the source remains available and that destination will be refused on retry.
    """
    source = Path(source).expanduser().absolute()
    destination = Path(destination).expanduser().absolute()
    if _is_link(source) or _is_link(destination):
        raise ValueError("Source and destination must be real directories")
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if not source.is_dir():
        raise NotADirectoryError("Source must be a directory")
    if destination.exists():
        raise FileExistsError("Destination already exists; choose a new directory")
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Source and destination directories must not overlap")

    directories, files, databases, skipped = _inventory(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".aster-migration-", dir=destination.parent)).resolve()
    try:
        for relative in directories:
            (staging / relative).mkdir(parents=True, exist_ok=True)
        for relative in files:
            original = source / relative
            copied = staging / relative
            if relative in databases:
                _copy_database(original, copied)
            else:
                shutil.copy2(original, copied)
        destination.mkdir(mode=0o700)
        for child in staging.iterdir():
            child.rename(destination / child.name)
        return MigrationResult(files=len(files), databases=len(databases), skipped=skipped)
    finally:
        # Cleanup is limited to the temporary sibling created by this invocation.
        if staging.parent == destination.parent and staging.name.startswith(".aster-migration-"):
            shutil.rmtree(staging)


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy saved data into a new Aster data directory.")
    parser.add_argument("source", type=Path, help="Existing project or user data directory")
    parser.add_argument("destination", type=Path, help="New destination directory (must not exist)")
    args = parser.parse_args()
    try:
        result = migrate_directory(args.source, args.destination)
    except (OSError, sqlite3.Error, ValueError) as exc:
        # Files may contain credentials or private conversation text. Report
        # only the failure category, never exception payloads or file contents.
        parser.exit(
            1,
            f"Migration failed ({type(exc).__name__}). Source data is unchanged; "
            "check both paths and use a new destination before retrying.\n",
        )
    print(
        f"Copied {result.files} files, including {result.databases} SQLite databases; "
        f"skipped {result.skipped} rebuildable or transient entries. Source data is unchanged."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
