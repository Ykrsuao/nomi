# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Saved-data relocation preserves content, committed SQLite state and its source."""

import runpy
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "migrate_user_data.py"
migrate_directory = runpy.run_path(str(SCRIPT))["migrate_directory"]


def test_copy_preserves_config_drafts_binary_data_and_source(tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    values = {
        "settings.yaml": b"tui:\n  default_model: my-saved-model\n",
        "secrets.yaml": b"env:\n  PRIVATE_KEY: keep-exactly\n",
        "drafts/new.json": '{"text":"保留原文"}'.encode(),
        "attachments/image.bin": bytes(range(256)),
    }
    for relative, data in values.items():
        path = source / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)
    (source / "empty").mkdir()
    destination = tmp_path / "current"

    result = migrate_directory(source, destination)

    assert result.files == len(values)
    assert result.databases == 0
    for relative, data in values.items():
        assert (source / relative).read_bytes() == data
        assert (destination / relative).read_bytes() == data
    assert (destination / "empty").is_dir()


def test_sqlite_backup_includes_committed_wal_and_excludes_live_sidecars(tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    database = source / "session.db"
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA wal_autocheckpoint=0")
        connection.execute("CREATE TABLE messages (text TEXT)")
        connection.execute("INSERT INTO messages VALUES ('committed')")
        connection.commit()
        connection.execute("INSERT INTO messages VALUES ('pending')")
        assert database.with_suffix(".db-wal").exists()
        (source / "session.lock").write_text("active source process")
        destination = tmp_path / "current"

        result = migrate_directory(source, destination)

        assert result.databases == 1
        with sqlite3.connect(destination / "session.db") as copied:
            assert copied.execute("SELECT text FROM messages").fetchall() == [("committed",)]
            assert copied.execute("PRAGMA quick_check").fetchone() == ("ok",)
        assert not (destination / "session.db-wal").exists()
        assert not (destination / "session.db-shm").exists()
        assert not (destination / "session.lock").exists()
        assert (source / "session.lock").read_text() == "active source process"
        connection.commit()
        assert connection.execute("SELECT COUNT(*) FROM messages").fetchone() == (2,)
    finally:
        connection.close()


@pytest.mark.parametrize("existing_file", [False, True])
def test_existing_destination_is_never_overwritten(tmp_path, existing_file):
    source = tmp_path / "original"
    source.mkdir()
    (source / "saved.txt").write_text("source")
    destination = tmp_path / "current"
    if existing_file:
        destination.write_text("destination")
    else:
        destination.mkdir()

    with pytest.raises(FileExistsError):
        migrate_directory(source, destination)

    assert (source / "saved.txt").read_text() == "source"
    if existing_file:
        assert destination.read_text() == "destination"
    else:
        assert list(destination.iterdir()) == []


def test_destination_cannot_be_inside_source(tmp_path):
    source = tmp_path / "original"
    source.mkdir()

    with pytest.raises(ValueError, match="overlap"):
        migrate_directory(source, source / "nested")

    assert list(source.iterdir()) == []


def test_rebuildable_cache_is_skipped_and_preserved_in_source(tmp_path):
    source = tmp_path / "original"
    cache = source / "acp-code-runtime-cache"
    cache.mkdir(parents=True)
    (cache / "cached.bin").write_bytes(b"rebuildable")
    (source / "settings.yaml").write_text("tui: {}")
    destination = tmp_path / "current"

    result = migrate_directory(source, destination)

    assert result.skipped == 1
    assert result.files == 1
    assert not (destination / cache.name).exists()
    assert (cache / "cached.bin").read_bytes() == b"rebuildable"


def test_active_hidden_draft_lock_is_skipped_without_reading_it(tmp_path):
    source = tmp_path / "original"
    drafts = source / "tui-drafts"
    drafts.mkdir(parents=True)
    (drafts / "new.json").write_text('{"draft":"saved"}')
    lock_path = drafts / ".lock"
    lock_path.write_bytes(b"x")
    destination = tmp_path / "current"
    with lock_path.open("r+b") as lock:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)

        result = migrate_directory(source, destination)

    assert result.files == 1
    assert result.skipped == 1
    assert (destination / "tui-drafts/new.json").read_text() == '{"draft":"saved"}'
    assert not (destination / "tui-drafts/.lock").exists()
    assert lock_path.read_bytes() == b"x"


def test_symbolic_links_do_not_copy_files_outside_source(tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.txt").write_text("outside data")
    try:
        (source / "linked").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Host cannot create symbolic links")

    with pytest.raises(ValueError, match="Linked"):
        migrate_directory(source, tmp_path / "current")

    assert not (tmp_path / "current").exists()


def test_cli_reports_counts_without_printing_config_contents(tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    secret = "private-value-never-output"
    (source / "secrets.yaml").write_text(f"env:\n  KEY: {secret}\n")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(source), str(tmp_path / "current")],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Copied 1 files" in result.stdout
    assert secret not in result.stdout + result.stderr
