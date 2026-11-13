# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for shared durable coding-agent session storage."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime

import aster_cli.sessions.store as store_module
import pytest
from aster_cli.sessions import InvalidSessionIdError, SessionNotFoundError, SessionStore

from aster.interactive import AgentMessage
from aster.storage import SessionAlreadyActiveError


def test_create_record_title_list_and_resume(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create(
        session_id="session-one",
        origin="tui",
        model="test/model",
        agent="CodingAgent",
        working_directory="/workspace",
    )
    session.record_user_message("hello")
    session.events.add(AgentMessage(content="hi back"))
    session.set_title("First session", user_set=True)
    session.close()

    info = store.list()[0]
    assert info.id == "session-one"
    assert info.origin == "tui"
    assert info.model == "test/model"
    assert info.agent == "CodingAgent"
    assert info.working_directory == "/workspace"
    assert info.title == "First session"
    assert info.title_is_user_set is True
    assert info.turn_count == 1

    resumed = store.open("session-one")
    try:
        assert resumed.info == info
        assert [(turn.role, turn.content) for turn in resumed.turns()] == [
            ("user", "hello"),
            ("agent", "hi back"),
        ]
    finally:
        resumed.close()


def test_session_summary_uses_targeted_queries(tmp_path, monkeypatch):
    store = SessionStore(tmp_path)
    session = store.create(session_id="query-shape")
    session.record_user_message("hello")
    session.set_title("Targeted")
    session.close()

    queries: list[str] = []
    real_connect = sqlite3.connect

    class RecordingConnection:
        def __init__(self, *args, **kwargs):
            self._connection = real_connect(*args, **kwargs)

        def execute(self, query, parameters=()):
            queries.append(" ".join(query.split()))
            return self._connection.execute(query, parameters)

        def close(self):
            self._connection.close()

    monkeypatch.setattr(store_module.sqlite3, "connect", RecordingConnection)

    assert store.get("query-shape").title == "Targeted"
    assert not any(
        query == "SELECT event_type, data FROM events ORDER BY insertion_order" for query in queries
    )
    assert any("SELECT COUNT(*)" in query for query in queries)
    assert sum("LIMIT 1" in query for query in queries) == 2


def test_unused_provisional_session_is_hidden_and_removed_on_close(tmp_path):
    store = SessionStore(tmp_path)
    prepared = store.create(session_id="prepared", provisional=True)
    path = prepared.path
    assert path.exists()
    assert prepared.info.provisional is True
    assert store.get(prepared.id).provisional is True
    assert store.list() == []
    prepared.close()
    assert not path.exists()
    assert not path.with_name(f"{path.name}-wal").exists()
    assert not path.with_name(f"{path.name}-shm").exists()

    # A second close must never delete a later session using the same ID.
    replacement = store.create(session_id="prepared")
    replacement.close()
    prepared.close()
    assert store.get("prepared").provisional is False


def test_provisional_session_becomes_durable_on_first_user_message(tmp_path):
    store = SessionStore(tmp_path)
    prepared = store.create(session_id="prepared", provisional=True)
    prepared.record_user_message("keep this even if generation fails")
    assert prepared.info.turn_count == 1
    assert [item.id for item in store.list()] == [prepared.id]
    prepared.close()

    resumed = store.open("prepared")
    assert resumed.info.turn_count == 1
    assert [turn.content for turn in resumed.turns()] == ["keep this even if generation fails"]
    resumed.close()
    assert store.get("prepared").turn_count == 1


def test_provisional_session_does_not_displace_saved_sessions_at_list_limit(tmp_path):
    store = SessionStore(tmp_path)
    with store.create(session_id="ordinary-empty"):
        pass
    with store.create(session_id="saved") as saved:
        saved.record_user_message("old conversation")
    prepared = store.create(session_id="newest-but-empty", provisional=True)
    # Re-read from disk using a separate store, as a new process would after a crash.
    another_reader = SessionStore(tmp_path)
    assert [item.id for item in another_reader.list(limit=1)] == ["saved"]
    assert {item.id for item in another_reader.list()} == {"saved", "ordinary-empty"}
    prepared.close()


def test_failed_first_user_write_does_not_make_provisional_session_durable(tmp_path, monkeypatch):
    store = SessionStore(tmp_path)
    prepared = store.create(session_id="failed-write", provisional=True)

    def fail_write(_event):
        raise OSError("disk unavailable")

    monkeypatch.setattr(prepared.events, "add", fail_write)
    with pytest.raises(OSError, match="disk unavailable"):
        prepared.record_user_message("not saved")
    assert prepared.info.turn_count == 0
    prepared.close()
    assert not prepared.path.exists()


def test_provisional_close_waits_for_first_user_message_commit(tmp_path, monkeypatch):
    store = SessionStore(tmp_path)
    prepared = store.create(session_id="concurrent", provisional=True, check_same_thread=False)
    written = threading.Event()
    finish_write = threading.Event()
    closing = threading.Event()
    closed = threading.Event()
    errors: list[Exception] = []
    real_add = prepared.events.add

    def paused_add(event):
        tag = real_add(event)
        written.set()
        assert finish_write.wait(5)
        return tag

    def write():
        try:
            prepared.record_user_message("accepted before shutdown")
        except Exception as error:
            errors.append(error)

    def close():
        closing.set()
        try:
            prepared.close()
        except Exception as error:
            errors.append(error)
        finally:
            closed.set()

    monkeypatch.setattr(prepared.events, "add", paused_add)
    writer = threading.Thread(target=write)
    closer = threading.Thread(target=close)
    writer.start()
    try:
        assert written.wait(5)
        closer.start()
        assert closing.wait(5)
        assert not closed.wait(0.05)
    finally:
        finish_write.set()
        writer.join(timeout=5)
        if closer.ident is not None:
            closer.join(timeout=5)
        prepared.close()
    assert errors == []
    assert not writer.is_alive() and not closer.is_alive()
    assert store.get("concurrent").turn_count == 1


def test_different_session_databases_can_be_open_together(tmp_path):
    store = SessionStore(tmp_path)
    first = store.create(session_id="first")
    second = store.create(session_id="second")
    try:
        first.record_user_message("one")
        second.record_user_message("two")
        assert [turn.content for turn in first.turns()] == ["one"]
        assert [turn.content for turn in second.turns()] == ["two"]
    finally:
        first.close()
        second.close()


def test_transcript_reader_is_readonly_and_excludes_internal_events(tmp_path, monkeypatch):
    from aster_cli.sessions import SessionActivity

    from aster.events import DebugTrace

    store = SessionStore(tmp_path)
    with store.create(session_id="transcript") as session:
        session.record_user_message("hello")
        session.events.add(DebugTrace(content="private provider payload"))
        session.events.add(
            SessionActivity(activity={"event_type": "FileEdit", "path": "saved.txt"})
        )
        session.events.add(
            SessionActivity(activity={"event_type": "DebugTrace", "content": "hidden"})
        )
        real_connect = sqlite3.connect
        calls = []

        def connect(database, **kwargs):
            calls.append((database, kwargs))
            return real_connect(database, **kwargs)

        monkeypatch.setattr(store_module.sqlite3, "connect", connect)
        assert [item.event_type for item in store.load_transcript(session.id)] == [
            "SessionUserMessage",
            "FileEdit",
        ]
        assert calls and all(
            database.endswith("?mode=ro") and kwargs.get("uri") for database, kwargs in calls
        )
        # The reader leaves ownership with the original handle.
        with pytest.raises(SessionAlreadyActiveError):
            store.open(session.id)
    assert store.load_transcript("missing") == []
    assert not store.path_for("missing").exists()


def test_same_live_session_cannot_be_opened_twice(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create(session_id="live")
    try:
        with pytest.raises(SessionAlreadyActiveError):
            store.open("live")
    finally:
        session.close()

    resumed = store.open("live")
    resumed.close()


def test_user_messages_are_thread_safe_when_host_opts_into_cross_thread_access(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create(session_id="threaded", check_same_thread=False)
    barrier = threading.Barrier(3)
    errors: list[Exception] = []

    def write(prefix: str) -> None:
        try:
            barrier.wait(timeout=5)
            for index in range(25):
                session.record_user_message(f"{prefix}-{index}")
        except Exception as error:
            errors.append(error)

    writers = [threading.Thread(target=write, args=(prefix,)) for prefix in ("a", "b")]
    for writer in writers:
        writer.start()
    barrier.wait(timeout=5)
    for writer in writers:
        writer.join(timeout=10)

    try:
        assert errors == []
        assert all(not writer.is_alive() for writer in writers)
        assert len(session.turns()) == 50
        assert session.info.turn_count == 50
    finally:
        session.close()


def test_open_missing_or_invalid_session(tmp_path):
    store = SessionStore(tmp_path)
    with pytest.raises(SessionNotFoundError):
        store.open("missing")
    for unsafe in ("", ".", "..", "../escape", "nested/id", "nested\\id", "bad\x00id"):
        with pytest.raises(InvalidSessionIdError):
            store.path_for(unsafe)


def test_delete_refuses_live_session_then_removes_database_and_sidecars(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create(session_id="delete-me")
    path = session.path
    with pytest.raises(SessionAlreadyActiveError):
        store.delete(session.id)

    session.close()
    path.with_name(f"{path.name}-wal").touch()
    path.with_name(f"{path.name}-shm").touch()
    assert store.delete("delete-me") is True
    assert not path.exists()
    assert not path.with_name(f"{path.name}-wal").exists()
    assert not path.with_name(f"{path.name}-shm").exists()
    assert store.delete("delete-me") is False


def test_delete_missing_session_from_missing_root_returns_false(tmp_path):
    store = SessionStore(tmp_path / "not-created")
    assert store.delete("missing") is False


def test_list_skips_corrupt_and_non_session_databases(tmp_path):
    store = SessionStore(tmp_path)
    valid = store.create(session_id="valid")
    valid.close()
    (tmp_path / "corrupt.db").write_bytes(b"not sqlite")
    connection = sqlite3.connect(tmp_path / "no-start.db")
    connection.execute("CREATE TABLE events (event_type TEXT, data TEXT, insertion_order INTEGER)")
    connection.close()

    assert [info.id for info in store.list()] == ["valid"]


def test_prefix_search_is_literal_and_sorted(tmp_path):
    store = SessionStore(tmp_path)
    for session_id in ("prefix-a", "prefix-b", "other"):
        session = store.create(session_id=session_id)
        session.close()

    assert set(store.find_by_prefix("prefix-")) == {"prefix-a", "prefix-b"}
    assert store.find_by_prefix("*") == []
    assert store.find_by_prefix("../") == []


def test_reads_legacy_tui_session_events(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create(session_id="legacy-placeholder")
    session.close()
    store.delete("legacy-placeholder")

    path = tmp_path / "legacy.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE events (
            tag TEXT PRIMARY KEY,
            event_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            data TEXT NOT NULL,
            insertion_order INTEGER NOT NULL
        );
        """
    )
    timestamp = datetime.now(UTC).isoformat()
    legacy_events = [
        (
            "TUISessionStart",
            {
                "event_type": "TUISessionStart",
                "timestamp": timestamp,
                "model": "legacy/model",
                "agent_cls": "TUIAgent",
                "working_dir": "/legacy",
            },
        ),
        (
            "TUIUserInput",
            {"event_type": "TUIUserInput", "timestamp": timestamp, "text": "old user"},
        ),
        (
            "TUIAgentMessage",
            {
                "event_type": "TUIAgentMessage",
                "timestamp": timestamp,
                "content": "old agent",
            },
        ),
        (
            "TUISessionRename",
            {
                "event_type": "TUISessionRename",
                "timestamp": timestamp,
                "name": "Legacy title",
                "user_named": True,
            },
        ),
    ]
    for order, (event_type, data) in enumerate(legacy_events):
        connection.execute(
            "INSERT INTO events VALUES (?, ?, ?, 'active', ?, ?)",
            (str(order + 1), f"id-{order}", event_type, json.dumps(data), order),
        )
    connection.commit()
    connection.close()

    info = store.get("legacy")
    assert info.origin == "tui"
    assert info.model == "legacy/model"
    assert info.agent == "TUIAgent"
    assert info.working_directory == "/legacy"
    assert info.title == "Legacy title"
    assert info.title_is_user_set is True
    assert info.turn_count == 1
    assert [(turn.role, turn.content) for turn in store.load_turns("legacy")] == [
        ("user", "old user"),
        ("agent", "old agent"),
    ]
