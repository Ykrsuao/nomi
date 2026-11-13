# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only checkpoint copies preserve source ownership and fail atomically."""

import json
import sqlite3

import pytest

from aster import Agent
from aster.context_blocks import UserEvent
from aster.errors.storage import SnapshotNotFoundError
from aster.storage import SessionAlreadyActiveError, SQLiteStorageManager
from aster.unifiedllm import FakeLLMClient


class ForkAgent(Agent, llm=FakeLLMClient()):
    note: str


def _event(storage: SQLiteStorageManager, tag: str) -> UserEvent:
    event = storage.event_backend.get(tag)
    assert isinstance(event, UserEvent)
    return event


def _contents(storage: SQLiteStorageManager) -> list[str]:
    result = []
    for event in storage.event_backend.all_events():
        assert isinstance(event, UserEvent)
        result.append(event.content)
    return result


@pytest.mark.parametrize("checkpoint_id", ["saved", None])
def test_readonly_fork_while_source_is_owned_preserves_state_and_event_counters(
    tmp_path, checkpoint_id
):
    source_path = tmp_path / "source.db"
    with (
        SQLiteStorageManager(source_path) as source,
        SQLiteStorageManager(tmp_path / "target.db") as target,
    ):
        agent = ForkAgent(storage=source)
        first = agent.event_manager.add(UserEvent(content="original"))
        agent.note = "saved state"
        agent.context["saved"] = "saved context"
        source.save_checkpoint("saved", agent)
        source.save_current_snapshot(agent)
        original = _event(source, first).model_dump_json()

        SQLiteStorageManager.fork_checkpoint_from_path(
            source_path, checkpoint_id, target, retain=["saved"]
        )
        with pytest.raises(SessionAlreadyActiveError):
            SQLiteStorageManager(source_path)

        restored = ForkAgent(storage=target)
        assert target.restore_current_snapshot(restored)
        assert restored.note == "saved state"
        assert restored.context["saved"] == "saved context"
        assert target.event_backend.active_tags() == [first]
        assert _event(target, first).model_dump_json() == original
        assert target.checkpoint_ids() == ["saved"]
        next_tag = restored.event_manager.add(UserEvent(content="branch only"))
        assert next_tag != first
        assert len(source.event_backend) == 1
        assert _event(source, first).model_dump_json() == original


def test_readonly_fork_holds_one_source_transaction_during_concurrent_append(tmp_path, monkeypatch):
    source_path = tmp_path / "source.db"
    with (
        SQLiteStorageManager(source_path) as source,
        SQLiteStorageManager(tmp_path / "target.db") as target,
    ):
        agent = ForkAgent(storage=source)
        agent.event_manager.add(UserEvent(content="before"))
        agent.note = "before"
        source.save_current_snapshot(agent)
        real_connect = sqlite3.connect
        appended = False

        class AppendingConnection:
            def __init__(self, *args, **kwargs):
                self.connection = real_connect(*args, **kwargs)

            def execute(self, query, parameters=()):
                nonlocal appended
                cursor = self.connection.execute(query, parameters)
                if query == "SELECT data FROM snapshots WHERE snapshot_id = 'session-current'":
                    assert self.connection.in_transaction
                    agent.event_manager.add(UserEvent(content="after"))
                    agent.note = "after"
                    source.save_current_snapshot(agent)
                    appended = True
                return cursor

            def close(self):
                self.connection.close()

        monkeypatch.setattr(sqlite3, "connect", AppendingConnection)
        SQLiteStorageManager.fork_checkpoint_from_path(source_path, None, target, retain=[])
        assert appended
        restored = ForkAgent(storage=target)
        assert target.restore_current_snapshot(restored)
        assert restored.note == "before"
        assert _contents(target) == ["before"]
        assert _contents(source) == ["before", "after"]


@pytest.mark.parametrize("checkpoint_id", ["missing", None])
def test_readonly_fork_missing_snapshot_leaves_empty_target(tmp_path, checkpoint_id):
    source_path = tmp_path / "source.db"
    with (
        SQLiteStorageManager(source_path),
        SQLiteStorageManager(tmp_path / "target.db") as target,
    ):
        with pytest.raises(SnapshotNotFoundError):
            SQLiteStorageManager.fork_checkpoint_from_path(
                source_path, checkpoint_id, target, retain=[]
            )
        assert len(target.event_backend) == 0
        assert target.get_latest_snapshot_id() is None


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("event_refs", "{}"),
        ("event_refs", '[{"bad": "reference"}]'),
        ("active_tags", '["missing"]'),
        ("active_tags", "{}"),
        ("snapshot", '{"context": "invalid"}'),
    ],
)
@pytest.mark.parametrize("selected", [True, False])
def test_fork_rejects_corrupt_selected_and_retained_checkpoints(tmp_path, column, value, selected):
    source_path = tmp_path / "source.db"
    with (
        SQLiteStorageManager(source_path) as source,
        SQLiteStorageManager(tmp_path / "target.db") as target,
    ):
        agent = ForkAgent(storage=source)
        agent.event_manager.add(UserEvent(content="original"))
        source.save_checkpoint("earlier", agent)
        source.save_checkpoint("selected", agent)
        damaged = "selected" if selected else "earlier"
        with sqlite3.connect(source_path) as connection:
            connection.execute(
                f"UPDATE checkpoints SET {column} = ? WHERE checkpoint_id = ?", (value, damaged)
            )
            if column == "event_refs":
                connection.execute(
                    "UPDATE checkpoints SET active_tags = '[]' WHERE checkpoint_id = ?", (damaged,)
                )
        with pytest.raises(ValueError):
            source.fork_checkpoint("selected", target, retain=["earlier"])
        assert len(target.event_backend) == 0
        assert target.get_latest_snapshot_id() is None
        assert _contents(source) == ["original"]


def test_fork_rejects_changed_event_payload_before_writing_target(tmp_path):
    source_path = tmp_path / "source.db"
    with (
        SQLiteStorageManager(source_path) as source,
        SQLiteStorageManager(tmp_path / "target.db") as target,
    ):
        agent = ForkAgent(storage=source)
        agent.event_manager.add(UserEvent(content="original"))
        source.save_checkpoint("saved", agent)
        with sqlite3.connect(source_path) as connection:
            digest, payload = connection.execute(
                "SELECT digest, row_data FROM checkpoint_events LIMIT 1"
            ).fetchone()
            row = json.loads(payload)
            row[4] = row[4].replace("original", "changed")
            connection.execute(
                "UPDATE checkpoint_events SET row_data = ? WHERE digest = ?",
                (json.dumps(row), digest),
            )
        with pytest.raises(ValueError, match="damaged event data"):
            SQLiteStorageManager.fork_checkpoint_from_path(source_path, "saved", target, retain=[])
        assert len(target.event_backend) == 0
        assert target.checkpoint_ids() == []


@pytest.mark.parametrize("existing", ["snapshot", "checkpoint"])
def test_fork_refuses_eventless_target_with_existing_saved_state(tmp_path, existing):
    with (
        SQLiteStorageManager(tmp_path / "source.db") as source,
        SQLiteStorageManager(tmp_path / "target.db") as target,
    ):
        source.save_checkpoint("selected", ForkAgent(storage=source))
        previous = ForkAgent(storage=target)
        previous.note = "keep this"
        if existing == "snapshot":
            target.save_snapshot(previous)
        else:
            target.save_checkpoint("previous", previous)
        before_snapshot = target.get_latest_snapshot_id()
        before_checkpoints = target.checkpoint_ids()
        with pytest.raises(ValueError, match="must be empty"):
            source.fork_checkpoint("selected", target, retain=[])
        assert target.get_latest_snapshot_id() == before_snapshot
        assert target.checkpoint_ids() == before_checkpoints
        assert len(target.event_backend) == 0
