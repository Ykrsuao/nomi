# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable checkpoints preserve event versions as well as their active order."""

import sqlite3

import pytest

from aster import Agent
from aster.context_blocks import EventStatus, UserEvent
from aster.errors.storage import SerializationError, SnapshotNotFoundError
from aster.storage import SQLiteStorageManager
from aster.unifiedllm import FakeLLMClient


class CheckpointAgent(Agent, llm=FakeLLMClient()):
    pass


def test_checkpoint_survives_event_mutation_removal_and_reopen(tmp_path):
    source_path = tmp_path / "source.db"
    with SQLiteStorageManager(source_path) as source:
        agent = CheckpointAgent(storage=source)
        first = agent.event_manager.add(UserEvent(content="before checkpoint"))
        archived = agent.event_manager.add(UserEvent(content="archived before checkpoint"))
        source.event_backend.remove_active_tag(archived)
        original = source.event_backend.get(first).model_dump_json()
        agent.answer = "old state"
        agent.context["note"] = "earlier context"
        source.save_checkpoint("before-second", agent)

        source.event_backend.update(first, content="changed after checkpoint")
        source.event_backend.set_status(first, EventStatus.ARCHIVED)
        source.event_backend.remove(first)
        agent.event_manager.add(UserEvent(content="future knowledge"))
        agent.answer = "new state"
        agent.future_attribute = "later only"
        agent.context["future"] = "later context"
        source.save_checkpoint("later", agent)

    with (
        SQLiteStorageManager(source_path) as source,
        SQLiteStorageManager(tmp_path / "branch.db") as branch,
    ):
        source.fork_checkpoint("before-second", branch, retain=["before-second"])
        restored = CheckpointAgent(storage=branch)
        assert branch.restore_current_snapshot(restored)
        assert restored.answer == "old state"
        assert not hasattr(restored, "future_attribute")
        assert "future" not in restored.context
        assert branch.event_backend.active_tags() == [first]
        assert branch.event_backend.get(first).model_dump_json() == original
        assert branch.event_backend.get(archived) is not None
        assert len(branch.event_backend) == 2
        assert branch.checkpoint_ids() == ["before-second"]
        new_tag = restored.event_manager.add(UserEvent(content="new branch input"))
        assert new_tag not in {first, archived}
        assert branch.event_backend.get(first).model_dump_json() == original
        assert source.event_backend.get(first) is None


def test_unchanged_event_payloads_are_shared_by_checkpoints(tmp_path):
    path = tmp_path / "shared.db"
    with SQLiteStorageManager(path) as storage:
        agent = CheckpointAgent(storage=storage)
        agent.event_manager.add(UserEvent(content="a large unchanged event" * 100))
        storage.save_checkpoint("one", agent)
        storage.save_checkpoint("two", agent)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM checkpoint_events").fetchone() == (1,)
        assert connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone() == (2,)


def test_checkpoint_requires_serializable_state_and_does_not_publish_partial_state(tmp_path):
    class Unsupported:
        pass

    with SQLiteStorageManager(tmp_path / "source.db") as source:
        agent = CheckpointAgent(storage=source)
        agent.unsupported = Unsupported()
        with pytest.raises(SerializationError, match="unsupported"):
            source.save_checkpoint("bad", agent)
        assert source.checkpoint_ids() == []


def test_missing_checkpoint_and_nonempty_target_leave_target_intact(tmp_path):
    with (
        SQLiteStorageManager(tmp_path / "source.db") as source,
        SQLiteStorageManager(tmp_path / "target.db") as target,
    ):
        agent = CheckpointAgent(storage=source)
        source.save_checkpoint("available", agent)
        target_agent = CheckpointAgent(storage=target)
        existing = target_agent.event_manager.add(UserEvent(content="keep me"))
        with pytest.raises(SnapshotNotFoundError):
            source.fork_checkpoint("missing", target, retain=[])
        with pytest.raises(ValueError, match="empty"):
            source.fork_checkpoint("available", target, retain=[])
        assert target.event_backend.get(existing).content == "keep me"


def test_current_snapshot_is_distinct_from_preturn_checkpoints(tmp_path):
    with SQLiteStorageManager(tmp_path / "source.db") as source:
        agent = CheckpointAgent(storage=source)
        agent.value = "before"
        source.save_checkpoint("before-turn", agent)
        agent.value = "after"
        source.save_current_snapshot(agent)
        restored = CheckpointAgent(storage=source)
        assert source.restore_current_snapshot(restored)
        assert restored.value == "after"
        source.invalidate_current_snapshot()
        fresh = CheckpointAgent(storage=source)
        assert not source.restore_current_snapshot(fresh)
        assert not hasattr(fresh, "value")
