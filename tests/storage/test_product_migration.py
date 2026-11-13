# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Historical typed snapshots restore through Aster's classes and allowlist."""

import sys
from types import ModuleType

import pytest
from aster_cli.sessions import SessionTurn

from aster.errors.storage import DeserializationError
from aster.storage.serialization import deserialize, serialize
from aster.storage.snapshot_vars import SnapshotVars
from aster.tools.todo import TodoManager


def test_restoring_previous_core_class_ignores_installed_old_package(monkeypatch):
    previous = ModuleType("nooa.storage.snapshot_vars")

    class PreviousSnapshotVars:
        def __init__(self):
            pytest.fail("The previous installation must not construct saved state")

    previous.SnapshotVars = PreviousSnapshotVars
    monkeypatch.setitem(sys.modules, previous.__name__, previous)
    fqn = "nooa.storage.snapshot_vars.SnapshotVars"

    restored = deserialize(
        {"__type__": "dict_class", "__class__": fqn, "data": {"_data": {"answer": 42}}},
        {fqn},
    )

    assert type(restored) is SnapshotVars
    assert restored["answer"] == 42


def test_previous_cli_snapshot_restores_message_without_rewriting_text():
    fqn = "nooa_cli.sessions.store.SessionTurn"
    text = "Keep this original message: import nooa"
    restored = deserialize(
        {
            "__type__": "dataclass",
            "__class__": fqn,
            "data": {"role": "user", "content": text, "timestamp": 123.0},
        },
        {fqn},
    )

    assert type(restored) is SessionTurn
    assert restored.content == text
    blob, allowlist = serialize(restored)
    assert blob["__class__"] == "aster_cli.sessions.store.SessionTurn"
    assert allowlist == {"aster_cli.sessions.store.SessionTurn"}


@pytest.mark.parametrize("allowlist", [set(), {"aster.storage.snapshot_vars.SnapshotVars"}])
def test_previous_class_still_requires_its_original_allowlist_entry(allowlist):
    with pytest.raises(DeserializationError, match="not in the allowlist"):
        deserialize(
            {
                "__type__": "dict_class",
                "__class__": "nooa.storage.snapshot_vars.SnapshotVars",
                "data": {"_data": {}},
            },
            allowlist,
        )


def test_previous_todos_restore_and_serialize_with_current_namespace():
    blob, _ = serialize(TodoManager())
    previous = "nooa.tools.todo.TodoManager"
    blob["__class__"] = previous

    restored = deserialize(blob, {previous})

    assert type(restored) is TodoManager
    current, allowlist = serialize(restored)
    assert current["__class__"] == "aster.tools.todo.TodoManager"
    assert allowlist == {"aster.tools.todo.TodoManager"}
