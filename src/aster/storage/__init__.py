# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent persistence — StorageManager protocol and implementations."""

from aster.storage.in_memory import InMemoryStorageManager
from aster.storage.json_snapshot import snapshot_from_json, snapshot_to_json
from aster.storage.manager import StorageManager
from aster.storage.markers import nosnapshot, snapshotable
from aster.storage.persistent_vars import PersistentVars
from aster.storage.serialization import SKIP, deserialize, serialize
from aster.storage.snapshot import AgentSnapshot
from aster.storage.sqlite import (
    SessionAlreadyActiveError,
    SQLiteStorageManager,
    delete_sqlite_database,
    read_lock_owner,
)

__all__ = [
    "AgentSnapshot",
    "InMemoryStorageManager",
    "PersistentVars",
    "SKIP",
    "SQLiteStorageManager",
    "SessionAlreadyActiveError",
    "StorageManager",
    "deserialize",
    "delete_sqlite_database",
    "nosnapshot",
    "read_lock_owner",
    "serialize",
    "snapshotable",
    "snapshot_from_json",
    "snapshot_to_json",
]
