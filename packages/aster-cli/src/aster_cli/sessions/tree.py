# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Message ancestry and navigation across immutable copies of session branches.

Checkpoint forks preserve event identities. Their shared start event identifies
one logical conversation, and copied message IDs identify its shared prefix.
Navigation creates a fresh owned runtime store; no branch is rewritten in place.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from aster.errors.storage import SnapshotNotFoundError
from aster.storage.sqlite import SQLiteStorageManager
from aster_cli.sessions.store import (
    _AGENT_EVENT_TYPES,
    _START_EVENT_TYPES,
    _TURN_EVENT_TYPES,
    _USER_EVENT_TYPES,
    SessionCheckpointUnavailableError,
    SessionStore,
)

logger = logging.getLogger(__name__)

MAX_SESSION_FILES = 4096
MAX_TREE_NODES = 10_000
MAX_EVENT_BYTES = 32 * 1024 * 1024
PREVIEW_CHARS = 500
# These records cannot advance agent state. Unknown legacy tail events are
# conservatively treated as execution, so a progress message is not a boundary.
_PASSIVE_TAIL_TYPES = ("DebugTrace", "SessionTitleUpdated", "TUISessionRename")
_OUTCOME_TYPE = "SessionTurnFinished"


@dataclass(frozen=True, slots=True)
class _Entry:
    id: str
    parent_id: str | None
    role: Literal["user", "assistant"]
    preview: str
    fingerprint: str
    timestamp: float
    user_index: int
    user_id: str | None
    matched: bool


@dataclass(frozen=True, slots=True)
class _Branch:
    session_id: str
    root_id: str
    entries: tuple[_Entry, ...]
    checkpoints: frozenset[str]
    completed: dict[str, tuple[str, int]]
    outcomes: dict[str, str | None]
    current_reply_id: str | None
    selected_prompt: str | None


@dataclass(frozen=True, slots=True)
class _Selection:
    branch: _Branch
    offset: int
    checkpoint_id: str | None
    priority: int


def _valid_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= 256
        and not any(ord(char) < 32 for char in value)
    )


def _event_data(event_id: str, data: object) -> dict[str, object]:
    if not isinstance(data, str) or len(data) > MAX_EVENT_BYTES:
        raise SessionCheckpointUnavailableError("A saved message exceeds the tree size limit.")
    raw = json.loads(data)
    if not isinstance(raw, dict):
        raise SessionCheckpointUnavailableError("A saved message could not be read.")
    if raw.get("id", event_id) != event_id:
        raise SessionCheckpointUnavailableError("A saved message has conflicting identities.")
    return raw


def _outcome(raw: dict[str, object]) -> tuple[str, str | None]:
    user_id = raw.get("user_message_id")
    reply_id = raw.get("completed_message_id")
    if (
        not isinstance(user_id, str)
        or not _valid_id(user_id)
        or "completed_message_id" not in raw
        or (reply_id is not None and (not isinstance(reply_id, str) or not _valid_id(reply_id)))
    ):
        raise SessionCheckpointUnavailableError("A saved turn has an invalid completion boundary.")
    return user_id, reply_id


def _merge_outcome(outcomes: dict[str, str | None], user_id: str, reply_id: str | None) -> None:
    if user_id in outcomes and outcomes[user_id] != reply_id:
        raise SessionCheckpointUnavailableError(
            "Saved branches disagree about a shared turn's completion; their files were preserved."
        )
    outcomes[user_id] = reply_id


def _checkpoint_tail(conn: sqlite3.Connection, raw_refs: object) -> tuple[str, str, str] | None:
    """Read only the saved tail needed to identify an immutable reply boundary."""
    if not isinstance(raw_refs, str) or len(raw_refs) > MAX_EVENT_BYTES:
        raise SessionCheckpointUnavailableError("A saved checkpoint exceeds the tree size limit.")
    refs = json.loads(raw_refs)
    if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs):
        raise SessionCheckpointUnavailableError("A saved checkpoint has invalid event references.")
    for digest in reversed(refs):
        saved = conn.execute(
            "SELECT row_data FROM checkpoint_events WHERE digest = ?", (digest,)
        ).fetchone()
        if saved is None or not isinstance(saved[0], str) or len(saved[0]) > 2 * MAX_EVENT_BYTES:
            raise SessionCheckpointUnavailableError("A saved checkpoint has missing event data.")
        payload = saved[0]
        if hashlib.sha256(payload.encode()).hexdigest() != digest:
            raise SessionCheckpointUnavailableError("A saved checkpoint has damaged event data.")
        event = json.loads(payload)
        if (
            not isinstance(event, list)
            or len(event) != 6
            or any(not isinstance(value, str) for value in event[:5])
            or type(event[5]) is not int
        ):
            raise SessionCheckpointUnavailableError("A saved checkpoint has invalid event data.")
        if event[2] not in _PASSIVE_TAIL_TYPES:
            return event[1], event[2], event[4]
    return None


def _read_branch(
    path: Path,
    *,
    root_id: str | None = None,
    selected: str | None = None,
    query: str | None = None,
) -> _Branch | None:
    matched = False
    try:
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            conn.execute("BEGIN")
            start = conn.execute(
                "SELECT event_id FROM events WHERE event_type IN (?, ?) "
                "ORDER BY insertion_order LIMIT 1",
                tuple(_START_EVENT_TYPES),
            ).fetchone()
            if start is None or not _valid_id(start[0]):
                if root_id is None:
                    raise SessionCheckpointUnavailableError(
                        "This conversation has no saved identity."
                    )
                return None
            if root_id is not None and start[0] != root_id:
                return None
            matched = True
            entries: list[_Entry] = []
            seen: set[str] = set()
            user_index = -1
            user_id = None
            prompt = None
            placeholders = ", ".join("?" for _ in _TURN_EVENT_TYPES)
            for event_id, event_type, data in conn.execute(
                "SELECT event_id, event_type, data FROM events "
                f"WHERE event_type IN ({placeholders}) ORDER BY insertion_order",
                tuple(_TURN_EVENT_TYPES),
            ):
                if len(entries) >= MAX_TREE_NODES:
                    raise SessionCheckpointUnavailableError(
                        f"The conversation tree exceeds {MAX_TREE_NODES} messages; saved branches were preserved."
                    )
                if not _valid_id(event_id) or event_id in seen:
                    raise SessionCheckpointUnavailableError(
                        "A saved branch contains invalid or repeated message identities."
                    )
                raw = _event_data(event_id, data)
                role: Literal["user", "assistant"] = (
                    "user" if event_type in _USER_EVENT_TYPES else "assistant"
                )
                text = raw.get("content", raw.get("text", ""))
                if not isinstance(text, str):
                    raise SessionCheckpointUnavailableError("A saved message has invalid text.")
                if role == "user":
                    user_index += 1
                    user_id = event_id
                    if event_id == selected:
                        prompt = text
                entries.append(
                    _Entry(
                        id=event_id,
                        parent_id=entries[-1].id if entries else None,
                        role=role,
                        preview=text[:PREVIEW_CHARS],
                        fingerprint=hashlib.sha256(text.encode()).hexdigest(),
                        timestamp=SessionStore._timestamp(raw, fallback=float(len(entries))),
                        user_index=user_index,
                        user_id=user_id,
                        matched=query is not None
                        and (query in text.lower() or query in ("you" if role == "user" else role)),
                    )
                )
                seen.add(event_id)
            tables = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
            checkpoints = (
                frozenset(row[0] for row in conn.execute("SELECT checkpoint_id FROM checkpoints"))
                if "checkpoints" in tables
                else frozenset()
            )
            if any(not _valid_id(key) for key in checkpoints):
                raise SessionCheckpointUnavailableError(
                    "A saved branch has invalid checkpoint names."
                )
            outcomes: dict[str, str | None] = {}
            for event_id, data in conn.execute(
                "SELECT event_id, data FROM events WHERE event_type = ? ORDER BY insertion_order",
                (_OUTCOME_TYPE,),
            ):
                _merge_outcome(outcomes, *_outcome(_event_data(event_id, data)))
            by_id = {entry.id: entry for entry in entries}
            completed: dict[str, tuple[str, int]] = {}
            if "checkpoints" in tables:
                for key, raw_refs in conn.execute(
                    "SELECT checkpoint_id, event_refs FROM checkpoints"
                ):
                    if not key.startswith("after:"):
                        continue
                    tail = _checkpoint_tail(conn, raw_refs)
                    if tail is None:
                        continue
                    event_id, event_type, data = tail
                    if event_type == _OUTCOME_TYPE:
                        bound_user, reply_id = _outcome(_event_data(event_id, data))
                        _merge_outcome(outcomes, bound_user, reply_id)
                        priority = 0
                    elif event_type in _AGENT_EVENT_TYPES:
                        reply_id, priority = event_id, 1
                    else:
                        continue
                    reply = by_id.get(reply_id) if reply_id is not None else None
                    if (
                        reply is not None
                        and reply.role == "assistant"
                        and key == f"after:{reply.user_index}"
                    ):
                        completed[key] = (reply.id, priority)
            for bound_user, reply_id in outcomes.items():
                user = by_id.get(bound_user)
                reply = by_id.get(reply_id) if reply_id is not None else None
                if (
                    user is None
                    or user.role != "user"
                    or (
                        reply_id is not None
                        and (
                            reply is None
                            or reply.role != "assistant"
                            or reply.user_id != bound_user
                        )
                    )
                ):
                    raise SessionCheckpointUnavailableError(
                        "A saved turn has an invalid completion boundary."
                    )
            has_current = (
                "snapshots" in tables
                and conn.execute(
                    "SELECT 1 FROM snapshots WHERE snapshot_id = 'session-current'"
                ).fetchone()
                is not None
            )
            tail = conn.execute(
                "SELECT event_id, event_type FROM events "
                "WHERE event_type NOT IN (?, ?, ?) ORDER BY insertion_order DESC LIMIT 1",
                _PASSIVE_TAIL_TYPES,
            ).fetchone()
            current_reply_id = (
                tail[0]
                if has_current and tail is not None and tail[1] in _AGENT_EVENT_TYPES
                else None
            )
            return _Branch(
                path.stem,
                start[0],
                tuple(entries),
                checkpoints,
                completed,
                outcomes,
                current_reply_id,
                prompt,
            )
    except (OSError, sqlite3.Error, ValueError) as exc:
        if isinstance(exc, SessionCheckpointUnavailableError):
            raise
        if root_id is None or matched:
            raise SessionCheckpointUnavailableError(
                "A saved conversation branch could not be read; its files were preserved."
            ) from exc
        # A damaged unrelated database must not prevent opening this conversation.
        logger.debug("Could not inspect session-tree candidate %s", path, exc_info=True)
        return None


def _selection(branch: _Branch, offset: int) -> _Selection | None:
    entry = branch.entries[offset]
    if entry.role == "user":
        key = f"user:{entry.user_index}"
        return _Selection(branch, offset, key, 0) if key in branch.checkpoints else None
    if entry.user_id is None:
        return None
    after = f"after:{entry.user_index}"
    boundary = branch.completed.get(after)
    if boundary is not None and boundary[0] == entry.id:
        return _Selection(branch, offset, after, boundary[1])
    if after in branch.checkpoints or entry.user_id in branch.outcomes:
        return None
    # A legacy pre-user checkpoint is not an exact post-reply state: a host may
    # change vars or context between turns. Only a completed leaf can fall back
    # to the older current-snapshot format.
    if (
        offset == len(branch.entries) - 1
        and branch.current_reply_id == entry.id
        and f"user:{entry.user_index + 1}" not in branch.checkpoints
    ):
        return _Selection(branch, offset, None, 2)
    return None


def _collect(
    store: SessionStore,
    current_id: str,
    *,
    selected: str | None = None,
    query: str | None = None,
) -> tuple[_Branch, dict[str, _Entry], set[str], _Selection | None]:
    root = store.root.resolve()
    current = _read_branch(store.path_for(current_id), selected=selected, query=query)
    if current is None:
        raise SessionCheckpointUnavailableError("The current conversation is unavailable.")
    paths: list[Path] = []
    for path in store.root.glob("*.db"):
        if path.stem.endswith("-memory"):
            continue
        if len(paths) >= MAX_SESSION_FILES:
            raise SessionCheckpointUnavailableError(
                f"This workspace exceeds {MAX_SESSION_FILES} saved sessions; no tree entries were removed."
            )
        paths.append(path)

    def branches() -> Iterator[_Branch]:
        yield current
        for path in sorted(paths):
            if path.stem == current_id or path.is_symlink() or path.resolve().parent != root:
                continue
            branch = _read_branch(path, root_id=current.root_id, selected=selected, query=query)
            if branch is not None:
                yield branch

    entries: dict[str, _Entry] = {}
    navigable: set[str] = set()
    exact: set[str] = set()
    exact_turns: set[str] = set()
    priorities: dict[str, int] = {}
    outcomes: dict[str, str | None] = {}
    best: _Selection | None = None
    best_priority: tuple[int, bool, str] | None = None
    # Keep only the selected node's best source branch. Holding every copy of a
    # shared prefix would multiply memory by the number of saved branch files.
    for branch in branches():
        for user_id, reply_id in branch.outcomes.items():
            _merge_outcome(outcomes, user_id, reply_id)
        for offset, entry in enumerate(branch.entries):
            existing = entries.get(entry.id)
            if existing is not None and (
                existing.parent_id != entry.parent_id
                or existing.role != entry.role
                or existing.fingerprint != entry.fingerprint
            ):
                raise SessionCheckpointUnavailableError(
                    "Saved branches disagree about a shared message; their files were preserved."
                )
            entries.setdefault(entry.id, entry)
            if len(entries) > MAX_TREE_NODES:
                raise SessionCheckpointUnavailableError(
                    f"The conversation tree exceeds {MAX_TREE_NODES} messages; saved branches were preserved."
                )
            if choice := _selection(branch, offset):
                navigable.add(entry.id)
                priorities[entry.id] = min(
                    priorities.get(entry.id, choice.priority), choice.priority
                )
                if choice.checkpoint_id is not None:
                    exact.add(entry.id)
                    if entry.role == "assistant" and entry.user_id is not None:
                        exact_turns.add(entry.user_id)
                if entry.id == selected:
                    priority = (
                        choice.priority,
                        choice.branch.session_id != current_id,
                        choice.branch.session_id,
                    )
                    if best_priority is None or priority < best_priority:
                        best, best_priority = choice, priority
    # Outcomes apply to the logical turn across every saved copy. A legacy
    # sibling must not resurrect a progress or background message, and modern
    # completed turns require the binding inside their own saved checkpoint.
    for node_id in tuple(navigable):
        entry = entries[node_id]
        if priorities[node_id] == 2 and entry.user_id in exact_turns:
            navigable.remove(node_id)
            continue
        if entry.role == "assistant" and entry.user_id is not None and entry.user_id in outcomes:
            if outcomes[entry.user_id] != entry.id or priorities[node_id] != 0:
                navigable.remove(node_id)
    # Older rewind copies did not retain a before-user marker. Do not treat a
    # truncated copy's current snapshot as an exact reply when another saved
    # branch proves that this message has successors.
    parents = {entry.parent_id for entry in entries.values() if entry.parent_id is not None}
    navigable.difference_update(parents - exact)
    if selected not in navigable:
        best = None
    return current, entries, navigable, best


def session_tree(
    store: SessionStore, current_id: str, *, query: str | None = None
) -> dict[str, object]:
    if query is not None and (not isinstance(query, str) or len(query) > 4096):
        raise SessionCheckpointUnavailableError("Search text must be at most 4096 characters.")
    current, entries, navigable, _ = _collect(
        store, current_id, query=query.strip().lower() if query is not None else None
    )
    active = {entry.id for entry in current.entries}
    children: dict[str | None, list[_Entry]] = {}
    for entry in entries.values():
        children.setdefault(entry.parent_id, []).append(entry)
    for values in children.values():
        values.sort(key=lambda entry: (entry.timestamp, entry.id))
    result: list[dict[str, object]] = []
    matches: list[str] = []
    pending = list(reversed(children.get(None, [])))
    seen: set[str] = set()
    while pending:
        entry = pending.pop()
        if entry.id in seen:
            raise SessionCheckpointUnavailableError("The saved conversation tree contains a cycle.")
        seen.add(entry.id)
        if entry.matched:
            matches.append(entry.id)
        result.append(
            {
                "id": entry.id,
                "parentId": entry.parent_id,
                "role": entry.role,
                "preview": entry.preview,
                "navigable": entry.id in navigable,
                "active": entry.id in active,
            }
        )
        pending.extend(reversed(children.get(entry.id, [])))
    if len(seen) != len(entries):
        raise SessionCheckpointUnavailableError("The saved conversation tree has missing ancestry.")
    tree: dict[str, object] = {
        "nodes": result,
        "currentNodeId": current.entries[-1].id if current.entries else None,
    }
    if query is not None:
        tree.update(query=query, matchedNodeIds=matches)
    return tree


def navigate_tree(
    store: SessionStore,
    current_id: str,
    node_id: str,
    expected_current_node_id: str | None,
) -> tuple[str, str | None]:
    if not _valid_id(node_id) or (
        expected_current_node_id is not None and not _valid_id(expected_current_node_id)
    ):
        raise SessionCheckpointUnavailableError("Choose a valid saved message from the tree.")
    current, _, _, choice = _collect(store, current_id, selected=node_id)
    active_node = current.entries[-1].id if current.entries else None
    if active_node != expected_current_node_id:
        raise SessionCheckpointUnavailableError("The conversation changed; open the tree again.")
    if choice is None:
        raise SessionCheckpointUnavailableError(
            "This message has no complete saved conversation state."
        )
    entry = choice.branch.entries[choice.offset]
    end = choice.offset + (entry.role == "assistant")
    prefix = choice.branch.entries[:end]
    cutoff = entry.user_index + (entry.role == "assistant")
    retain = []
    for key in sorted(choice.branch.checkpoints):
        kind, separator, index = key.partition(":")
        if separator and kind in {"user", "after"} and index.isdecimal() and int(index) < cutoff:
            retain.append(key)
    if entry.role == "user":
        # Preserve the selected before-user boundary as well. Its index equals
        # the new branch's user count, distinguishing its current snapshot from
        # a completed legacy reply until that next input is actually accepted.
        retain.append(f"user:{entry.user_index}")
    new_id = str(uuid.uuid4())
    new_path = store.path_for(new_id)
    # Reserve the new name before owning or cleaning it up. Even an unlikely ID
    # collision must never open or delete somebody else's saved branch.
    try:
        new_path.touch(exist_ok=False)
    except OSError as exc:
        raise SessionCheckpointUnavailableError(
            "A new conversation branch could not be created."
        ) from exc
    storage = None
    try:
        storage = SQLiteStorageManager(new_path, check_same_thread=False, must_exist=True)
        SQLiteStorageManager.fork_checkpoint_from_path(
            store.path_for(choice.branch.session_id), choice.checkpoint_id, storage, retain=retain
        )
        copied = _read_branch(store.path_for(new_id))
        if (
            copied is None
            or copied.root_id != current.root_id
            or [(item.id, item.parent_id, item.role, item.fingerprint) for item in copied.entries]
            != [(item.id, item.parent_id, item.role, item.fingerprint) for item in prefix]
        ):
            raise SessionCheckpointUnavailableError(
                "The saved checkpoint does not match this message; original branches were preserved."
            )
        # Even a branch before the first message is durable, as for /rewind.
        for event in storage.event_backend.all_events():
            if event.event_type in _START_EVENT_TYPES:
                tag = storage.event_backend.find_tag(event)
                if tag is not None:
                    storage.event_backend.update(tag, provisional=False)
                break
    except BaseException as exc:
        if storage is not None:
            storage.close()
        store.delete(new_id)
        if isinstance(
            exc, (OSError, sqlite3.Error, ValueError, SnapshotNotFoundError)
        ) and not isinstance(exc, SessionCheckpointUnavailableError):
            raise SessionCheckpointUnavailableError(
                "The saved conversation state could not be copied; original branches were preserved."
            ) from exc
        raise
    storage.close()
    return new_id, choice.branch.selected_prompt if entry.role == "user" else None
