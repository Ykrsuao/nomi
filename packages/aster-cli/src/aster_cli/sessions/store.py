# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SQLite-backed coding-agent sessions discoverable by interactive hosts."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Literal

from aster.paths import get_project_dir
from aster.runtime.event_manager import EventManager
from aster.storage.sqlite import SQLiteStorageManager, delete_sqlite_database
from aster_cli.sessions.events import (
    SESSION_EVENT_TYPES,
    SessionStarted,
    SessionSteering,
    SessionTitleUpdated,
    SessionTurnFinished,
    SessionUserMessage,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aster.agent import Agent

_START_EVENT_TYPES = frozenset(("SessionStarted", "TUISessionStart"))
_TITLE_EVENT_TYPES = frozenset(("SessionTitleUpdated", "TUISessionRename"))
_USER_EVENT_TYPES = frozenset(("SessionUserMessage", "TUIUserInput"))
_AGENT_EVENT_TYPES = frozenset(("AgentMessage", "TUIAgentMessage"))
_TURN_EVENT_TYPES = _USER_EVENT_TYPES | _AGENT_EVENT_TYPES
_TRANSCRIPT_EVENT_TYPES = _TURN_EVENT_TYPES | frozenset(
    (
        "SessionActivity",
        "SessionSteering",
        "ToolCallEvent",
        "PythonOutput",
        "FileEdit",
        "TerminalCommandStarted",
        "TerminalCommandOutput",
        "TerminalCommandFinished",
    )
)


class InvalidSessionIdError(ValueError):
    """Raised before an unsafe or empty session ID can become a file path."""


class SessionNotFoundError(FileNotFoundError):
    """Raised when a durable session does not exist or lacks start metadata."""


class SessionCheckpointUnavailableError(ValueError):
    """The selected user message has no complete saved conversation state."""


@dataclass(frozen=True, slots=True)
class SessionInfo:
    """Host-neutral metadata used by session lists and resume flows."""

    id: str
    model: str
    agent: str
    started_at: float
    last_active: float
    turn_count: int = 0
    working_directory: str = ""
    title: str | None = None
    title_is_user_set: bool = False
    origin: str = ""
    # Background preparation becomes durable on the first accepted user message.
    provisional: bool = False


@dataclass(frozen=True, slots=True)
class SessionTurn:
    role: Literal["user", "agent"]
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class SessionTranscriptEvent:
    """A persisted public transcript record, in original insertion order."""

    event_type: str
    data: dict[str, object]


class SessionHandle:
    """An agent-runtime-owned session database and metadata writer.

    Exactly one live agent runtime owns this handle. Interactive hosts attach
    to that runtime through their transport; they do not open the database for
    writing alongside it.
    """

    def __init__(
        self,
        store: SessionStore,
        storage: SQLiteStorageManager,
        info: SessionInfo,
    ) -> None:
        self._store = store
        self._storage = storage
        self._events = EventManager(backend=storage.event_backend)
        for event_type in SESSION_EVENT_TYPES:
            self._events.register_event_type(event_type)
        self._info = info
        self._metadata_lock = RLock()
        self._closed = False

    @property
    def id(self) -> str:
        return self._info.id

    @property
    def info(self) -> SessionInfo:
        with self._metadata_lock:
            return self._info

    @property
    def storage(self) -> SQLiteStorageManager:
        """Storage to pass to the agent constructed by the owning runtime."""
        return self._storage

    @property
    def events(self) -> EventManager:
        """Event manager for runtime-owned session metadata."""
        return self._events

    @property
    def path(self) -> Path:
        return self._store.path_for(self.id)

    def set_title(self, title: str, *, user_set: bool = False) -> None:
        """Persist a title and update this handle's current metadata."""
        self._ensure_open()
        event = SessionTitleUpdated(title=title, user_set=user_set)
        self._events.add(event)
        with self._metadata_lock:
            self._info = replace(
                self._info,
                title=title,
                title_is_user_set=self._info.title_is_user_set or user_set,
                last_active=event.timestamp.timestamp(),
            )

    def record_user_message(
        self, content: str, *, turn_id: str | None = None
    ) -> tuple[str, SessionUserMessage]:
        """Persist raw user text before dispatching its agent turn."""
        with self._metadata_lock:
            self._ensure_open()
            event = SessionUserMessage(content=content, turn_id=turn_id)
            tag = self._events.add(event)
            self._info = replace(
                self._info,
                turn_count=self._info.turn_count + 1,
                last_active=event.timestamp.timestamp(),
            )
        return tag, event

    def record_steering(self, event: SessionSteering) -> None:
        """Record steering without creating a user turn or a rewind checkpoint."""
        with self._metadata_lock:
            self._ensure_open()
            self._events.add(event)
            self._info = replace(self._info, last_active=event.timestamp.timestamp())

    def turns(self) -> list[SessionTurn]:
        return self._store.load_turns(self.id)

    def checkpoint(self, agent: Agent) -> None:
        """Capture state immediately before recording the next user message."""
        with self._metadata_lock:
            self._ensure_open()
            key = f"user:{self._info.turn_count}"
            self._storage.discard_checkpoint(key)
            self._storage.save_checkpoint(key, agent)

    def checkpoint_completed(
        self, agent: Agent, *, completed_message_id: str | None, turn_id: str | None = None
    ) -> None:
        """Capture complete post-turn state for navigation to the final reply."""
        with self._metadata_lock:
            self._ensure_open()
            outcome = self._record_turn_finished(
                completed_message_id=completed_message_id, turn_id=turn_id
            )
            if outcome is None:
                return
            key = f"after:{self._info.turn_count - 1}"
            self._storage.discard_checkpoint(key)
            if outcome.completed_message_id is not None:
                # Include the binding in the immutable checkpoint itself so it
                # survives navigation, rewind, and loading a retained boundary.
                self._storage.save_checkpoint(key, agent)

    def checkpoint_interrupted(self, *, turn_id: str | None = None) -> None:
        """Keep partial messages visible without treating them as completed replies."""
        with self._metadata_lock:
            self._ensure_open()
            if self._record_turn_finished(completed_message_id=None, turn_id=turn_id) is not None:
                self._storage.discard_checkpoint(f"after:{self._info.turn_count - 1}")

    def _record_turn_finished(
        self, *, completed_message_id: str | None, turn_id: str | None
    ) -> SessionTurnFinished | None:
        user = None
        reply_id = None
        for event in self._storage.event_backend.all_events():
            if event.event_type in _USER_EVENT_TYPES:
                user, reply_id = event, None
            elif event.event_type in _AGENT_EVENT_TYPES and user is not None:
                reply_id = event.id
        if user is None or (turn_id is not None and getattr(user, "turn_id", None) != turn_id):
            # Validation may have failed before this prompt was recorded. It
            # must not change the outcome of the preceding accepted turn.
            return None
        outcome = SessionTurnFinished(
            user_message_id=user.id,
            # A checkpoint includes all public messages preceding its state.
            # If another message was appended synchronously during publication,
            # it cannot be restored at the earlier foreground message's prefix.
            completed_message_id=completed_message_id if reply_id == completed_message_id else None,
        )
        self._events.add(outcome)
        return outcome

    def session_tree(self, query: str | None = None) -> dict[str, object]:
        """List message ancestry across this conversation's preserved branches."""
        from aster_cli.sessions.tree import session_tree

        with self._metadata_lock:
            self._ensure_open()
            return session_tree(self._store, self.id, query=query)

    def navigate_tree(
        self, node_id: str, expected_current_node_id: str | None
    ) -> tuple[str, str | None]:
        """Restore a tree point into a fresh branch.

        User messages return their exact text for editing at their parent. Other
        eligible messages resume after the selected reply. No prompt is sent.
        """
        from aster_cli.sessions.tree import navigate_tree

        with self._metadata_lock:
            self._ensure_open()
            return navigate_tree(self._store, self.id, node_id, expected_current_node_id)

    def checkpoints(self) -> list[tuple[int, str]]:
        """Eligible zero-based user-message indices and their original text."""
        self._ensure_open()
        available = set(self._storage.checkpoint_ids())
        users = [turn.content for turn in self.turns() if turn.role == "user"]
        return [(index, text) for index, text in enumerate(users) if f"user:{index}" in available]

    def rewind(self, user_message_index: int) -> tuple[str, str]:
        """Fork before one user message, preserving the source and working files."""
        with self._metadata_lock:
            self._ensure_open()
            choices = dict(self.checkpoints())
            if type(user_message_index) is not int or user_message_index not in choices:
                raise SessionCheckpointUnavailableError(
                    "This message has no saved rewind checkpoint. Choose a newer message."
                )
            session_id = str(uuid.uuid4())
            path = self._store.path_for(session_id)
            storage = SQLiteStorageManager(path, check_same_thread=False)
            try:
                self._storage.fork_checkpoint(
                    f"user:{user_message_index}",
                    storage,
                    retain=[
                        f"{kind}:{index}"
                        for index in range(user_message_index)
                        for kind in ("user", "after")
                    ]
                    + [f"user:{user_message_index}"],
                )
                # A fork before the first message is still a real conversation,
                # even when its source began as a hidden prepared session.
                for event in storage.event_backend.all_events():
                    if event.event_type in _START_EVENT_TYPES:
                        tag = storage.event_backend.find_tag(event)
                        if tag is not None:
                            storage.event_backend.update(tag, provisional=False)
                        break
            except BaseException:
                storage.close()
                self._store.delete(session_id)
                raise
            storage.close()
            return session_id, choices[user_message_index]

    def transcript(self) -> list[SessionTranscriptEvent]:
        return self._store.load_transcript(self.id)

    def close(self) -> None:
        with self._metadata_lock:
            if self._closed:
                return
            self._storage.close()
            self._closed = True
            unused_preparation = self._info.provisional and self._info.turn_count == 0
        if unused_preparation:
            # Use the existing ownership lock and exact SQLite sidecar cleanup;
            # only this handle's unused background session is eligible.
            self._store.delete(self.id)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"Session {self.id!r} is closed")

    def __enter__(self) -> SessionHandle:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class SessionStore:
    """Repository and factory for project-local durable coding-agent sessions.

    A daemon may use read-only operations such as :meth:`list` and :meth:`get`
    for discovery. Only the process running the agent opens a
    :class:`SessionHandle` for writes.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else get_project_dir("sessions")

    def path_for(self, session_id: str) -> Path:
        session_id = self._validate_id(session_id)
        return self.root / f"{session_id}.db"

    def create(
        self,
        *,
        model: str = "",
        agent: str = "",
        working_directory: str = "",
        origin: str = "",
        provisional: bool = False,
        session_id: str | None = None,
        check_same_thread: bool = True,
    ) -> SessionHandle:
        session_id = self._validate_id(session_id or str(uuid.uuid4()))
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(session_id)
        if path.exists():
            raise FileExistsError(f"Session {session_id!r} already exists")

        storage = SQLiteStorageManager(path, check_same_thread=check_same_thread)
        events = EventManager(backend=storage.event_backend)
        for event_type in SESSION_EVENT_TYPES:
            events.register_event_type(event_type)
        started = SessionStarted(
            origin=origin,
            model=model,
            agent=agent,
            working_directory=working_directory,
            provisional=provisional,
        )
        try:
            events.add(started)
        except BaseException:
            storage.close()
            raise
        timestamp = started.timestamp.timestamp()
        return SessionHandle(
            self,
            storage,
            SessionInfo(
                id=session_id,
                model=model,
                agent=agent,
                started_at=timestamp,
                last_active=timestamp,
                working_directory=working_directory,
                origin=origin,
                provisional=provisional,
            ),
        )

    def open(self, session_id: str, *, check_same_thread: bool = True) -> SessionHandle:
        path = self.path_for(session_id)
        info = self._read_info(path)
        if info is None:
            raise SessionNotFoundError(f"Session {session_id!r} was not found or is invalid")
        storage = SQLiteStorageManager(path, check_same_thread=check_same_thread)
        return SessionHandle(self, storage, info)

    def get(self, session_id: str) -> SessionInfo:
        path = self.path_for(session_id)
        info = self._read_info(path)
        if info is None:
            raise SessionNotFoundError(f"Session {session_id!r} was not found or is invalid")
        return info

    def list(self, *, limit: int = 20) -> list[SessionInfo]:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        if limit == 0 or not self.root.exists():
            return []
        sessions = [
            info
            for path in self.root.glob("*.db")
            if not path.stem.endswith("-memory")
            if (info := self._read_info(path)) is not None
            if not (info.provisional and info.turn_count == 0)
        ]
        sessions.sort(key=lambda info: info.last_active, reverse=True)
        return sessions[:limit]

    def load_turns(self, session_id: str) -> list[SessionTurn]:
        path = self.path_for(session_id)
        rows = self._read_rows(path, event_types=_TURN_EVENT_TYPES)
        turns: list[SessionTurn] = []
        for event_type, raw in rows:
            try:
                if event_type in _USER_EVENT_TYPES:
                    content = raw.get("content", raw.get("text", ""))
                    role: Literal["user", "agent"] = "user"
                else:
                    content = raw.get("content", "")
                    role = "agent"
                turns.append(
                    SessionTurn(
                        role=role,
                        content=str(content),
                        timestamp=self._timestamp(raw, fallback=time.time()),
                    )
                )
            except Exception:
                logger.debug("Skipping invalid turn in %s", path, exc_info=True)
        return turns

    def find_by_prefix(self, prefix: str) -> list[str]:
        if not prefix or any(separator in prefix for separator in ("/", "\\", "\x00")):
            return []
        if not self.root.exists():
            return []
        matches = [
            path
            for path in self.root.glob("*.db")
            if path.stem.startswith(prefix) and not path.stem.endswith("-memory")
        ]
        matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return [path.stem for path in matches]

    def load_transcript(self, session_id: str) -> list[SessionTranscriptEvent]:
        """Read messages and saved tool activity without acquiring session ownership.

        Provider replies, private reasoning and runtime state are excluded at the
        SQL boundary. Runtime-only activities from older versions were not saved
        and therefore cannot be reconstructed by this reader.
        """
        rows = self._read_rows(self.path_for(session_id), event_types=_TRANSCRIPT_EVENT_TYPES)
        result: list[SessionTranscriptEvent] = []
        for event_type, raw in rows:
            if event_type == "SessionActivity":
                activity = raw.get("activity")
                if not isinstance(activity, dict):
                    continue
                event_type = str(activity.get("event_type", ""))
                if event_type not in _TRANSCRIPT_EVENT_TYPES or event_type == "SessionActivity":
                    continue
                raw = activity
            result.append(SessionTranscriptEvent(event_type, raw))
        return result

    def delete(self, session_id: str) -> bool:
        """Delete an inactive session database and its SQLite sidecars."""
        return delete_sqlite_database(self.path_for(session_id))

    def _read_info(self, path: Path) -> SessionInfo | None:
        try:
            fallback = path.stat().st_mtime
        except OSError:
            return None

        try:
            connection = sqlite3.connect(str(path))
            try:
                start_row = connection.execute(
                    "SELECT event_type, data FROM events "
                    "WHERE event_type IN (?, ?) ORDER BY insertion_order LIMIT 1",
                    tuple(_START_EVENT_TYPES),
                ).fetchone()
                if start_row is None:
                    return None

                title_rows = connection.execute(
                    "SELECT data FROM events WHERE event_type IN (?, ?) ORDER BY insertion_order",
                    tuple(_TITLE_EVENT_TYPES),
                ).fetchall()
                turn_count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM events WHERE event_type IN (?, ?)",
                        tuple(_USER_EVENT_TYPES),
                    ).fetchone()[0]
                )
                last_row = connection.execute(
                    "SELECT data FROM events ORDER BY insertion_order DESC LIMIT 1"
                ).fetchone()
            finally:
                connection.close()
        except (OSError, sqlite3.Error):
            logger.debug("Could not summarize session database %s", path, exc_info=True)
            return None

        start = self._decode_data(start_row[1], path)
        if start is None:
            return None
        start_event_type = str(start_row[0])
        started_at = self._timestamp(start, fallback=fallback)
        last_active = fallback
        if last_row is not None:
            last = self._decode_data(last_row[0], path)
            if last is not None:
                last_active = max(last_active, self._timestamp(last, fallback=fallback))

        title: str | None = None
        title_is_user_set = False
        for (data,) in title_rows:
            raw = self._decode_data(data, path)
            if raw is None:
                continue
            title = str(raw.get("title", raw.get("name", ""))) or None
            title_is_user_set = title_is_user_set or bool(
                raw.get("user_set", raw.get("user_named", False))
            )

        return SessionInfo(
            id=path.stem,
            model=str(start.get("model", "")),
            agent=str(start.get("agent", start.get("agent_cls", ""))),
            started_at=started_at,
            last_active=last_active,
            turn_count=turn_count,
            provisional=start.get("provisional") is True,
            working_directory=str(start.get("working_directory", start.get("working_dir", ""))),
            title=title,
            title_is_user_set=title_is_user_set,
            origin=str(
                start.get(
                    "origin",
                    start.get(
                        "host",
                        "tui" if start_event_type == "TUISessionStart" else "",
                    ),
                )
            ),
        )

    @staticmethod
    def _decode_data(data: object, path: Path) -> dict[str, object] | None:
        if not isinstance(data, (str, bytes, bytearray)):
            logger.debug("Skipping invalid session event payload in %s", path)
            return None
        try:
            raw = json.loads(data)
        except (TypeError, json.JSONDecodeError):
            logger.debug("Skipping corrupt session event in %s", path, exc_info=True)
            return None
        return raw if isinstance(raw, dict) else None

    @staticmethod
    def _read_rows(
        path: Path,
        *,
        event_types: frozenset[str] | None = None,
    ) -> list[tuple[str, dict[str, object]]]:
        if not path.exists():
            return []
        try:
            connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
            try:
                if event_types:
                    placeholders = ", ".join("?" for _ in event_types)
                    query = (
                        "SELECT event_type, data FROM events "
                        f"WHERE event_type IN ({placeholders}) ORDER BY insertion_order"
                    )
                    db_rows = connection.execute(query, tuple(event_types)).fetchall()
                else:
                    db_rows = connection.execute(
                        "SELECT event_type, data FROM events ORDER BY insertion_order"
                    ).fetchall()
            finally:
                connection.close()
        except (OSError, sqlite3.Error):
            logger.debug("Could not read session database %s", path, exc_info=True)
            return []

        rows: list[tuple[str, dict[str, object]]] = []
        for event_type, data in db_rows:
            raw = SessionStore._decode_data(data, path)
            if raw is not None:
                rows.append((str(event_type), raw))
        return rows

    @staticmethod
    def _timestamp(raw: dict[str, object], *, fallback: float) -> float:
        value = raw.get("timestamp")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value).timestamp()
            except ValueError:
                pass
        return fallback

    @staticmethod
    def _validate_id(session_id: str) -> str:
        if (
            not session_id
            or session_id in {".", ".."}
            or any(separator in session_id for separator in ("/", "\\", "\x00"))
        ):
            raise InvalidSessionIdError(f"Invalid session ID {session_id!r}")
        return session_id
