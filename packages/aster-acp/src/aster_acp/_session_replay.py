# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persist observational tool activity and render it without executing old tools."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from acp import (
    start_tool_call,
    text_block,
    tool_content,
    tool_diff_content,
    update_agent_message,
    update_user_message,
)
from acp.schema import ToolCallLocation, ToolCallStart
from aster_cli.coding import (
    FileEdit,
    TerminalCommandFinished,
    TerminalCommandOutput,
    TerminalCommandStarted,
)
from aster_cli.sessions import (
    SessionActivity,
    SessionHandle,
    SessionSteering,
    SessionTranscriptEvent,
)
from pydantic import ValidationError

from aster.agentdoc import pformat
from aster.context_blocks.events import EventBase, ResultStatus, ToolCallEvent
from aster.events import PythonOutput
from aster.runtime.event_manager import EventManager
from aster_acp._steering import steering_update
from aster_acp._terminal_result import terminal_output_text, terminal_result_update
from aster_acp.event_bridge import _python_content

logger = logging.getLogger(__name__)
_ACTIVITY_TYPES = (FileEdit, TerminalCommandStarted, TerminalCommandOutput, TerminalCommandFinished)
_REPLAY_TYPES = {cls.__name__: cls for cls in (*_ACTIVITY_TYPES, ToolCallEvent, PythonOutput)}
_USER_TYPES = {"SessionUserMessage", "TUIUserInput"}
_MESSAGE_TYPES = _USER_TYPES | {"AgentMessage", "TUIAgentMessage"}


def record_activity(events: EventManager, handle: SessionHandle) -> Callable[[], None]:
    """Mirror otherwise transient activity into metadata, never model context.

    The source event remains transient. In particular, terminal stdin is not
    copied into history. Keeping its original ID/timestamp allows deterministic
    replay and execution timing independently of notification delivery delays.
    """

    def record(event: EventBase) -> None:
        if isinstance(event, _ACTIVITY_TYPES):
            handle.events.add(
                SessionActivity(
                    activity=event.model_dump(mode="json", exclude={"stdin"}),
                    timestamp=event.timestamp,
                )
            )

    return events.on("*", record)


def _timestamp(raw: dict[str, object]) -> float | None:
    value = raw.get("timestamp")
    try:
        if isinstance(value, str):
            value = datetime.fromisoformat(value).timestamp()
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            return float(value) if math.isfinite(value) else None
    except (ValueError, OverflowError, OSError):
        pass
    return None


@dataclass
class _Tool:
    card: ToolCallStart
    started: float | None
    source: str = ""
    output: str = ""

    def finish(self, raw: dict[str, object]) -> None:
        ended = _timestamp(raw)
        if ended is not None and self.started is not None and ended >= self.started:
            self.card.field_meta = {"aster": {"elapsedSeconds": ended - self.started}}


def transcript_updates(records: list[SessionTranscriptEvent]) -> list[Any]:
    """Render final cards at their original start positions, preserving chronology.

    Replayed cards are terminal from the outset. A crash that never recorded an
    output remains interrupted, with no fabricated duration or running spinner.
    Unknown or corrupt individual tool records do not discard healthy messages.
    """
    updates: list[Any] = []
    python: dict[str, _Tool] = {}
    terminal: dict[str, _Tool] = {}
    steering: dict[str, tuple[int, SessionSteering]] = {}
    for index, record in enumerate(records):
        if record.event_type != "SessionSteering":
            continue
        try:
            event = SessionSteering.model_validate(record.data)
        except ValidationError:
            continue
        first_index = steering[event.message_id][0] if event.message_id in steering else index
        steering[event.message_id] = first_index, event
    steering_at = dict(steering.values())
    for index, record in enumerate(records):
        kind, raw = record.event_type, record.data
        if kind == "SessionSteering":
            event = steering_at.get(index)
            if event is not None:
                if event.state == "accepted":
                    event = event.model_copy(
                        update={
                            "state": "interrupted",
                            "reason": "The original turn was interrupted before application.",
                        }
                    )
                updates.append(steering_update(event))
            continue
        if kind in _MESSAGE_TYPES:
            content = raw.get("content", raw.get("text", ""))
            if not isinstance(content, str):
                continue
            # Preserve separation for ACP clients that concatenate chunks without
            # interpreting ASTER's explicit complete-message metadata.
            content = content if content.endswith("\n") else content + "\n"
            update = (update_user_message if kind in _USER_TYPES else update_agent_message)(
                text_block(content)
            )
            metadata = raw.get("metadata")
            stream_id = metadata.get("stream_message_id") if isinstance(metadata, dict) else None
            message_id = stream_id or raw.get("id") or f"replay-{index}"
            update.field_meta = {
                "aster": {"messageId": str(message_id), "messageState": "complete", "replace": True}
            }
            updates.append(update)
            continue
        model = _REPLAY_TYPES.get(kind)
        if model is None:
            continue
        try:
            event = model.model_validate(raw)
        except (ValidationError, ValueError, TypeError):
            logger.debug("Skipping invalid saved %s activity", kind)
            continue
        if isinstance(event, ToolCallEvent):
            if (
                event.name not in {"execute_python", "python_cell"}
                or event.metadata.get("prefill") is True
                or event.metadata.get("synthetic") is True
            ):
                continue
            code = event.arguments.get("code", "")
            if not isinstance(code, str):
                code = repr(code)
            card = start_tool_call(
                event.tool_call_id,
                "Python interrupted",
                kind="other",
                status="failed",
                content=_python_content(
                    code, "No completion was recorded before the session stopped."
                ),
            )
            python[event.tool_call_id] = _Tool(card, _timestamp(raw), source=code)
            updates.append(card)
        elif isinstance(event, PythonOutput):
            tool = python.pop(event.tool_call_id, None)
            if tool is None:
                continue
            parts = [
                part.rstrip() for part in (event.stdout, event.stderr, event.error) if part.strip()
            ]
            if event.value is not None:
                rendered = pformat(event.value, max_string=10_000, unquote_strings=True)
                parts.append(f"Out[{event.execution_count}]: {rendered}")
            if event.execution_status is ResultStatus.CANCELLED:
                parts.append("Cancelled by user.")
                tool.card.title = "Cancelled"
            elif event.execution_status is ResultStatus.ERROR:
                tool.card.title = "Python failed"
            elif event.execution_status is ResultStatus.COMPLETE:
                tool.card.title = "Ran Python"
                tool.card.status = "completed"
            else:
                parts.append("No completion was recorded before the session stopped.")
            tool.card.content = _python_content(tool.source, "\n".join(parts) or "Completed.")
            if event.execution_status is not ResultStatus.RUNNING:
                tool.finish(raw)
        elif isinstance(event, FileEdit):
            content = (
                [tool_diff_content(event.path, event.new_text, event.old_text)]
                if event.content_complete
                else [tool_content(text_block(event.diff or "File content was truncated."))]
            )
            line = max(0, event.start_line - 1) if event.start_line is not None else None
            updates.append(
                start_tool_call(
                    f"file-edit-{event.id}",
                    f"{'Created' if event.operation == 'create' else 'Edited'} {Path(event.path).name}",
                    kind="edit",
                    status="completed",
                    content=content,
                    locations=[ToolCallLocation(path=event.path, line=line)],
                    raw_input={"path": event.path, "operation": event.operation},
                )
            )
        elif isinstance(event, TerminalCommandStarted):
            card = start_tool_call(
                event.command_id,
                f"$ {event.command}",
                kind="execute",
                status="failed",
                content=[
                    tool_content(
                        text_block("No completion was recorded before the session stopped.")
                    )
                ],
                raw_input={"command": event.command, "working_directory": event.working_directory},
            )
            terminal[event.command_id] = _Tool(card, _timestamp(raw))
            updates.append(card)
        elif isinstance(event, TerminalCommandOutput):
            if (tool := terminal.get(event.command_id)) is not None:
                tool.output += terminal_output_text(event)
                tool.card.content = [
                    tool_content(
                        text_block(
                            tool.output + "\nNo completion was recorded before the session stopped."
                        )
                    )
                ]
        elif isinstance(event, TerminalCommandFinished):
            if (tool := terminal.pop(event.command_id, None)) is None:
                continue
            result = terminal_result_update(event, tool.output)
            tool.card.status = result.status
            tool.card.content = result.content
            tool.card.raw_output = result.raw_output
            tool.finish(raw)
    return updates
