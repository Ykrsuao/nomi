# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One foreground prompt's waiting activity and content-free diagnostics."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from contextlib import suppress
from contextvars import ContextVar
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from acp import start_tool_call, update_tool_call

from aster.context_blocks.events import ResultStatus, ToolCallEvent
from aster.events import LLMCallEnd, LLMCallStart, PythonOutput


class PromptProgress:
    """Keep waiting feedback separate from the answer and its delivery time.

    The activity ends when an answer is ready to enqueue. Diagnostics continue
    through delivery and prompt completion, including cleanup after the answer.
    Nothing here is added to the agent's event store or model context.
    """

    def __init__(
        self,
        started_at: float,
        backend: str,
        publish: Callable[[Any, PromptProgress], None],
    ) -> None:
        self.started_at = started_at
        self.backend = backend
        self.turn_id = uuid4().hex
        self.tool_call_id = f"aster-progress-{self.turn_id}"
        self.outcome: str | None = None
        self._publish = publish
        self._timing = any(
            os.getenv(name) == "1" for name in ("ASTER_PROMPT_TIMING", "ASTER_STARTUP_TIMING")
        )
        self._first: set[str] = set()
        self._title = "Preparing response"
        self._waiting = True
        self._accept_answers = True
        self._completed = False
        self._prefills: set[str] = set()
        self._calls: dict[tuple[str, int], int] = {}
        self._call_count = 0

    def start(self) -> None:
        self.mark("received", at=self.started_at)
        self.mark("accepted")
        self._publish(
            start_tool_call(self.tool_call_id, self._title, kind="other", status="in_progress"),
            self,
        )

    def mark(
        self,
        stage: str,
        *,
        at: float | None = None,
        call: int | None = None,
        outcome: str | None = None,
    ) -> None:
        """Only callers' fixed stage/outcome names and opaque IDs reach stderr."""
        if not self._timing:
            return
        record = {
            "pid": os.getpid(),
            "turn_id": self.turn_id,
            "backend": self.backend,
            "stage": stage,
            "elapsed_s": round((perf_counter() if at is None else at) - self.started_at, 6),
        }
        if call is not None:
            record["call"] = call
        if outcome is not None:
            record["outcome"] = outcome
        # Nested native generations run inside CodeAct's stderr capture. Like
        # runtime diagnostics, timings belong on the host stream, never in the
        # cell output returned to the model. Keep any caller/pytest redirection.
        from aster.runtime.actor import _stderr_buffer_var

        token = _stderr_buffer_var.set(None)
        try:
            with suppress(OSError, ValueError):
                print("[aster-prompt] " + json.dumps(record), file=sys.stderr, flush=True)
        finally:
            _stderr_buffer_var.reset(token)

    def _mark_first(self, stage: str) -> None:
        if stage not in self._first:
            self._first.add(stage)
            self.mark(stage)

    def status_sent(self) -> None:
        """Called by the message pump after a successful transport write."""
        self._mark_first("first_status")

    def answer_sent(self) -> None:
        """An answer queued by this prompt has reached the ACP transport."""
        self._mark_first("first_answer")

    def answer_queued(self) -> bool:
        """Return whether this is answer content, excluding host stop/error notes."""
        if not self._accept_answers:
            return False
        self._finish_wait("Response ready", "completed")
        return True

    def _set_title(self, title: str) -> None:
        if self._waiting and title != self._title:
            self._title = title
            self._publish(
                update_tool_call(self.tool_call_id, title=title, status="in_progress"), self
            )

    def _finish_wait(self, title: str, status: Literal["completed", "failed"]) -> None:
        if self._waiting:
            self._waiting = False
            self._publish(update_tool_call(self.tool_call_id, title=title, status=status), self)

    def stop(self, outcome: str) -> None:
        """Stop accepting answer candidates before a host-authored outcome note."""
        if self._completed:
            return
        self.outcome = outcome
        self._accept_answers = False
        if outcome == "cancelled":
            self._finish_wait("Cancelled", "failed")
        elif outcome in {"max_tokens", "max_turn_requests"}:
            self._finish_wait("Stopped at generation limit", "failed")
        elif outcome == "end_turn":
            self._finish_wait("Finished", "completed")
        else:
            self._finish_wait("Response failed", "failed")

    def complete(self, outcome: str) -> None:
        if self._completed:
            return
        self._completed = True
        self.outcome = outcome
        self.mark("complete", outcome=outcome)

    def on_tool_call(self, event: ToolCallEvent) -> None:
        if event.metadata.get("prefill") is True:
            self._prefills.add(event.tool_call_id)
            self.mark("prefill_start")
            self._set_title("Preparing execution environment")
        elif event.metadata.get("synthetic") is not True:
            self._set_title("Working on your request")

    def on_python_output(self, event: PythonOutput) -> None:
        if event.tool_call_id not in self._prefills:
            return
        self._prefills.remove(event.tool_call_id)
        outcome = (
            "cancelled"
            if event.execution_status is ResultStatus.CANCELLED
            else "failed"
            if event.execution_status is ResultStatus.ERROR
            else "complete"
        )
        self.mark("prefill_end", outcome=outcome)
        self._set_title("Preparing model request")

    def on_model_start(self, event: LLMCallStart) -> None:
        self._call_count += 1
        self._calls[event.generation_id, event.turn_number] = self._call_count
        self.mark("model_start", call=self._call_count)
        self._set_title("Waiting for model")

    def on_model_end(self, event: LLMCallEnd) -> None:
        call = self._calls.pop((event.generation_id, event.turn_number), None)
        if call is None:
            return
        outcome = (
            "complete"
            if event.success
            else "cancelled"
            if event.exception_type == "CancelledError"
            else "failed"
        )
        self.mark("model_end", call=call, outcome=outcome)
        if not self._calls:
            self._set_title(
                "Processing response"
                if event.success
                else "Stopping"
                if outcome == "cancelled"
                else "Model call failed"
            )

    def model_response(self) -> None:
        # acall() delivers a complete, validated response. Streaming clients
        # currently collect their chunks first: this is NOT provider first-token time.
        self._mark_first("first_model_response")


# Tasks spawned by a turn inherit its identity. A late callback from a cancelled
# turn must never be attributed to the next turn using the same ACP session.
current_prompt: ContextVar[PromptProgress | None] = ContextVar("aster_acp_prompt", default=None)
