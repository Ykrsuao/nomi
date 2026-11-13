# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate observational ASTER events into ACP session updates."""

import asyncio
import logging
import math
import re
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from acp import (
    start_tool_call,
    text_block,
    tool_content,
    tool_diff_content,
    update_agent_message,
    update_tool_call,
)
from acp.interfaces import Client
from acp.schema import ContentToolCallContent, Cost, ToolCallLocation, ToolCallProgress, UsageUpdate
from aster_cli.coding.activity import (
    FileEdit,
    TerminalCommandFinished,
    TerminalCommandOutput,
    TerminalCommandStarted,
)
from aster_cli.coding.agent import CodingAgent

from aster.agentdoc import pformat
from aster.agents.summarization import summary_fork_active
from aster.context_blocks.events import EventBase, ResultStatus, ToolCallEvent
from aster.events import LLMCallEnd, LLMCallStart, LLMResponse, PythonOutput
from aster.interactive import AgentMessage, Done, NeedInput, Waiting
from aster.unifiedllm.streaming import observe_stream
from aster_acp._prompt import PromptProgress, current_prompt
from aster_acp._streaming_reply import ReplyStream
from aster_acp._tasks import TaskTracker
from aster_acp._terminal_result import terminal_output_text, terminal_result_update

# ACP owns stdout for JSON-RPC; diagnostics belong on stderr, which is where
# the logging default sends them.
logger = logging.getLogger(__name__)

_STOP = object()

# Bound on a rendered Out[n] value; large results belong in the agent's
# context, not repeated in full inside a client tool card.
_MAX_VALUE_CHARS = 10_000


@dataclass(frozen=True, slots=True)
class _BestEffortUpdate:
    value: Any
    progress: PromptProgress | None = None


@dataclass(frozen=True, slots=True)
class _AnswerUpdate:
    value: Any
    progress: PromptProgress


def _fenced_code(text: str, language: str) -> str:
    """Wrap text in a Markdown fence that cannot collide with its contents."""
    longest_run = max((len(match.group()) for match in re.finditer(r"`+", text)), default=0)
    fence = "`" * max(3, longest_run + 1)
    return f"{fence}{language}\n{text}\n{fence}"


def _python_content(code: str, output: str | None = None) -> list[ContentToolCallContent]:
    """Render Python source and output as Markdown visible in ACP clients."""

    def block(text: str) -> ContentToolCallContent:
        value = text_block(text)
        # The fence is a transport hint for clients that render Markdown. TUI
        # clients can remove only this wrapper while preserving literal
        # backticks that occur inside the source or output.
        value.field_meta = {"aster": {"format": "fenced-code"}}
        return tool_content(value)

    content = [block(_fenced_code(code, "python"))]
    if output is not None:
        content.append(block(_fenced_code(output, "text")))
    return content


def _python_title(code: str) -> str | None:
    """Use an explicit purpose comment, never guess that arbitrary code is safe."""
    first = next((line.strip() for line in code.splitlines() if line.strip()), "")
    if first.startswith("#"):
        title = first.lstrip("#").strip()
        if 3 <= len(title) <= 120 and not title.startswith("!"):
            return title
    return None


class ACPEventBridge:
    def __init__(
        self,
        agent: CodingAgent,
        client: Client,
        session_id: str,
        *,
        stream_messages: bool = False,
        structured_progress: bool = False,
    ) -> None:
        self.agent = agent
        self.client = client
        self.session_id = session_id
        self._stream_messages = stream_messages
        self._structured_progress = structured_progress
        self._tasks = TaskTracker(agent)
        self._tasks_wait_reason = ""
        self._tasks_last_update: dict[str, Any] | None = None
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._error: Exception | None = None
        # Set when the pump exits on a BaseException it cannot handle. Nothing
        # resolves flush markers after that, so flush must fail rather than wait.
        self._pump_failure: BaseException | None = None
        self._closed = False
        self._open_tools: set[str] = set()
        self._tool_started: dict[str, float] = {}
        self._tool_elapsed: dict[str, float] = {}
        self._python_source: dict[str, str] = {}
        self._terminal_output: dict[str, str] = {}
        # Partial stdout/stderr of a cancelled cell, shown on its card when
        # fail_open_tools closes it.
        self._cancelled_output: dict[str, str] = {}
        self._cost_usd = 0.0
        self._cost_complete = True
        self._prompt_progress: PromptProgress | None = None
        self._reply_stream: ReplyStream | None = None
        self._unsubscribers: list[Callable[[], None]] = [
            agent.event_manager.intercept("llm_call", self._stream_reply),
            agent.event_manager.on("AgentMessage", self._on_agent_message),
            agent.event_manager.on("ToolCallEvent", self._on_tool_call),
            agent.event_manager.on("PythonOutput", self._on_python_output),
            agent.event_manager.on("LLMResponse", self._on_llm_response),
            agent.event_manager.on("LLMCallStart", self._on_llm_start),
            agent.event_manager.on("LLMCallEnd", self._on_llm_end),
            agent.event_manager.on("FileEdit", self._on_file_edit),
            agent.event_manager.on("TerminalCommandStarted", self._on_terminal_started),
            agent.event_manager.on("TerminalCommandOutput", self._on_terminal_output),
            agent.event_manager.on("TerminalCommandFinished", self._on_terminal_finished),
        ]
        self._pump_task = asyncio.create_task(self._pump(), name="aster-acp-events")
        self._tasks_poll_task = (
            asyncio.create_task(self._poll_tasks(), name="aster-acp-tasks")
            if structured_progress
            else None
        )

    def task_snapshot(self) -> dict[str, Any]:
        """Return actual plan state and background handles for the task inspector."""
        return self._tasks.snapshot(self._tasks_wait_reason)

    async def stop_task(self, job_id: str) -> dict[str, Any]:
        stopped = await self._tasks.stop(job_id)
        self.refresh_tasks()
        return {"id": job_id, "stopped": stopped, "tasks": self.task_snapshot()}

    def set_wait_reason(self, reason: str) -> None:
        self._tasks_wait_reason = reason
        self.refresh_tasks()

    def refresh_tasks(self) -> None:
        """Coalesce observational updates without adding them to model history."""
        if not self._structured_progress or self._closed:
            return
        update = self._tasks.update(self._tasks_wait_reason)
        serialized = update.model_dump(mode="json", by_alias=True, exclude_none=True)
        if serialized != self._tasks_last_update:
            self._tasks_last_update = serialized
            self.publish_best_effort(update)

    def observe_result(self, result: Any) -> None:
        if isinstance(result, Waiting):
            self.set_wait_reason(f"{result.explanation} (waiting: {', '.join(result.on)})")
        elif isinstance(result, NeedInput):
            self.set_wait_reason(result.reason or result.question)
        elif isinstance(result, Done):
            self.set_wait_reason("")
        else:
            self.refresh_tasks()

    async def _poll_tasks(self) -> None:
        # Jobs and todo mutations can occur inside a long-running cell before
        # PythonOutput. The bounded host snapshot exposes them while it runs.
        while True:
            await asyncio.sleep(0.5)
            if self._prompt_progress is not None or self.agent.queue_manager.jobs():
                self.refresh_tasks()

    def _enqueue(self, update: Any) -> None:
        if not self._closed:
            self._queue.put_nowait(update)

    def _stop_tool_clock(self, tool_call_id: str) -> None:
        started = self._tool_started.pop(tool_call_id, None)
        if started is not None:
            self._tool_elapsed[tool_call_id] = max(0.0, perf_counter() - started)

    def _timed_update(self, update: ToolCallProgress) -> ToolCallProgress:
        # Measure execution before the asynchronous pump delivers notifications.
        # A cancelled Python cell has already stopped its clock at PythonOutput.
        self._stop_tool_clock(update.tool_call_id)
        elapsed = self._tool_elapsed.pop(update.tool_call_id, None)
        if elapsed is not None:
            update.field_meta = {"aster": {"elapsedSeconds": elapsed}}
        return update

    def publish(self, update: Any) -> None:
        """Queue a host-originated session update on the ordered ACP stream."""
        self._enqueue(update)

    def publish_best_effort(self, update: Any) -> None:
        """Queue bootstrap metadata without poisoning the live event stream."""
        self._enqueue(_BestEffortUpdate(update))

    def _publish_progress(self, update: Any, progress: PromptProgress) -> None:
        self._enqueue(_BestEffortUpdate(update, progress))

    def _current_progress(self) -> PromptProgress | None:
        progress = current_prompt.get()
        if progress is self._prompt_progress and progress is not None and progress.outcome is None:
            return progress
        return None

    @asynccontextmanager
    async def prompt(self, started_at: float, backend: str) -> AsyncIterator[PromptProgress]:
        """Observe one claimed foreground turn, including its final transport flush."""
        progress = PromptProgress(started_at, backend, self._publish_progress)
        token = current_prompt.set(progress)
        self._prompt_progress = progress
        self.set_wait_reason("Preparing response")
        try:
            progress.start()
            try:
                yield progress
            except BaseException as exc:
                if self._reply_stream is not None:
                    self._reply_stream.interrupt()
                outcome = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
                self.set_wait_reason("Cancelled" if outcome == "cancelled" else "Response failed")
                progress.stop(outcome)
                with suppress(Exception):
                    await self.flush(progress=progress)
                raise
            else:
                if self._reply_stream is not None:
                    self._reply_stream.interrupt()
                outcome = progress.outcome or "end_turn"
                if outcome == "end_turn":
                    self.set_wait_reason("")
                elif outcome == "cancelled":
                    self.set_wait_reason("Cancelled")
                progress.stop(outcome)
                try:
                    await self.flush(progress=progress)
                except BaseException as exc:
                    # The pump still owns queued answers if its waiter is
                    # cancelled. Its flush marker records completion only once
                    # those writes have actually finished (or been dropped).
                    progress.stop(
                        "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
                    )
                    raise
        finally:
            self._reply_stream = None
            self._prompt_progress = None
            current_prompt.reset(token)

    def _on_agent_message(self, event: EventBase) -> None:
        if not isinstance(event, AgentMessage):
            return
        update = update_agent_message(text_block(event.content))
        message_id = (
            self._reply_stream.accept(event.content)
            if self._reply_stream and self._current_progress() is not None
            else None
        )
        if message_id:
            # Event handlers run before persistence; keep the final event linked
            # to its preview so loading the session uses the same identity.
            event.metadata["stream_message_id"] = message_id
        update.field_meta = {
            "aster": {
                "messageId": message_id or str(event.id),
                "messageState": "complete",
                "replace": True,
            }
        }
        progress = self._current_progress()
        if progress is not None and event.content.strip() and progress.answer_queued():
            self._enqueue(_AnswerUpdate(update, progress))
        else:
            self._enqueue(update)

    async def _stream_reply(self, ctx: Any, nxt: Any) -> Any:
        progress = self._current_progress()
        method = getattr(getattr(ctx.runtime, "_current_method", None), "__name__", "")
        if (
            not self._stream_messages
            or progress is None
            or ctx.agent is not self.agent
            or method != "handle"
            or summary_fork_active()
        ):
            return await nxt(ctx)
        if self._reply_stream is not None:
            self._reply_stream.interrupt()

        def publish(message_id: str, text: str, state: str, replace: bool) -> None:
            if self._current_progress() is not progress:
                return
            update = update_agent_message(text_block(text))
            update.field_meta = {
                "aster": {
                    "messageId": message_id,
                    "messageState": state,
                    "replace": replace,
                }
            }
            if text.strip() and progress.answer_queued():
                self._enqueue(_AnswerUpdate(update, progress))
            else:
                self._enqueue(update)

        stream = ReplyStream(publish)
        self._reply_stream = stream

        def observe(kind: str, value: Any) -> None:
            # If this middleware wraps the summarizer, its task inherits the
            # observer. Ignore both summary deltas and provider resets so the
            # pending foreground reply retains its identity until validation.
            if not summary_fork_active():
                stream.observe(kind, value)

        try:
            with observe_stream(observe):
                result = await nxt(ctx)
            stream.finish(result.response)
            return result
        except BaseException:
            stream.interrupt()
            raise

    def _on_tool_call(self, event: EventBase) -> None:
        if isinstance(event, ToolCallEvent) and (progress := self._current_progress()) is not None:
            progress.on_tool_call(event)
        if (
            not isinstance(event, ToolCallEvent)
            or event.name not in {"execute_python", "python_cell"}
            or event.metadata.get("prefill") is True
            # codeact manufactures an execute_python call to carry a prose-only
            # reply. Nothing ran, so showing it as a Python card would present
            # the model's own text, commented out, as a completed execution.
            or event.metadata.get("synthetic") is True
        ):
            return
        code = event.arguments.get("code", "")
        if not isinstance(code, str):
            code = repr(code)
        self._open_tools.add(event.tool_call_id)
        self._tool_started[event.tool_call_id] = perf_counter()
        self._tool_elapsed.pop(event.tool_call_id, None)
        self._python_source[event.tool_call_id] = code
        title = _python_title(code)
        skipped = event.metadata.get("not_executed") is True
        if not skipped:
            self.set_wait_reason(title or "Executing Python")
        self._enqueue(
            start_tool_call(
                event.tool_call_id,
                "Not executed" if skipped else title or "Running Python",
                # Zed 1.14 treats every ``execute`` tool as a terminal card.
                # A plain-content execute card has neither a terminal nor an
                # output disclosure, so its source cannot be opened. Python is
                # structured Markdown content, not a client-owned terminal.
                kind="other",
                status="failed" if skipped else "in_progress",
                content=_python_content(code),
            )
        )

    def _on_python_output(self, event: EventBase) -> None:
        if isinstance(event, PythonOutput):
            self.refresh_tasks()
        if isinstance(event, PythonOutput) and (progress := self._current_progress()) is not None:
            progress.on_python_output(event)
        if not isinstance(event, PythonOutput) or event.tool_call_id not in self._open_tools:
            return
        self._stop_tool_clock(event.tool_call_id)
        # Core records a cancelled cell with whatever it printed. Keep that
        # output and leave the card open so cancel()'s fail_open_tools closes
        # it as "Cancelled" with the partial output shown.
        if event.execution_status is ResultStatus.CANCELLED:
            partial = "\n".join(
                part.rstrip() for part in (event.stdout, event.stderr) if part.strip()
            )
            if partial:
                self._cancelled_output[event.tool_call_id] = partial
            return
        self._open_tools.discard(event.tool_call_id)
        code = self._python_source.pop(event.tool_call_id, "")
        title = _python_title(code)
        parts = [
            part.rstrip() for part in (event.stdout, event.stderr, event.error) if part.strip()
        ]
        # A cell whose last line is a bare expression produces no stdout: the
        # result arrives as ``value`` and codeact shows it to the model as
        # Out[n]. Without this the client is told there was no output while the
        # agent is reasoning from one.
        if event.value is not None:
            rendered = pformat(event.value, max_string=_MAX_VALUE_CHARS, unquote_strings=True)
            parts.append(f"Out[{event.execution_count}]: {rendered}")
        output = "\n".join(parts) or "Completed."
        status: Literal["failed", "completed"] = (
            "failed" if event.execution_status is ResultStatus.ERROR else "completed"
        )
        self._enqueue(
            self._timed_update(
                update_tool_call(
                    event.tool_call_id,
                    title="Not executed"
                    if event.metadata.get("not_executed") is True
                    else (f"Failed: {title}" if title else "Python failed")
                    if status == "failed"
                    else (f"Completed: {title}" if title else "Ran Python"),
                    status=status,
                    content=_python_content(code, output),
                )
            )
        )

    def _on_file_edit(self, event: EventBase) -> None:
        if not isinstance(event, FileEdit):
            return
        tool_call_id = f"file-edit-{uuid4()}"
        path = event.path
        title = f"{'Created' if event.operation == 'create' else 'Edited'} {Path(path).name}"
        if event.content_complete:
            content = [tool_diff_content(path, event.new_text, event.old_text)]
        else:
            content = [tool_content(text_block(event.diff or "File content was truncated."))]
        line = max(0, event.start_line - 1) if event.start_line is not None else None
        self._enqueue(
            start_tool_call(
                tool_call_id,
                title,
                kind="edit",
                status="completed",
                content=content,
                locations=[ToolCallLocation(path=path, line=line)],
                raw_input={"path": path, "operation": event.operation},
            )
        )

    def _on_terminal_started(self, event: EventBase) -> None:
        if not isinstance(event, TerminalCommandStarted):
            return
        self.set_wait_reason(f"Running command: {event.command[:500]}")
        self._open_tools.add(event.command_id)
        self._tool_started[event.command_id] = perf_counter()
        self._tool_elapsed.pop(event.command_id, None)
        self._terminal_output[event.command_id] = ""
        self._enqueue(
            start_tool_call(
                event.command_id,
                f"$ {event.command}",
                kind="execute",
                status="in_progress",
                raw_input={
                    "command": event.command,
                    "working_directory": event.working_directory,
                },
            )
        )

    def _on_terminal_output(self, event: EventBase) -> None:
        if not isinstance(event, TerminalCommandOutput) or event.command_id not in self._open_tools:
            return
        output = self._terminal_output.get(event.command_id, "") + terminal_output_text(event)
        self._terminal_output[event.command_id] = output
        self._enqueue(
            update_tool_call(
                event.command_id,
                status="in_progress",
                content=[tool_content(text_block(output))],
            )
        )

    def _on_terminal_finished(self, event: EventBase) -> None:
        if not isinstance(event, TerminalCommandFinished):
            return
        self._stop_tool_clock(event.command_id)
        self._open_tools.discard(event.command_id)
        output = self._terminal_output.pop(event.command_id, "")
        self._enqueue(self._timed_update(terminal_result_update(event, output)))
        self.refresh_tasks()

    def _on_llm_start(self, event: EventBase) -> None:
        if isinstance(event, LLMCallStart) and (progress := self._current_progress()) is not None:
            progress.on_model_start(event)
            self.set_wait_reason("Waiting for model")

    def _on_llm_end(self, event: EventBase) -> None:
        if isinstance(event, LLMCallEnd) and (progress := self._current_progress()) is not None:
            progress.on_model_end(event)
            self.set_wait_reason("Processing response" if event.success else "Model call stopped")

    def _on_llm_response(self, event: EventBase) -> None:
        if not isinstance(event, LLMResponse):
            return
        if (progress := self._current_progress()) is not None:
            progress.model_response()
        usage = event.usage
        if usage is None:
            self._cost_complete = False
            self._enqueue(
                UsageUpdate(
                    session_update="usage_update",
                    used=0,
                    size=0,
                    field_meta={"aster": {"usageAvailable": False}},
                )
            )
            return
        if (
            not usage.cost_known
            or isinstance(usage.cost_usd, bool)
            or not isinstance(usage.cost_usd, (int, float))
            or not math.isfinite(usage.cost_usd)
            or usage.cost_usd < 0
        ):
            self._cost_complete = False
        elif self._cost_complete:
            self._cost_usd += usage.cost_usd
            self._cost_complete = math.isfinite(self._cost_usd)
        context_window = getattr(self.agent.llm, "context_window", None)
        size = (
            max(context_window, usage.input_tokens) if context_window and context_window > 0 else 0
        )
        self._enqueue(
            UsageUpdate(
                session_update="usage_update",
                used=usage.input_tokens if usage.input_tokens_reported else 0,
                size=size if usage.input_tokens_reported else 0,
                cost=Cost(amount=self._cost_usd, currency="USD") if self._cost_complete else None,
                field_meta=(
                    None if usage.input_tokens_reported else {"aster": {"usageAvailable": False}}
                ),
            )
        )

    def _stopped_error(self, cause: BaseException) -> RuntimeError:
        error = RuntimeError("ACP event bridge stopped")
        error.__cause__ = cause
        return error

    def _fail_pending_flushes(self, cause: BaseException) -> None:
        """Resolve every queued flush marker once the pump can no longer run."""
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if isinstance(item, asyncio.Future) and not item.done():
                item.set_exception(self._stopped_error(cause))
            self._queue.task_done()

    async def _pump(self) -> None:
        try:
            await self._pump_loop()
        except BaseException as exc:
            # CancelledError is a BaseException, so a transport cancelled during
            # client disconnect used to kill this task silently. flush() waits on
            # a marker only the pump resolves, so every later flush — and close(),
            # which flushes first — blocked forever.
            self._pump_failure = exc
            self._fail_pending_flushes(exc)
            raise

    async def _pump_loop(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is _STOP:
                    return
                if isinstance(item, asyncio.Future):
                    # Hand the failure to the turn that is flushing, then clear
                    # it. Latching it would mean every later turn ran the model
                    # and edited files while the client saw nothing and got a
                    # stale exception it could not act on.
                    error, self._error = self._error, None
                    if not item.done():
                        if error is None:
                            item.set_result(None)
                        else:
                            item.set_exception(error)
                    continue
                if isinstance(item, _BestEffortUpdate):
                    try:
                        await self.client.session_update(self.session_id, item.value)
                    except Exception:
                        logger.debug(
                            "ACP client rejected best-effort session update",
                            exc_info=True,
                        )
                    else:
                        if item.progress is not None:
                            item.progress.status_sent()
                elif self._error is None:
                    try:
                        update = item.value if isinstance(item, _AnswerUpdate) else item
                        await self.client.session_update(self.session_id, update)
                    except Exception as exc:
                        # Skip the rest of this turn's updates rather than
                        # hammering a transport that just failed; the next
                        # flush reports the error and resets.
                        self._error = exc
                        logger.warning(
                            "ACP session %s dropped an update after a failed send",
                            self.session_id,
                            exc_info=True,
                        )
                    else:
                        if isinstance(item, _AnswerUpdate):
                            item.progress.answer_sent()
            finally:
                self._queue.task_done()

    async def flush(self, *, progress: PromptProgress | None = None) -> None:
        if self._closed:
            if progress is not None:
                progress.complete("failed")
                raise RuntimeError("ACP event bridge is closed")
            return
        if self._pump_failure is not None:
            if progress is not None:
                progress.complete("failed")
            raise self._stopped_error(self._pump_failure)
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        if progress is not None:

            def on_flushed(done: asyncio.Future[None]) -> None:
                # Consume the result even if the prompt stopped awaiting it.
                # This callback belongs to the FIFO boundary, not its waiter.
                try:
                    done.result()
                except asyncio.CancelledError:
                    progress.complete("cancelled")
                except BaseException:
                    progress.complete("failed")
                else:
                    progress.complete(progress.outcome or "end_turn")

            future.add_done_callback(on_flushed)
        self._queue.put_nowait(future)
        # Race the pump: if it dies without draining this marker, waiting on the
        # marker alone would never return.
        await asyncio.wait({future, self._pump_task}, return_when=asyncio.FIRST_COMPLETED)
        if future.done():
            future.result()
            return
        raise self._stopped_error(self._pump_failure or RuntimeError("pump exited"))

    async def fail_open_tools(self, reason: str, *, title: str | None = None) -> None:
        """Close out open tool calls, titling them with what actually happened.

        The title is the collapsed-card text, so it is the only thing a user
        sees without expanding. A fixed "Python interrupted" made a deliberate
        cancellation read as a technical failure.
        """
        for tool_call_id in tuple(self._open_tools):
            code = self._python_source.pop(tool_call_id, None)
            # Output shown before the close: a cancelled cell's partial output, or
            # what a terminal command streamed so far.
            partial = self._cancelled_output.pop(tool_call_id, None)
            if partial is None:
                partial = self._terminal_output.pop(tool_call_id, "").rstrip("\n")
            output = f"{partial}\n\n{reason}" if partial else reason
            content = (
                _python_content(code, output)
                if code is not None
                else [tool_content(text_block(output))]
            )
            self._enqueue(
                self._timed_update(
                    update_tool_call(
                        tool_call_id,
                        title=title or ("Python interrupted" if code is not None else None),
                        status="failed",
                        content=content,
                    )
                )
            )
        self._open_tools.clear()
        self._python_source.clear()
        self._terminal_output.clear()
        self._cancelled_output.clear()
        self._tool_started.clear()
        self._tool_elapsed.clear()

    async def close(self) -> None:
        if self._closed:
            return
        if self._tasks_poll_task is not None:
            self._tasks_poll_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._tasks_poll_task
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        # A turn that ended before its PythonOutput — an exception escaping the
        # strategy, say — leaves cards in_progress and their source retained.
        # fail_open_tools is otherwise only reached from session/cancel, so this
        # is the sole purge on an ordinary close.
        with suppress(Exception):
            await self.fail_open_tools("Session closed before this finished.", title="Unfinished")
        with suppress(Exception):
            await self.flush()
        self._closed = True
        self._queue.put_nowait(_STOP)
        with suppress(BaseException):
            await self._pump_task
