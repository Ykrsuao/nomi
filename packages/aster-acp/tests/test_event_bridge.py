# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for translating ASTER events into ACP updates."""

import asyncio
import json
from time import perf_counter
from typing import Any, cast

import pytest
from acp.schema import (
    AgentMessageChunk,
    ContentToolCallContent,
    FileEditToolCallContent,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    UsageUpdate,
)
from aster_acp.event_bridge import ACPEventBridge
from aster_cli.coding import (
    CodingAgent,
    FileEdit,
    TerminalCommandFinished,
    TerminalCommandOutput,
    TerminalCommandStarted,
)

from aster.context_blocks.events import ResultStatus, ToolCallEvent
from aster.events import LLMResponse, PythonOutput
from aster.interactive import AgentMessage, InteractiveAgent
from aster.llm_types import LLMUsage
from aster.unifiedllm import FakeLLMClient


class _RecordingClient:
    def __init__(self) -> None:
        self.updates: list[tuple[str, object]] = []

    async def session_update(self, session_id: str, update: object, **kwargs) -> None:
        self.updates.append((session_id, update))


def _content_text(content: ContentToolCallContent) -> str:
    block = cast(TextContentBlock, content.content)
    return block.text


async def test_python_timing_measures_execution_before_buffered_acp_delivery(tmp_path):
    """A blocking real cell must retain its duration when ACP delivers both ends together."""

    class BufferedClient(_RecordingClient):
        def __init__(self):
            super().__init__()
            self.release = asyncio.Event()
            self.received: list[float] = []

        async def session_update(self, session_id, update, **kwargs):
            await self.release.wait()
            self.received.append(perf_counter())
            await super().session_update(session_id, update, **kwargs)

    llm = FakeLLMClient.with_tool_call(
        "execute_python",
        {"code": "import time\ntime.sleep(0.15)\nreturn_result(Done(explanation='timed'))"},
    )
    agent = CodingAgent(llm=llm, cwd=tmp_path)
    client = BufferedClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]
    observed: dict[str, float] = {}

    def observe(event):
        if isinstance(event, ToolCallEvent) and not event.metadata.get("prefill"):
            observed.setdefault(event.tool_call_id, perf_counter())
        elif isinstance(event, PythonOutput) and event.tool_call_id in observed:
            observed[event.tool_call_id] = perf_counter() - observed[event.tool_call_id]

    agent.event_manager.on("*", observe)
    try:
        await asyncio.wait_for(agent.handle({"user_messages": ["Run timing check"]}), 10)
        assert not client.updates
        client.release.set()
        await bridge.flush()
        completed = next(
            update
            for _, update in client.updates
            if isinstance(update, ToolCallProgress) and update.title == "Ran Python"
        )
        measured = observed[completed.tool_call_id]
        assert measured >= 0.12
        payload = completed.model_dump(mode="json", by_alias=True, exclude_none=True)
        elapsed = payload.get("_meta", {}).get("aster", {}).get("elapsedSeconds")
        assert elapsed is not None, "tool completion lost its execution duration"
        assert elapsed == pytest.approx(measured, abs=0.03)
        assert client.received[-1] - client.received[0] < measured / 2
    finally:
        client.release.set()
        await bridge.close()
        await agent.close()


@pytest.mark.parametrize(
    "status", [ResultStatus.COMPLETE, ResultStatus.ERROR, ResultStatus.CANCELLED]
)
async def test_python_timing_stops_at_output_even_when_cancel_cleanup_is_later(
    tmp_path, monkeypatch, status
):
    clock = [10.0]
    monkeypatch.setattr("aster_acp.event_bridge.perf_counter", lambda: clock[0], raising=False)
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]
    try:
        agent.event_manager.add(
            ToolCallEvent(tool_call_id="timed", name="execute_python", arguments={"code": "work()"})
        )
        clock[0] = 12.5
        agent.event_manager.add(
            PythonOutput(tool_call_id="timed", execution_status=status, execution_count=1)
        )
        clock[0] = 100.0
        if status is ResultStatus.CANCELLED:
            await bridge.fail_open_tools("Cancelled by user.", title="Cancelled")
        await bridge.flush()
        final = next(update for _, update in client.updates if isinstance(update, ToolCallProgress))
        payload = final.model_dump(mode="json", by_alias=True, exclude_none=True)
        assert payload.get("_meta", {}).get("aster", {}).get("elapsedSeconds") == 2.5
        assert not bridge._tool_started
        assert not bridge._tool_elapsed
    finally:
        await bridge.close()
        await agent.close()


@pytest.mark.parametrize(
    "finished",
    [
        {"exit_code": 0},
        {"exit_code": 1},
        {"cancelled": True},
        {"timed_out": True},
        {"error": "launch failed"},
    ],
)
async def test_terminal_timing_covers_completed_failed_and_cancelled_commands(
    tmp_path, monkeypatch, finished
):
    clock = [20.0]
    monkeypatch.setattr("aster_acp.event_bridge.perf_counter", lambda: clock[0], raising=False)
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]
    try:
        agent.event_manager.add(
            TerminalCommandStarted(
                command_id="timed", command="work", working_directory=str(tmp_path)
            )
        )
        clock[0] = 23.25
        agent.event_manager.add(TerminalCommandFinished(command_id="timed", **finished))
        clock[0] = 100.0
        await bridge.flush()
        final = next(update for _, update in client.updates if isinstance(update, ToolCallProgress))
        payload = final.model_dump(mode="json", by_alias=True, exclude_none=True)
        assert payload.get("_meta", {}).get("aster", {}).get("elapsedSeconds") == 3.25
        assert not bridge._tool_started
        assert not bridge._tool_elapsed
    finally:
        await bridge.close()
        await agent.close()


async def test_completed_file_edit_has_no_fabricated_execution_duration(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]
    try:
        agent.event_manager.add(
            FileEdit(path=str(tmp_path / "test.txt"), operation="create", new_text="hello")
        )
        await bridge.flush()
        update = client.updates[0][1]
        assert isinstance(update, ToolCallStart)
        payload = update.model_dump(mode="json", by_alias=True, exclude_none=True)
        assert "elapsedSeconds" not in payload.get("_meta", {}).get("aster", {})
    finally:
        await bridge.close()
        await agent.close()


@pytest.mark.parametrize("tool_name", ["execute_python", "python_cell"])
async def test_bridge_preserves_message_tool_and_usage_order(tmp_path, tool_name):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]

    agent.event_manager.add(AgentMessage(content="Final answer"))
    agent.event_manager.add(
        ToolCallEvent(
            tool_call_id="prefill-1",
            name="execute_python",
            arguments={"code": "print('internal setup')"},
            metadata={"prefill": True},
        )
    )
    agent.event_manager.add(
        PythonOutput(
            tool_call_id="prefill-1",
            execution_status=ResultStatus.COMPLETE,
            execution_count=1,
            stdout="internal setup\n",
        )
    )
    agent.event_manager.add(
        ToolCallEvent(
            tool_call_id="call-1",
            name=tool_name,
            arguments={"code": "print('hello')"},
        )
    )
    agent.event_manager.add(
        PythonOutput(
            tool_call_id="call-1",
            execution_status=ResultStatus.COMPLETE,
            execution_count=1,
            stdout="hello\n",
        )
    )
    agent.event_manager.add(
        LLMResponse(usage=LLMUsage(input_tokens=40, output_tokens=10, cost_usd=0.25))
    )
    await bridge.flush()

    updates = [update for _, update in client.updates]
    assert {session_id for session_id, _ in client.updates} == {"session-1"}
    assert [type(update) for update in updates] == [
        AgentMessageChunk,
        ToolCallStart,
        ToolCallProgress,
        UsageUpdate,
    ]
    assert cast(AgentMessageChunk, updates[0]).content.text == "Final answer"
    started = cast(ToolCallStart, updates[1])
    assert started.kind == "other"
    assert started.status == "in_progress"
    assert started.raw_input is None
    assert started.content is not None
    assert len(started.content) == 1
    source = _content_text(cast(ContentToolCallContent, started.content[0]))
    assert source == "```python\nprint('hello')\n```"
    assert started.model_dump(mode="json", by_alias=True, exclude_none=True)["content"][0] == {
        "type": "content",
        "content": {
            "type": "text",
            "text": "```python\nprint('hello')\n```",
            "_meta": {"aster": {"format": "fenced-code"}},
        },
    }

    completed = cast(ToolCallProgress, updates[2])
    assert completed.title == "Ran Python"
    assert completed.status == "completed"
    assert completed.content is not None
    assert len(completed.content) == 2
    completed_source = _content_text(cast(ContentToolCallContent, completed.content[0]))
    output = _content_text(cast(ContentToolCallContent, completed.content[1]))
    assert completed_source == "```python\nprint('hello')\n```"
    assert output == "```text\nhello\n```"
    usage = cast(UsageUpdate, updates[3])
    assert usage.cost is not None
    assert usage.cost.amount == 0.25
    await bridge.close()
    await agent.close()


async def test_bridge_marks_failed_python_output(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]

    agent.event_manager.add(
        ToolCallEvent(
            tool_call_id="call-1",
            name="execute_python",
            arguments={"code": "raise RuntimeError('boom')"},
        )
    )
    agent.event_manager.add(
        PythonOutput(
            tool_call_id="call-1",
            execution_status=ResultStatus.ERROR,
            execution_count=1,
            stderr="Execution error: RuntimeError: boom",
        )
    )
    await bridge.flush()

    progress = next(
        cast(ToolCallProgress, update)
        for _, update in client.updates
        if isinstance(update, ToolCallProgress)
    )
    assert progress.title == "Python failed"
    assert progress.status == "failed"
    assert progress.content is not None
    assert len(progress.content) == 2
    source = _content_text(cast(ContentToolCallContent, progress.content[0]))
    output = _content_text(cast(ContentToolCallContent, progress.content[1]))
    assert source == "```python\nraise RuntimeError('boom')\n```"
    assert output == "```text\nExecution error: RuntimeError: boom\n```"
    for item in progress.model_dump(mode="json", by_alias=True)["content"]:
        assert item["content"]["_meta"]["aster"]["format"] == "fenced-code"
    await bridge.close()
    await agent.close()


async def test_bridge_retains_python_source_when_interrupted(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]

    agent.event_manager.add(
        ToolCallEvent(
            tool_call_id="call-1",
            name="execute_python",
            arguments={"code": "await asyncio.sleep(30)"},
        )
    )
    await bridge.fail_open_tools("User canceled")
    await bridge.flush()

    progress = next(
        cast(ToolCallProgress, update)
        for _, update in client.updates
        if isinstance(update, ToolCallProgress)
    )
    assert progress.title == "Python interrupted"
    assert progress.status == "failed"
    assert progress.content is not None
    assert len(progress.content) == 2
    source = _content_text(cast(ContentToolCallContent, progress.content[0]))
    output = _content_text(cast(ContentToolCallContent, progress.content[1]))
    assert source == "```python\nawait asyncio.sleep(30)\n```"
    assert output == "```text\nUser canceled\n```"
    await bridge.close()
    await agent.close()


@pytest.mark.parametrize("context_window", [None, 0, 20, 128_000])
async def test_bridge_reports_tokens_and_cost_independently_of_context_window(
    tmp_path, context_window
):
    llm = FakeLLMClient()
    cast(Any, llm)._context_window = context_window
    agent = CodingAgent(llm=llm, cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]

    agent.event_manager.add(AgentMessage(content="alive"))
    agent.event_manager.add(
        LLMResponse(usage=LLMUsage(input_tokens=40, output_tokens=10, cost_usd=0.25))
    )
    await bridge.flush()

    assert any(
        isinstance(update, AgentMessageChunk) and update.content.text == "alive"
        for _, update in client.updates
    )
    updates = [update for _, update in client.updates if isinstance(update, UsageUpdate)]
    try:
        assert len(updates) == 1
        usage = updates[0]
        assert usage.used == 40
        assert usage.size == (max(context_window, 40) if context_window else 0)
        assert usage.cost is not None
        assert usage.cost.amount == 0.25
        assert usage.cost.currency == "USD"
        payload = usage.model_dump(mode="json", by_alias=True, exclude_none=True)
        assert payload.get("_meta", {}).get("aster", {}).get("usageAvailable", True) is True
        assert UsageUpdate.model_validate(payload) == usage
    finally:
        await bridge.close()
        await agent.close()


@pytest.mark.parametrize("missing_usage", [False, True])
async def test_bridge_unknown_usage_or_cost_invalidates_cumulative_total(tmp_path, missing_usage):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]
    try:
        responses = [
            LLMUsage(input_tokens=40, cost_usd=0.25),
            None if missing_usage else LLMUsage(input_tokens=50),
            LLMUsage(input_tokens=60, cost_usd=0.5),
        ]
        for usage in responses:
            agent.event_manager.add(LLMResponse(usage=usage))
        await bridge.flush()

        updates = [update for _, update in client.updates if isinstance(update, UsageUpdate)]
        assert len(updates) == 3
        assert updates[0].cost is not None
        assert updates[0].cost.amount == 0.25
        assert updates[1].cost is None
        assert updates[2].cost is None
        assert updates[2].used == 60
        payload = updates[1].model_dump(mode="json", by_alias=True, exclude_none=True)
        if missing_usage:
            assert (updates[1].used, updates[1].size) == (0, 0)
            assert payload["_meta"]["aster"]["usageAvailable"] is False
        else:
            assert updates[1].used == 50
            assert payload.get("_meta", {}).get("aster", {}).get("usageAvailable", True) is True
        assert "cost" not in payload
    finally:
        await bridge.close()
        await agent.close()


async def test_bridge_accumulates_known_cost_including_explicit_zero(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]
    try:
        for cost in (0.0, 0.25, 0.5):
            agent.event_manager.add(LLMResponse(usage=LLMUsage(input_tokens=40, cost_usd=cost)))
        await bridge.flush()

        updates = [update for _, update in client.updates if isinstance(update, UsageUpdate)]
        assert [update.cost.amount if update.cost else None for update in updates] == [
            0.0,
            0.25,
            0.75,
        ]
    finally:
        await bridge.close()
        await agent.close()


async def test_bridge_does_not_present_local_token_estimates_as_reported_context(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]
    try:
        agent.event_manager.add(
            LLMResponse(usage=LLMUsage(input_tokens=40, input_tokens_reported=False, cost_usd=0.25))
        )
        await bridge.flush()

        updates = [update for _, update in client.updates if isinstance(update, UsageUpdate)]
        assert len(updates) == 1
        update = updates[0]
        assert (update.used, update.size) == (0, 0)
        assert update.field_meta == {"aster": {"usageAvailable": False}}
        assert update.cost is not None
        assert update.cost.amount == 0.25
    finally:
        await bridge.close()
        await agent.close()


async def test_bridge_emits_structured_file_edit(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]
    path = str(tmp_path / "example.py")

    agent.event_manager.add(
        FileEdit(
            path=path,
            operation="update",
            old_text="old\n",
            new_text="new\n",
            start_line=3,
            end_line=3,
            diff="unused when complete",
        )
    )
    await bridge.flush()

    update = next(
        cast(ToolCallStart, update)
        for _, update in client.updates
        if isinstance(update, ToolCallStart)
    )
    assert update.kind == "edit"
    assert update.status == "completed"
    assert update.locations is not None
    assert update.locations[0].path == path
    assert update.locations[0].line == 2
    assert update.content is not None
    content = cast(FileEditToolCallContent, update.content[0])
    assert content.path == path
    assert content.old_text == "old\n"
    assert content.new_text == "new\n"
    await bridge.close()
    await agent.close()


async def test_bridge_emits_terminal_lifecycle(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]

    agent.event_manager.add(
        TerminalCommandStarted(
            command_id="command-1",
            command="pytest -q",
            working_directory=str(tmp_path),
        )
    )
    agent.event_manager.add(TerminalCommandOutput(command_id="command-1", stdout="2 passed\n"))
    agent.event_manager.add(TerminalCommandFinished(command_id="command-1", exit_code=0))
    await bridge.flush()

    updates = [update for _, update in client.updates]
    assert [type(update) for update in updates] == [
        ToolCallStart,
        ToolCallProgress,
        ToolCallProgress,
    ]
    started = cast(ToolCallStart, updates[0])
    assert started.kind == "execute"
    assert started.title == "$ pytest -q"
    progress = cast(ToolCallProgress, updates[1])
    assert progress.content is not None
    content = cast(ContentToolCallContent, progress.content[0])
    assert content.content.text == "2 passed\n"
    finished = cast(ToolCallProgress, updates[2])
    assert finished.status == "completed"
    assert finished.raw_output == {
        "exit_code": 0,
        "timed_out": False,
        "output_truncated": False,
    }
    await bridge.close()
    await agent.close()


class _BlockingClient(_RecordingClient):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def session_update(self, session_id: str, update: object, **kwargs) -> None:
        self.started.set()
        await self.release.wait()
        await super().session_update(session_id, update, **kwargs)


async def test_cancelled_flush_does_not_stop_update_pump(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _BlockingClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]
    agent.event_manager.add(AgentMessage(content="First"))
    flush_task = asyncio.create_task(bridge.flush())
    await asyncio.wait_for(client.started.wait(), timeout=1)

    flush_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await flush_task
    client.release.set()
    agent.event_manager.add(AgentMessage(content="Second"))
    await asyncio.wait_for(bridge.flush(), timeout=1)

    messages = [
        update.content.text for _, update in client.updates if isinstance(update, AgentMessageChunk)
    ]
    assert messages == ["First", "Second"]
    await bridge.close()
    await agent.close()


async def test_a_failed_update_does_not_silence_the_session_for_good(tmp_path):
    """A transport failure must end its own turn, not every later one.

    The error was latched and never cleared, so after one failed write the
    bridge dropped every subsequent update and re-raised the stale exception
    on every future flush — the agent kept running turns, at full cost, that
    the client never saw.
    """
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)

    class _FlakyClient:
        def __init__(self) -> None:
            self.updates: list[object] = []
            self.fail_next = True

        async def session_update(self, session_id: str, update: object, **kwargs) -> None:
            if self.fail_next:
                self.fail_next = False
                raise ConnectionResetError("transport went away")
            self.updates.append(update)

    client = _FlakyClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]

    # Turn 1: the write fails, and the turn is told about it.
    agent.event_manager.add(AgentMessage(content="first turn"))
    with pytest.raises(ConnectionResetError):
        await bridge.flush()

    # Turn 2: the transport is healthy again, so updates must flow.
    agent.event_manager.add(AgentMessage(content="second turn"))
    await bridge.flush()

    assert any(
        isinstance(update, AgentMessageChunk) and update.content.text == "second turn"
        for update in client.updates
    )
    await bridge.close()


async def test_a_cancelled_command_reads_as_cancellation_not_a_crash(tmp_path):
    """The client must see the user's action, not a Python exception name."""
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]

    agent.event_manager.add(
        TerminalCommandStarted(
            command_id="cmd-1",
            command="sleep 30",
            working_directory=str(tmp_path),
        )
    )
    agent.event_manager.add(TerminalCommandFinished(command_id="cmd-1", cancelled=True))
    await bridge.flush()

    progress = [update for _, update in client.updates if isinstance(update, ToolCallProgress)]
    finished = progress[-1]
    rendered = str(finished)
    assert "Cancelled by user." in rendered
    assert "CancelledError" not in rendered
    # Status too: dropping `or event.cancelled` renders a cancelled command as a
    # green completed card while the reason text still reads correctly.
    assert finished.status == "failed"
    await bridge.close()


async def test_bare_expression_result_is_shown_not_reported_as_no_output(tmp_path):
    """A cell whose value is its result must not render as "Completed.".

    codeact puts the value in the model's own context as Out[n], so dropping it
    means the client is told there was no output while the agent reasons from
    one.
    """
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]

    agent.event_manager.add(
        ToolCallEvent(tool_call_id="t1", name="execute_python", arguments={"code": "1 + 1"})
    )
    agent.event_manager.add(
        PythonOutput(
            tool_call_id="t1",
            execution_status=ResultStatus.COMPLETE,
            execution_count=3,
            value=42,
        )
    )
    await bridge.flush()

    rendered = "".join(str(update) for _, update in client.updates)
    assert "Out[3]: 42" in rendered, rendered
    assert "Completed." not in rendered
    await bridge.close()


async def test_synthetic_text_replies_are_not_rendered_as_python_runs(tmp_path):
    """codeact turns a prose-only reply into a synthetic execute_python call.

    Nothing was executed, so surfacing it as a Python tool call shows the user
    a run that never happened, with their model's prose commented out inside.
    """
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]

    agent.event_manager.add(
        ToolCallEvent(
            tool_call_id="t2",
            name="execute_python",
            arguments={"code": "# I think the answer is 42"},
            metadata={"synthetic": True, "synthetic_type": "text_response"},
        )
    )
    await bridge.flush()

    assert not [u for _, u in client.updates if isinstance(u, ToolCallStart)]
    await bridge.close()


async def test_an_unfinished_tool_call_does_not_leak_for_the_session(tmp_path):
    """Closing the bridge must not leave a card spinning or state retained."""
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]

    agent.event_manager.add(
        ToolCallEvent(tool_call_id="t3", name="execute_python", arguments={"code": "boom"})
    )
    await bridge.flush()
    assert bridge._open_tools == {"t3"}

    await bridge.close()
    assert bridge._open_tools == set()
    assert bridge._python_source == {}
    # And the card was actually closed out for the client: clearing the private
    # state alone leaves it spinning, which is what the docstring forbids.
    closing = [
        update
        for _, update in client.updates
        if isinstance(update, ToolCallProgress) and update.tool_call_id == "t3"
    ]
    assert closing and closing[-1].status == "failed", client.updates
    assert closing[-1].title == "Unfinished"


async def test_a_cancelled_tool_card_is_titled_cancelled(tmp_path):
    """The collapsed card must say what happened.

    "Python interrupted" was used for every reason, so a user who cancelled saw
    a technical-sounding failure and had to expand the card to learn it was
    their own action.
    """
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]

    agent.event_manager.add(
        ToolCallEvent(tool_call_id="t9", name="execute_python", arguments={"code": "sleep(60)"})
    )
    await bridge.flush()
    await bridge.fail_open_tools("Cancelled by user.", title="Cancelled")
    await bridge.flush()

    progress = [u for _, u in client.updates if isinstance(u, ToolCallProgress)]
    assert progress[-1].title == "Cancelled"
    await bridge.close()


async def test_a_dead_pump_fails_flush_instead_of_hanging(tmp_path):
    """A BaseException from the transport must not strand every later flush.

    _pump caught only Exception, so a CancelledError raised by
    client.session_update — a disconnect during transport teardown — killed the
    pump task without resolving the queued flush marker. flush() waits on that
    marker and never observed the task, so it blocked forever, and close()
    flushes before awaiting the pump, hanging session teardown too.
    """
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)

    class _DyingClient:
        async def session_update(self, session_id: str, update: object, **kwargs) -> None:
            raise asyncio.CancelledError()

    bridge = ACPEventBridge(agent, _DyingClient(), "session-1")  # type: ignore[arg-type]
    agent.event_manager.add(AgentMessage(content="first"))

    # Must be the specific error flush() raises. pytest.raises(BaseException)
    # also accepts the TimeoutError from wait_for, so it passed against the
    # hanging bridge too — the exact regression it claims to cover.
    with pytest.raises(RuntimeError, match="ACP event bridge stopped"):
        await asyncio.wait_for(bridge.flush(), timeout=5)

    # And teardown must not hang either.
    await asyncio.wait_for(bridge.close(), timeout=5)
    await agent.close()


async def test_bridge_shows_partial_output_on_cancelled_card(tmp_path):
    """A cancelled cell's stdout/stderr (recorded by core as a CANCELLED
    PythonOutput) stays visible on the card fail_open_tools closes."""
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]

    agent.event_manager.add(
        ToolCallEvent(
            tool_call_id="call-1",
            name="execute_python",
            arguments={"code": "print('half'); await asyncio.sleep(30)"},
        )
    )
    agent.event_manager.add(
        PythonOutput(
            tool_call_id="call-1",
            execution_status=ResultStatus.CANCELLED,
            execution_count=1,
            stdout="half\n",
        )
    )
    await bridge.flush()
    await bridge.fail_open_tools("Stopped at your request.", title="Cancelled")
    await bridge.flush()

    progress = [
        cast(ToolCallProgress, update)
        for _, update in client.updates
        if isinstance(update, ToolCallProgress)
    ]
    assert len(progress) == 1  # the CANCELLED output did not close the card early
    assert progress[0].title == "Cancelled"
    assert progress[0].status == "failed"
    assert progress[0].content is not None
    output = _content_text(cast(ContentToolCallContent, progress[0].content[1]))
    assert output == "```text\nhalf\n\nStopped at your request.\n```"
    await bridge.close()
    await agent.close()


async def test_bridge_keeps_streamed_terminal_output_on_force_closed_card(tmp_path):
    """A terminal command still running when fail_open_tools closes its card keeps
    the output it streamed, with the reason below it."""
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]

    agent.event_manager.add(
        TerminalCommandStarted(
            command_id="cmd-1", command="make build", working_directory=str(tmp_path)
        )
    )
    agent.event_manager.add(TerminalCommandOutput(command_id="cmd-1", stdout="step 1 of 3\n"))
    await bridge.flush()
    await bridge.fail_open_tools("Stopped at your request.", title="Cancelled")
    await bridge.flush()

    closed = cast(ToolCallProgress, client.updates[-1][1])
    assert closed.title == "Cancelled"
    assert closed.status == "failed"
    assert closed.content is not None
    text = cast(ContentToolCallContent, closed.content[0]).content.text
    assert text == "step 1 of 3\n\nStopped at your request."
    await bridge.close()
    await agent.close()


async def test_prompt_cancel_during_flush_records_completion_after_answer_delivery(
    tmp_path, monkeypatch, capsys
):
    """Cancelling the waiter must not mark a still-buffered answer as flushed."""
    monkeypatch.setenv("ASTER_PROMPT_TIMING", "1")
    answer_entered = asyncio.Event()
    answer_release = asyncio.Event()

    class _BlockedClient(_RecordingClient):
        async def session_update(self, session_id: str, update: object, **kwargs) -> None:
            if isinstance(update, AgentMessageChunk):
                answer_entered.set()
                await answer_release.wait()
            await super().session_update(session_id, update, **kwargs)

    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _BlockedClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]

    async def prompt() -> None:
        async with bridge.prompt(perf_counter(), "inprocess"):
            agent.event_manager.add(AgentMessage(content="Ready answer"))

    def timings(stderr: str) -> list[dict[str, Any]]:
        return [
            json.loads(line.removeprefix("[aster-prompt] "))
            for line in stderr.splitlines()
            if line.startswith("[aster-prompt] ")
        ]

    task = asyncio.create_task(prompt())
    try:
        await asyncio.wait_for(answer_entered.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)

        before = capsys.readouterr().err
        stages = [record["stage"] for record in timings(before)]
        assert "first_status" in stages
        assert "first_answer" not in stages
        assert "complete" not in stages
        assert not any(isinstance(update, AgentMessageChunk) for _, update in client.updates)

        answer_release.set()
        await asyncio.wait_for(bridge.flush(), timeout=5)
        records = timings(before + capsys.readouterr().err)
        stages = [record["stage"] for record in records]
        assert stages.count("first_answer") == stages.count("complete") == 1
        assert stages.index("first_answer") < stages.index("complete")
        assert records[-1]["stage"] == "complete"
        assert records[-1]["outcome"] == "cancelled"
        assert [
            update.content.text
            for _, update in client.updates
            if isinstance(update, AgentMessageChunk)
        ] == ["Ready answer"]
    finally:
        answer_release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.wait_for(bridge.close(), timeout=5)
        await agent.close()


async def test_nested_generation_timings_bypass_code_cell_stderr_capture(monkeypatch, capsys):
    """A generation invoked inside a cell still reports timing to process stderr."""
    monkeypatch.setenv("ASTER_PROMPT_TIMING", "1")
    llm = FakeLLMClient.with_tool_call(
        "execute_python", {"code": "return_result(Done(explanation='complete'))"}
    )
    agent = InteractiveAgent(llm=llm)
    bridge = ACPEventBridge(agent, _RecordingClient(), "session-1")  # type: ignore[arg-type]
    try:
        async with asyncio.timeout(10), bridge.prompt(perf_counter(), "inprocess"):
            result = await agent.runtime.execute_code(
                "import sys\n"
                "await self.handle({'user_messages': ['Nested request']})\n"
                "print('cell stderr remains captured', file=sys.stderr)",
                validate=False,
            )

        captured = capsys.readouterr()
        assert result.error is None
        assert "cell stderr remains captured" in result.stderr
        assert "cell stderr remains captured" not in captured.err
        assert "[aster-prompt] " not in result.stdout + result.stderr
        assert "[aster-prompt] " not in captured.out
        records = [
            json.loads(line.removeprefix("[aster-prompt] "))
            for line in captured.err.splitlines()
            if line.startswith("[aster-prompt] ")
        ]
        stages = [record["stage"] for record in records]
        for stage in (
            "prefill_start",
            "prefill_end",
            "model_start",
            "model_end",
            "first_model_response",
            "complete",
        ):
            assert stages.count(stage) == 1
        assert stages[-1] == "complete"
        assert records[-1]["outcome"] == "end_turn"
    finally:
        await asyncio.wait_for(bridge.close(), timeout=5)
        await agent.aclose()
