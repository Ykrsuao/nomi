# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable tool cards survive restart without executing historical commands."""

import json
from datetime import UTC, datetime, timedelta

import pytest
from acp import text_block
from acp.schema import AgentMessageChunk, ToolCallStart, UserMessageChunk
from aster_acp._session_replay import record_activity, transcript_updates
from aster_acp.event_bridge import ACPEventBridge
from aster_acp.server import CodingACPAdapter
from aster_cli.coding import (
    CodingAgent,
    FileEdit,
    TerminalCommandFinished,
    TerminalCommandOutput,
    TerminalCommandStarted,
)
from aster_cli.sessions import SessionStore, SessionTranscriptEvent

from aster.context_blocks.events import ResultStatus, ToolCallEvent
from aster.events import PythonOutput
from aster.interactive import AgentMessage
from aster.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


class RecordingClient:
    def __init__(self):
        self.updates = []

    async def session_update(self, session_id, update, **kwargs):
        self.updates.append(update)


def payload(update):
    return update.model_dump(mode="json", by_alias=True, exclude_none=True)


def text(card):
    return "\n".join(
        item["content"]["text"]
        for item in payload(card).get("content", [])
        if item["type"] == "content"
    )


def stamp(seconds):
    return datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds)


def test_transcript_preserves_order_outputs_diffs_timing_and_message_identity(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    with store.create() as handle:
        unsubscribe = record_activity(handle.events, handle)
        handle.record_user_message("change a file")
        handle.events.add(
            ToolCallEvent(
                tool_call_id="py",
                name="execute_python",
                arguments={"code": "do_work()"},
                timestamp=stamp(0),
            )
        )
        handle.events.add(
            TerminalCommandStarted(
                command_id="shell",
                command="check",
                working_directory=str(tmp_path),
                stdin="DO_NOT_PERSIST_STDIN",
                timestamp=stamp(1),
            )
        )
        handle.events.add(
            TerminalCommandOutput(command_id="shell", stdout="first\n", timestamp=stamp(2))
        )
        handle.events.add(
            FileEdit(
                path=str(tmp_path / "hello.txt"),
                operation="update",
                old_text="old\n",
                new_text="new\n",
                timestamp=stamp(3),
            )
        )
        handle.events.add(
            AgentMessage(
                content="progress",
                metadata={"stream_message_id": "stream-message"},
                timestamp=stamp(4),
            )
        )
        handle.events.add(
            TerminalCommandOutput(command_id="shell", stderr="second\n", timestamp=stamp(4))
        )
        handle.events.add(
            TerminalCommandFinished(command_id="shell", exit_code=0, timestamp=stamp(5))
        )
        handle.events.add(
            PythonOutput(
                tool_call_id="py",
                execution_status=ResultStatus.COMPLETE,
                execution_count=1,
                stdout="printed",
                value=42,
                timestamp=stamp(6),
            )
        )
        handle.events.add(AgentMessage(content="done", timestamp=stamp(7)))
        records = handle.transcript()
        assert "DO_NOT_PERSIST_STDIN" not in repr(records)
        assert [record.event_type for record in records] == [
            "SessionUserMessage",
            "ToolCallEvent",
            "TerminalCommandStarted",
            "TerminalCommandOutput",
            "FileEdit",
            "AgentMessage",
            "TerminalCommandOutput",
            "TerminalCommandFinished",
            "PythonOutput",
            "AgentMessage",
        ]
        assert [turn.content for turn in handle.turns()] == ["change a file", "progress", "done"]
        unsubscribe()
        session_id = handle.id

    updates = transcript_updates(store.load_transcript(session_id))
    assert [type(item) for item in updates] == [
        UserMessageChunk,
        ToolCallStart,
        ToolCallStart,
        ToolCallStart,
        AgentMessageChunk,
        AgentMessageChunk,
    ]
    python, shell, edit = updates[1:4]
    assert python.title == "Ran Python" and python.status == "completed"
    assert "printed" in text(python) and "Out[1]: 42" in text(python)
    assert payload(python)["_meta"]["aster"]["elapsedSeconds"] == 6.0
    assert text(shell) == "first\nsecond\n"
    assert payload(shell)["_meta"]["aster"]["elapsedSeconds"] == 4.0
    assert payload(edit)["content"][0] == {
        "type": "diff",
        "path": str(tmp_path / "hello.txt"),
        "newText": "new\n",
        "oldText": "old\n",
    }
    assert "_meta" not in payload(edit)
    assert payload(updates[4])["_meta"]["aster"] == {
        "messageId": "stream-message",
        "messageState": "complete",
        "replace": True,
    }
    assert not (tmp_path / "hello.txt").exists(), "Replay must not write historical edits"
    assert [payload(item) for item in transcript_updates(store.load_transcript(session_id))] == [
        payload(item) for item in updates
    ]


@pytest.mark.parametrize("status", [ResultStatus.ERROR, ResultStatus.CANCELLED])
def test_failed_and_cancelled_python_preserve_partial_output(tmp_path, status):
    store = SessionStore(tmp_path)
    with store.create() as handle:
        handle.events.add(
            ToolCallEvent(
                tool_call_id="py",
                name="python_cell",
                arguments={"code": "work()"},
                timestamp=stamp(0),
            )
        )
        handle.events.add(
            PythonOutput(
                tool_call_id="py",
                execution_status=status,
                execution_count=1,
                stdout="partial",
                error="error" if status is ResultStatus.ERROR else "",
                timestamp=stamp(1.23),
            )
        )
        (card,) = transcript_updates(handle.transcript())
    assert card.status == "failed"
    assert card.title == ("Python failed" if status is ResultStatus.ERROR else "Cancelled")
    assert "partial" in text(card)
    assert ("error" if status is ResultStatus.ERROR else "Cancelled by user.") in text(card)
    assert payload(card)["_meta"]["aster"]["elapsedSeconds"] == pytest.approx(1.23)
    for item in payload(card)["content"]:
        assert item["content"]["_meta"]["aster"]["format"] == "fenced-code"


def test_unfinished_tools_never_look_running_and_keep_partial_output(tmp_path):
    store = SessionStore(tmp_path)
    with store.create() as handle:
        record_activity(handle.events, handle)
        handle.events.add(
            ToolCallEvent(tool_call_id="py", name="execute_python", arguments={"code": "work()"})
        )
        handle.events.add(
            TerminalCommandStarted(
                command_id="shell", command="work", working_directory=str(tmp_path)
            )
        )
        handle.events.add(TerminalCommandOutput(command_id="shell", stdout="partial"))
        updates = transcript_updates(handle.transcript())
    assert len(updates) == 2
    assert all(update.status == "failed" for update in updates)
    assert all("_meta" not in payload(update) for update in updates)
    assert "partial" in text(updates[1])
    assert "No completion was recorded" in text(updates[1])


@pytest.mark.parametrize(
    "finished",
    [
        {"cancelled": True},
        {"timed_out": True},
        {"error": "Could not launch"},
        {"exit_code": 7},
    ],
)
def test_terminal_failures_restore_partial_output_and_recorded_status(tmp_path, finished):
    store = SessionStore(tmp_path)
    with store.create() as handle:
        record_activity(handle.events, handle)
        handle.events.add(
            TerminalCommandStarted(
                command_id="shell",
                command="work",
                working_directory=str(tmp_path),
                timestamp=stamp(0),
            )
        )
        handle.events.add(
            TerminalCommandOutput(command_id="shell", stdout="partial", timestamp=stamp(0.5))
        )
        handle.events.add(
            TerminalCommandFinished(command_id="shell", timestamp=stamp(1), **finished)
        )
        (card,) = transcript_updates(handle.transcript())
    assert card.status == "failed"
    assert "partial" in text(card)
    assert payload(card)["_meta"]["aster"]["elapsedSeconds"] == 1
    if finished.get("cancelled"):
        assert "Cancelled by user." in text(card)
    if finished.get("error"):
        assert finished["error"] in text(card)


@pytest.mark.parametrize(
    ("finished", "chunks", "status", "expected_text"),
    [
        ({}, [], "completed", "Completed."),
        (
            {"exit_code": 0, "output_truncated": True},
            [("one", "two\n"), ("three\n", "")],
            "completed",
            "one\ntwo\nthree\n",
        ),
        ({"exit_code": 3}, [("failed", "")], "failed", "failed"),
        (
            {"cancelled": True, "error": "transport cancelled"},
            [("partial", "stderr")],
            "failed",
            "partial\nstderr\nCancelled by user.",
        ),
        ({"timed_out": True}, [("partial\n", "")], "failed", "partial\n"),
        ({"error": "Could not launch"}, [], "failed", "Could not launch"),
        ({"error": "failed"}, [("partial\n", "")], "failed", "partial\nfailed"),
    ],
)
async def test_live_and_reopened_terminal_results_have_identical_content(
    tmp_path, finished, chunks, status, expected_text
):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = RecordingClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]
    store = SessionStore(tmp_path / "sessions")
    try:
        with store.create() as handle:
            unsubscribe = record_activity(agent.event_manager, handle)
            try:
                agent.event_manager.add(
                    TerminalCommandStarted(
                        command_id="shell", command="work", working_directory=str(tmp_path)
                    )
                )
                for stdout, stderr in chunks:
                    agent.event_manager.add(
                        TerminalCommandOutput(command_id="shell", stdout=stdout, stderr=stderr)
                    )
                event = TerminalCommandFinished(command_id="shell", **finished)
                agent.event_manager.add(event)
                await bridge.flush()
                session_id = handle.id
            finally:
                unsubscribe()

        (reopened,) = transcript_updates(store.load_transcript(session_id))
        live = client.updates[-1]
        assert live.status == reopened.status == status
        assert text(live) == text(reopened) == expected_text
        assert payload(live)["content"] == payload(reopened)["content"]
        assert (
            live.raw_output
            == reopened.raw_output
            == {
                "exit_code": event.exit_code,
                "timed_out": event.timed_out,
                "output_truncated": event.output_truncated,
            }
        )
    finally:
        await bridge.close()
        await agent.close()


def test_unsubscribing_recorder_stops_capture_and_snapshots_do_not_enter_model_context(tmp_path):
    from aster_cli.sessions import SessionActivity

    from aster.context_blocks.roles import Role

    store = SessionStore(tmp_path)
    with store.create() as handle:
        unsubscribe = record_activity(handle.events, handle)
        handle.events.add(FileEdit(path="first.txt", operation="create", new_text="first"))
        unsubscribe()
        handle.events.add(FileEdit(path="second.txt", operation="create", new_text="second"))
        records = handle.transcript()
        assert len(records) == 1 and records[0].data["path"] == "first.txt"
    assert SessionActivity(activity={}).event_role is Role.METADATA


async def test_streaming_negotiation_only_accepts_explicit_boolean():
    adapter = CodingACPAdapter(FakeLLMClient)
    try:
        for options, enabled in [
            (None, False),
            ({}, False),
            ({"messageStreaming": "true"}, False),
            ({"messageStreaming": True}, True),
        ]:
            await adapter.initialize(1, aster=options)
            assert adapter._stream_messages is enabled
    finally:
        await adapter.close()


def test_legacy_messages_hidden_tools_corrupt_rows_and_missing_timestamps():
    start = ToolCallEvent(
        tool_call_id="py", name="execute_python", arguments={"code": "pass"}
    ).model_dump(mode="json")
    output = PythonOutput(
        tool_call_id="py", execution_count=1, execution_status=ResultStatus.COMPLETE
    ).model_dump(mode="json")
    start.pop("timestamp")
    for metadata in ({"prefill": True}, {"synthetic": True}):
        hidden = {**start, "tool_call_id": "hidden", "metadata": metadata}
        assert transcript_updates([SessionTranscriptEvent("ToolCallEvent", hidden)]) == []
    records = [
        SessionTranscriptEvent("TUIUserInput", {"text": "legacy"}),
        SessionTranscriptEvent("ToolCallEvent", {"broken": "tool"}),
        SessionTranscriptEvent("ToolCallEvent", start),
        SessionTranscriptEvent("PythonOutput", output),
        SessionTranscriptEvent("TUIAgentMessage", {"content": "reply"}),
    ]
    updates = transcript_updates(records)
    assert [type(item) for item in updates] == [UserMessageChunk, ToolCallStart, AgentMessageChunk]
    assert updates[1].status == "completed"
    assert "_meta" not in payload(updates[1])
    assert updates[0].content.text == "legacy\n"


async def test_real_session_replays_saved_file_edit_without_reexecuting(tmp_path, monkeypatch):
    config = tmp_path / "config"
    config.mkdir()
    monkeypatch.setenv("ASTER_USER_DIR", str(config))
    monkeypatch.setattr("aster_acp.server.load_coding_skills_dirs", lambda root: [])
    first = CodingACPAdapter(
        lambda: FakeLLMClient(
            [
                LLMResponse(
                    raw_response=None,
                    content="",
                    tool_calls=[
                        ToolCall(id=f"replay-{index}", name=name, arguments=json.dumps(arguments))
                    ],
                    finish_reason="tool_calls",
                )
                for index, (name, arguments) in enumerate(
                    [
                        (
                            "execute_python",
                            {"code": "await self.shell.write_file('result.txt', 'saved file\\n')"},
                        ),
                        (
                            "return_result",
                            {"result": {"message": "finished", "explanation": "done"}},
                        ),
                    ]
                )
            ],
            strict_exhaustion=True,
        )
    )
    first.on_connect(RecordingClient())
    try:
        created = await first.new_session(str(tmp_path))
        await first.prompt(created.session_id, [text_block("write file")])
    finally:
        await first.close()
    assert (tmp_path / "result.txt").read_text() == "saved file\n"
    (tmp_path / "result.txt").write_text("new user content\n")
    # The loaded agent must not call the model or execute its old tool program.
    llm = FakeLLMClient()
    replay = CodingACPAdapter(lambda: llm)
    client = RecordingClient()
    replay.on_connect(client)
    try:
        await replay.load_session(str(tmp_path), created.session_id)
        cards = [update for update in client.updates if isinstance(update, ToolCallStart)]
        assert any(card.title == "Ran Python" for card in cards)
        edit = next(card for card in cards if card.title == "Created result.txt")
        assert payload(edit)["content"][0]["newText"] == "saved file\n"
        assert (tmp_path / "result.txt").read_text() == "new user content\n"
        assert not llm.calls
    finally:
        await replay.close()
