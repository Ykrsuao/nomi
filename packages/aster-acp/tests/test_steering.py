# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live steering retains the active interpreter and the original turn identity."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import pytest
from acp import RequestError, text_block
from acp.schema import AgentMessageChunk, UserMessageChunk
from aster_acp._session_replay import transcript_updates
from aster_acp.server import CodingACPAdapter
from aster_cli.coding.reply_strategy import CodingReplyStrategy
from aster_cli.sessions import SessionSteering, SessionStore

from aster.config import CodeActConfig
from aster.context_blocks.events import ToolCallEvent
from aster.events import ExecutionResult, Notification
from aster.interactive import AgentMessage, Done
from aster.runtime.actor import ActorRuntime
from aster.skill_registry import SkillRegistry
from aster.strategies.codeact import _ReturnResultSignal
from aster.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
from aster.unifiedllm.streaming import emit_stream


class _GateEvent(asyncio.Event):
    __nosnapshot__ = True


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTER_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("ASTER_PROJECT_DIR", str(tmp_path / "project"))
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(SkillRegistry, "discovered", lambda self: [])
    monkeypatch.setattr("aster_acp.server.load_coding_skills_dirs", lambda root: [])
    for key in ("ASTER_SETTINGS", "ASTER_RICH_URL", "ASTER_PROMPT_TIMING"):
        monkeypatch.delenv(key, raising=False)


class Client:
    def __init__(self):
        self.updates = []

    async def session_update(self, session_id, update, **kwargs):
        self.updates.append(update)


def response(*calls):
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=list(calls),
        finish_reason="tool_calls",
        reasoning=None,
        usage=None,
    )


def code(call_id, source):
    return ToolCall(id=call_id, name="execute_python", arguments=json.dumps({"code": source}))


def done(call_id, message="Complete"):
    return ToolCall(
        id=call_id,
        name="return_result",
        arguments=json.dumps({"result": {"message": message, "explanation": "finished"}}),
    )


@asynccontextmanager
async def harness(tmp_path, *responses, streaming=False):
    llm = FakeLLMClient(responses, strict_exhaustion=True)
    adapter = CodingACPAdapter(lambda: llm)
    client = Client()
    adapter.on_connect(client)
    await adapter.initialize(1, aster={"messageStreaming": streaming})
    session_id = (await adapter.new_session(cwd=str(tmp_path), mcp_servers=[])).session_id
    session = (await adapter._sessions.get(session_id)).value
    session.dispatcher._strategy = CodingReplyStrategy(
        config=CodeActConfig(execution_backend="inprocess", prefill=None, max_retries=1)
    )
    try:
        async with asyncio.timeout(15):
            yield adapter, session, client, llm
    finally:
        await session.dispatcher.cancel()
        session.cancel_complete.set()
        await adapter.close()


def prompt(adapter, session, turn="turn-1"):
    return asyncio.create_task(
        adapter.prompt(session.handle.id, [text_block("Do the work")], aster={"turnId": turn})
    )


async def steer(adapter, session, *, turn="turn-1", message="input-1", text="Use the new plan"):
    return await adapter.ext_method(
        "aster/session/steer",
        {"sessionId": session.handle.id, "turnId": turn, "messageId": message, "text": text},
    )


def states(session):
    return [
        record.data["state"]
        for record in session.handle.transcript()
        if record.event_type == "SessionSteering"
    ]


def gate_model(monkeypatch, llm, call_number=1):
    entered, release = asyncio.Event(), asyncio.Event()
    original = llm.acall

    async def acall(*args, **kwargs):
        result = await original(*args, **kwargs)
        if llm.call_count == call_number:
            entered.set()
            await release.wait()
        return result

    monkeypatch.setattr(llm, "acall", acall)
    return entered, release


@pytest.mark.parametrize("inline_final", [False, True])
async def test_tool_boundary_keeps_locals_and_pairs_skipped_native_calls(tmp_path, inline_final):
    first = (
        "state = ['kept']\n"
        "self._steering_entered.set()\n"
        "await self._steering_release.wait()\n"
        "state.append('finished')\n"
    )
    if inline_final:
        first += "return_result(Done(message=' '.join(state), explanation='old result'))"
    old = response(
        code("running", first),
        code("stale", "state.append('must not run')"),
        done("old-final", "Old answer"),
    )
    new = response(
        code(
            "revised",
            "assert state == ['kept', 'finished']\n"
            "return_result(Done(message=' '.join(state), explanation='revised'))",
        )
    )
    async with harness(tmp_path, old, new) as (adapter, session, client, llm):
        session.agent._steering_entered = _GateEvent()
        session.agent._steering_release = _GateEvent()
        turn = prompt(adapter, session)
        await session.agent._steering_entered.wait()
        active_task = session.dispatcher._active_task
        ack = await steer(adapter, session)
        assert ack["state"] == "accepted"
        assert states(session) == ["accepted"]
        assert session.dispatcher._active_task is active_task
        assert not any(
            isinstance(event, Notification) for event in session.agent.event_manager.values()
        )
        session.agent._steering_release.set()
        assert (await turn).stop_reason == "end_turn"
        assert states(session) == ["accepted", "applied"]
        assert llm.call_count == 2
        assert "Use the new plan" in str(llm.calls[1].messages)
        assert "Use the new plan" not in str(llm.calls[0].messages)
        calls = {
            event.tool_call_id: event
            for event in session.agent.event_manager.values()
            if isinstance(event, ToolCallEvent)
        }
        for call_id in ("stale", "old-final"):
            assert calls[call_id].result is not None
            assert calls[call_id].llm_response_id == old.id
            assert calls[call_id].metadata["not_executed"] is True
        assert [
            event.content
            for event in session.agent.event_manager.values()
            if isinstance(event, AgentMessage)
        ] == ["kept finished"]
        assert session.handle.info.turn_count == 1
        assert session.handle.checkpoints() == [(0, "Do the work")]


async def test_inflight_final_replans_and_idempotence_survives_completion(tmp_path, monkeypatch):
    async with harness(tmp_path, response(done("old")), response(done("new", "Revised"))) as (
        adapter,
        session,
        client,
        llm,
    ):
        entered, release = gate_model(monkeypatch, llm)
        turn = prompt(adapter, session)
        await entered.wait()
        first = await steer(adapter, session)
        assert await steer(adapter, session) == first
        assert states(session) == ["accepted"]
        with pytest.raises(RequestError):
            await steer(adapter, session, text="Different payload")
        with pytest.raises(RequestError):
            await steer(adapter, session, turn="previous", message="other")
        release.set()
        await turn
        assert (await steer(adapter, session))["state"] == "applied"
        with pytest.raises(RequestError):
            await steer(adapter, session, message="too-late")
        with pytest.raises(RequestError):
            await adapter.prompt(
                session.handle.id, [text_block("reused")], aster={"turnId": "turn-1"}
            )
        assert session.handle.info.turn_count == 1
        assert [
            event.content
            for event in session.agent.event_manager.values()
            if isinstance(event, AgentMessage)
        ] == ["Revised"]


async def test_cancelled_unapplied_input_does_not_leak_into_next_prompt(tmp_path, monkeypatch):
    async with harness(tmp_path, response(done("cancelled")), response(done("next"))) as (
        adapter,
        session,
        client,
        llm,
    ):
        entered, release = gate_model(monkeypatch, llm)
        turn = prompt(adapter, session)
        await entered.wait()
        await steer(adapter, session, text="Never apply this cancelled instruction")
        await adapter.cancel(session.handle.id)
        assert (await turn).stop_reason == "cancelled"
        assert states(session) == ["accepted", "cancelled"]
        release.set()
        await prompt(adapter, session, "turn-2")
        assert "Never apply this cancelled instruction" not in str(llm.calls[1].messages)
        assert (await steer(adapter, session, text="Never apply this cancelled instruction"))[
            "state"
        ] == "cancelled"
        with pytest.raises(RequestError):
            await steer(adapter, session, message="stale-id")
        assert session.handle.info.turn_count == 2


async def test_input_received_during_message_build_is_in_that_request(tmp_path, monkeypatch):
    async with harness(tmp_path, response(done("finish"))) as (adapter, session, client, llm):
        entered, release = asyncio.Event(), asyncio.Event()
        original = ActorRuntime._build_messages

        async def build(runtime, *args, **kwargs):
            messages = await original(runtime, *args, **kwargs)
            if runtime.agent is session.agent and not entered.is_set():
                entered.set()
                await release.wait()
            return messages

        monkeypatch.setattr(ActorRuntime, "_build_messages", build)
        turn = prompt(adapter, session)
        await entered.wait()
        await steer(adapter, session)
        assert states(session) == ["accepted"]
        release.set()
        await turn
        assert llm.call_count == 1
        matching = [m for m in llm.calls[0].messages if "Use the new plan" in str(m)]
        assert len(matching) == 1
        assert matching[0]["role"] == "user"
        assert states(session) == ["accepted", "applied"]


async def test_steering_wakes_background_wait_without_new_user_turn(tmp_path, monkeypatch):
    waiting = ToolCall(
        id="wait",
        name="return_result",
        arguments=json.dumps({"result": {"on": ["system_messages"], "explanation": "waiting"}}),
    )
    async with harness(tmp_path, response(waiting), response(done("finish"))) as (
        adapter,
        session,
        client,
        llm,
    ):
        entered = asyncio.Event()
        original = session.agent.queue_manager.race

        async def race():
            if llm.call_count == 1:
                entered.set()
            return await original()

        monkeypatch.setattr(session.agent.queue_manager, "race", race)
        turn = prompt(adapter, session)
        await entered.wait()
        await steer(adapter, session)
        await turn
        assert states(session) == ["accepted", "applied"]
        assert llm.call_count == 2
        assert session.handle.info.turn_count == 1


def test_replay_coalesces_states_at_acceptance_without_adding_user_turns(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    with store.create() as handle:
        handle.record_user_message("Original", turn_id="turn-1")
        first = SessionSteering(message_id="s1", turn_id="turn-1", text="First", state="accepted")
        second = SessionSteering(message_id="s2", turn_id="turn-1", text="Second", state="accepted")
        handle.record_steering(first)
        handle.events.add(AgentMessage(content="Progress"))
        handle.record_steering(second)
        handle.record_steering(first.model_copy(update={"state": "applied"}))
        updates = transcript_updates(handle.transcript())
        assert isinstance(updates[0], UserMessageChunk)
        assert updates[1].field_meta["aster"]["steering"]["state"] == "applied"
        assert isinstance(updates[2], AgentMessageChunk)
        assert updates[3].field_meta["aster"]["steering"]["state"] == "interrupted"
        assert len(updates) == 4
        assert handle.info.turn_count == 1
        assert [turn.content for turn in handle.turns() if turn.role == "user"] == ["Original"]


@pytest.mark.parametrize("explicit_return", [False, True])
async def test_completion_is_sealed_before_async_worker_teardown(
    tmp_path, monkeypatch, explicit_return
):
    final = (
        code("finish", "answer = str(42)\nreturn Done(message=answer, explanation='complete')")
        if explicit_return
        else done("finish")
    )
    async with harness(tmp_path, response(final)) as (adapter, session, client, llm):
        entered, release = asyncio.Event(), asyncio.Event()
        strategy = session.dispatcher._strategy
        original = strategy._close_sandbox

        async def close(worker_session):
            entered.set()
            await release.wait()
            await original(worker_session)

        monkeypatch.setattr(strategy, "_close_sandbox", close)
        turn = prompt(adapter, session)
        await entered.wait()
        try:
            with pytest.raises(RequestError):
                await steer(adapter, session)
            assert states(session) == []
        finally:
            release.set()
        await turn


@pytest.mark.parametrize("explicit_return", [False, True])
async def test_sandbox_worker_is_retained_across_steered_completion(
    tmp_path, monkeypatch, explicit_return
):
    async with harness(tmp_path, response(code("old", "pass")), response(code("new", "pass"))) as (
        adapter,
        session,
        client,
        llm,
    ):
        entered, release = asyncio.Event(), asyncio.Event()

        class Executor:
            calls = 0
            closed = False

            async def run_cell(self, source, *, execution_count):
                self.calls += 1
                if self.calls == 1:
                    self.value = "worker retained"
                    entered.set()
                    await release.wait()
                assert self.value == "worker retained"
                value = Done(message=self.value, explanation="computed")
                if explicit_return:
                    return ExecutionResult(returned_value=value, explicit_return=True)
                return ExecutionResult(signal=_ReturnResultSignal(result={"result": value}))

            async def aclose(self):
                self.closed = True

        created = []
        strategy = CodingReplyStrategy(
            config=CodeActConfig(execution_backend="sandbox", prefill=None)
        )

        def create(*args):
            executor = Executor()
            created.append(executor)
            return executor

        monkeypatch.setattr(strategy, "_create_sandbox_executor", create)
        session.dispatcher._strategy = strategy
        turn = prompt(adapter, session)
        await entered.wait()
        await steer(adapter, session)
        release.set()
        await turn
        assert len(created) == 1
        assert created[0].calls == 2
        assert created[0].closed
        assert states(session) == ["accepted", "applied"]
        assert [
            event.content
            for event in session.agent.event_manager.values()
            if isinstance(event, AgentMessage)
        ] == ["worker retained"]


async def test_nested_generation_cannot_consume_root_steering(tmp_path, monkeypatch):
    source = (
        "@strategy(CodeActStrategy())\n"
        "async def child() -> str:\n"
        "    '''Produce a short result.'''\n"
        "    ...\n"
        "self._steering_entered.set()\n"
        "await self._steering_release.wait()\n"
        "nested = await child()\n"
    )
    child = ToolCall(id="child", name="return_result", arguments=json.dumps({"result": "nested"}))
    async with harness(
        tmp_path, response(code("root", source)), response(child), response(done("final"))
    ) as (adapter, session, client, llm):
        session.agent._steering_entered = _GateEvent()
        session.agent._steering_release = _GateEvent()
        nested_entered, nested_release = gate_model(monkeypatch, llm, call_number=2)
        turn = prompt(adapter, session)
        await session.agent._steering_entered.wait()
        await steer(adapter, session)
        session.agent._steering_release.set()
        await nested_entered.wait()
        assert states(session) == ["accepted"]
        nested_release.set()
        await turn
        assert states(session) == ["accepted", "applied"]
        assert "Use the new plan" in str(llm.calls[2].messages)


async def test_superseded_stream_preview_is_interrupted_before_revised_commit(
    tmp_path, monkeypatch
):
    old, new = response(done("old", "Old preview")), response(done("new", "Revised answer"))
    async with harness(tmp_path, old, new, streaming=True) as (adapter, session, client, llm):
        entered, release = asyncio.Event(), asyncio.Event()
        original = llm.acall

        async def acall(*args, **kwargs):
            result = await original(*args, **kwargs)
            emit_stream(
                "chat",
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {
                                            "name": "return_result",
                                            "arguments": result.tool_calls[0].arguments,
                                        },
                                    }
                                ]
                            },
                        }
                    ]
                },
            )
            if llm.call_count == 1:
                entered.set()
                await release.wait()
            return result

        monkeypatch.setattr(llm, "acall", acall)
        turn = prompt(adapter, session)
        await entered.wait()
        await session.bridge.flush()
        old_preview = next(
            update
            for update in client.updates
            if isinstance(update, AgentMessageChunk) and update.content.text == "Old preview"
        )
        message_id = old_preview.field_meta["aster"]["messageId"]
        await steer(adapter, session)
        release.set()
        await turn
        messages = [update for update in client.updates if isinstance(update, AgentMessageChunk)]
        old_states = [
            update.field_meta["aster"]["messageState"]
            for update in messages
            if update.field_meta["aster"]["messageId"] == message_id
        ]
        assert old_states == ["delta", "interrupted"]
        commits = [
            update.content.text
            for update in messages
            if update.field_meta["aster"]["messageState"] == "complete"
        ]
        assert commits == ["Revised answer"]


async def test_recovered_accepted_input_is_interrupted_and_replayed_once(tmp_path):
    async with harness(tmp_path, response(done("finish"))) as (adapter, session, client, llm):
        session_id = session.handle.id
        session.handle.record_user_message("Interrupted request", turn_id="lost-turn")
        session.handle.record_steering(
            SessionSteering(
                message_id="lost", turn_id="lost-turn", text="Lost input", state="accepted"
            )
        )
        session.agent.event_manager.add(
            Notification(
                source="New message from the user while you were working.",
                description="Lost input",
                metadata={"steering_message_id": "lost", "turn_id": "lost-turn"},
            )
        )
        await adapter.close_session(session_id)
        client.updates.clear()
        await adapter.load_session(cwd=str(tmp_path), session_id=session_id)
        restored = (await adapter._sessions.get(session_id)).value
        assert states(restored) == ["accepted", "interrupted"]
        replayed = [
            update.field_meta["aster"]["steering"]
            for update in client.updates
            if (getattr(update, "field_meta", None) or {}).get("aster", {}).get("steering")
        ]
        assert len(replayed) == 1
        assert replayed[0]["state"] == "interrupted"
        assert (
            await steer(adapter, restored, turn="lost-turn", message="lost", text="Lost input")
        )["state"] == "interrupted"
        assert not restored.steering.pending
        await prompt(adapter, restored, "new-turn")
        assert "Lost input" not in str(llm.calls[0].messages)
