# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session protocol exercises approvals, live settings, tasks and input snapshots."""

from __future__ import annotations

import asyncio
import base64
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from acp import RequestError, text_block
from acp.schema import AgentPlanUpdate, RequestPermissionResponse
from aster_acp.server import CodingACPAdapter
from aster_cli.coding.reply_strategy import CodingReplyStrategy

from aster.config import CodeActConfig
from aster.events import Task
from aster.skill_registry import SkillRegistry
from aster.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l1QAAAAASUVORK5CYII="
)


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTER_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("ASTER_PROJECT_DIR", str(tmp_path / "project"))
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.delenv("ASTER_SETTINGS", raising=False)
    monkeypatch.setattr(SkillRegistry, "discovered", lambda self: [])
    monkeypatch.setattr("aster_acp.server.load_coding_skills_dirs", lambda root: [])


class Client:
    def __init__(self):
        self.updates = []
        self.permissions = []
        self.choice = "allow_once"
        self.entered = asyncio.Event()
        self.release = None

    async def session_update(self, session_id, update, **kwargs):
        self.updates.append(update)

    async def request_permission(self, session_id, tool_call, options, **kwargs):
        self.permissions.append(tool_call)
        self.entered.set()
        if self.release is not None:
            await self.release.wait()
        return RequestPermissionResponse.model_validate(
            {"outcome": {"outcome": "selected", "optionId": self.choice}}
        )


def reply(identifier, *, source=None, message="Complete"):
    return LLMResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[
            ToolCall(
                id=identifier,
                name="execute_python" if source is not None else "return_result",
                arguments=json.dumps(
                    {"code": source}
                    if source is not None
                    else {"result": {"message": message, "explanation": "complete"}}
                ),
            )
        ],
    )


@asynccontextmanager
async def session_for(tmp_path, *responses, enabled=True, images=False):
    llm = FakeLLMClient(responses, strict_exhaustion=True)
    if images:
        llm.media_inputs = ("image",)
    adapter = CodingACPAdapter(lambda: llm)
    client = Client()
    adapter.on_connect(client)
    handshake = await adapter.initialize(
        1, aster={"runtimeControls": enabled, "structuredProgress": enabled}
    )
    new = await adapter.new_session(cwd=str(tmp_path), mcp_servers=[])
    session = (await adapter._sessions.get(new.session_id)).value
    session.dispatcher._strategy = CodingReplyStrategy(
        config=CodeActConfig(prefill=None, max_retries=3)
    )
    try:
        async with asyncio.timeout(15):
            yield adapter, session, client, llm, handshake, new
    finally:
        for session_id in await adapter._sessions.ids():
            await adapter.cancel(session_id)
        await adapter.close()


async def extension(adapter, session, method, **params):
    return await adapter.ext_method(
        f"aster/session/{method}", {"sessionId": session.handle.id, **params}
    )


async def test_negotiation_leaves_existing_clients_execution_unchanged(tmp_path):
    target = tmp_path / "legacy.txt"
    source = f"open({str(target)!r}, 'w').write('legacy')"
    async with session_for(
        tmp_path, reply("write", source=source), reply("done"), enabled=False
    ) as (adapter, session, client, llm, handshake, new):
        assert handshake.field_meta["aster"]["runtimeControls"] is True
        assert handshake.agent_capabilities.prompt_capabilities.image is True
        assert handshake.agent_capabilities.prompt_capabilities.embedded_context is True
        assert session.agent._execution_control is None
        assert new.modes is None
        await adapter.prompt(session.handle.id, [text_block("Write the file")])
        assert target.read_text() == "legacy"
        assert not client.permissions
        with pytest.raises(RequestError):
            await extension(adapter, session, "configure", mode="plan")


@pytest.mark.parametrize(
    "mode,choice,written,permissions",
    [
        ("code", "allow_once", True, 1),
        ("code", "reject", False, 1),
        ("plan", "allow_once", False, 0),
    ],
)
async def test_execution_policy_is_enforced_through_prompt(
    tmp_path, mode, choice, written, permissions
):
    target = tmp_path / "effect.txt"
    source = f"open({str(target)!r}, 'w').write('allowed')"
    async with session_for(tmp_path, reply("write", source=source), reply("done")) as (
        adapter,
        session,
        client,
        llm,
        handshake,
        new,
    ):
        assert new.modes.current_mode_id == "code"
        assert session.agent._execution_control is session.controls
        client.choice = choice
        await adapter.set_session_mode(session.handle.id, mode)
        configured = await extension(adapter, session, "runtime")
        assert configured["mode"] == mode
        assert configured["executionBackend"] == "native"
        assert (
            await adapter.prompt(session.handle.id, [text_block("Write if allowed")])
        ).stop_reason == "end_turn"
        assert target.exists() is written
        assert len(client.permissions) == permissions
        if permissions:
            assert client.permissions[0].raw_input == {"code": source}
            assert client.permissions[0].tool_call_id == "write"


async def test_pending_approval_can_be_cancelled_and_settings_do_not_race_turn(tmp_path):
    target = tmp_path / "never.txt"
    async with session_for(
        tmp_path, reply("write", source=f"open({str(target)!r}, 'w').write('x')")
    ) as (adapter, session, client, llm, handshake, new):
        client.release = asyncio.Event()
        turn = asyncio.create_task(
            adapter.prompt(session.handle.id, [text_block("Request approval")])
        )
        await client.entered.wait()
        assert (await extension(adapter, session, "runtime"))["approvalPending"]
        with pytest.raises(RequestError):
            await extension(adapter, session, "configure", approval="session")
        with pytest.raises(RequestError):
            await adapter.set_session_mode(session.handle.id, "plan")
        await adapter.cancel(session.handle.id)
        assert (await turn).stop_reason == "cancelled"
        assert not target.exists()
        state = await extension(adapter, session, "runtime")
        assert state["approval"] == "ask"
        assert not state["approvalPending"]


async def test_model_change_updates_next_call_and_session_grant_is_not_restored(
    tmp_path, monkeypatch
):
    async with session_for(tmp_path) as (adapter, session, client, llm, handshake, new):
        candidate = FakeLLMClient([reply("next", message="New model")], strict_exhaustion=True)
        candidate.model = "beta"
        session.controls._model_names = lambda: ["beta"]
        session.controls._model_factory = lambda name: candidate
        close = AsyncMock()
        monkeypatch.setattr(llm, "aclose", close)
        changed = await extension(adapter, session, "configure", model="beta", approval="session")
        assert changed["model"] == "beta"
        assert session.agent.llm is candidate
        close.assert_awaited_once()
        await adapter.prompt(session.handle.id, [text_block("Use new model")])
        assert candidate.call_count == 1 and llm.call_count == 0
        with pytest.raises(RequestError):
            await extension(adapter, session, "configure", model="not-configured")
        await adapter.close_session(session.handle.id)
        loaded = await adapter.load_session(cwd=str(tmp_path), session_id=session.handle.id)
        restored = (await adapter._sessions.get(session.handle.id)).value
        assert loaded.modes.current_mode_id == "code"
        assert restored.controls.status()["approval"] == "ask"


async def test_task_inspection_and_targeted_stop_work_during_foreground_prompt(
    tmp_path, monkeypatch
):
    async with session_for(tmp_path, reply("final")) as (
        adapter,
        session,
        client,
        llm,
        handshake,
        new,
    ):
        first = session.agent.todo.add("Inspect inputs")
        session.agent.todo.add("Apply change", deps=[first])
        session.agent.todo.activate(first)
        started_one, started_two = asyncio.Event(), asyncio.Event()
        entered, release = asyncio.Event(), asyncio.Event()

        async def run(started):
            started.set()
            await asyncio.Event().wait()

        session.agent.queue_manager.queue("one")
        session.agent.queue_manager.queue("two")
        one = session.agent.queue_manager.spawn(
            run(started_one), channel="one", label="First check"
        )
        two = session.agent.queue_manager.spawn(
            run(started_two), channel="two", label="Second check"
        )
        await asyncio.gather(started_one.wait(), started_two.wait())
        with pytest.raises(RequestError):
            await extension(adapter, session, "configure", mode="plan")
        original = llm.acall

        async def acall(*args, **kwargs):
            result = await original(*args, **kwargs)
            entered.set()
            await release.wait()
            return result

        monkeypatch.setattr(llm, "acall", acall)
        turn = asyncio.create_task(adapter.prompt(session.handle.id, [text_block("Inspect jobs")]))
        await entered.wait()
        active = session.dispatcher._active_task
        snapshot = await extension(adapter, session, "tasks")
        assert snapshot["currentTaskId"] == first.id
        assert snapshot["steps"][1]["blockedBy"] == [first.id]
        job = next(row for row in snapshot["jobs"] if row["channel"] == "one")
        stopped = await extension(adapter, session, "stop_task", taskId=job["id"])
        assert stopped["stopped"]
        assert one.state == "cancelled" and two.state == "running"
        assert session.dispatcher._active_task is active
        with pytest.raises(RequestError):
            await extension(adapter, session, "stop_task", taskId="unknown")
        release.set()
        await turn
        plans = [update for update in client.updates if isinstance(update, AgentPlanUpdate)]
        assert plans and plans[-1].field_meta["aster"]["tasks"]["currentTaskId"] == first.id


def image():
    return {
        "type": "image",
        "mimeType": "image/png",
        "data": _PNG,
        "_meta": {"aster": {"name": "screen.png"}},
    }


def resource():
    return {
        "type": "resource",
        "resource": {
            "uri": "file:///unreadable/from-server.py",
            "mimeType": "text/plain",
            "text": "Exact supplied contents sentinel_9817",
        },
    }


async def test_attachments_reach_request_and_history_through_protocol(tmp_path):
    async with session_for(tmp_path, reply("final"), enabled=False, images=True) as (
        adapter,
        session,
        client,
        llm,
        handshake,
        new,
    ):
        assert handshake.field_meta["aster"]["attachments"]
        await adapter.prompt(session.handle.id, [text_block("/help"), image(), resource()])
        assert llm.call_count == 1, "Attached context must not be swallowed by slash dispatch"
        assert "sentinel_9817" in str(llm.calls[0].messages)
        blocks = [
            block
            for message in llm.calls[0].messages
            for block in message.get("content", [])
            if hasattr(block, "get")
        ]
        images = [block for block in blocks if block.get("type") == "image_url"]
        assert len(images) == 1
        assert base64.b64decode(images[0]["image_url"]["url"].split(",", 1)[1]) == base64.b64decode(
            _PNG
        )
        saved = [
            event
            for event in session.agent.event_manager.values()
            if isinstance(event, Task) and event.metadata.get("aster_attachments")
        ]
        assert len(saved) == 1 and saved[0].images
        assert session.handle.info.turn_count == 1
        transcript = session.handle.turns()[0].content
        assert "screen.png" in transcript and "from-server.py" in transcript
        assert "sentinel_9817" not in transcript


async def test_invalid_attachment_is_rejected_before_recording_or_using_turn_id(tmp_path):
    async with session_for(tmp_path, reply("final"), enabled=False) as (
        adapter,
        session,
        client,
        llm,
        handshake,
        new,
    ):
        with pytest.raises(RequestError):
            await adapter.prompt(
                session.handle.id, [text_block("Look"), image()], aster={"turnId": "same"}
            )
        assert session.handle.info.turn_count == 0
        assert llm.call_count == 0
        await adapter.prompt(session.handle.id, [text_block("Just text")], aster={"turnId": "same"})
        assert session.handle.info.turn_count == 1
