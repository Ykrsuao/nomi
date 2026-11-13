# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conversation rewind changes the next model's context and survives reloads."""

import json
from contextlib import suppress

import pytest
from acp import RequestError, text_block
from aster_acp.server import CodingACPAdapter

from aster.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


class RecordingClient:
    def __init__(self):
        self.updates = []

    async def session_update(self, session_id, update, **kwargs):
        self.updates.append((session_id, update))


def completed_llm():
    return FakeLLMClient(
        [
            LLMResponse(
                raw_response=None,
                content="",
                tool_calls=[
                    ToolCall(
                        id=f"done-{index}",
                        name="return_result",
                        arguments=json.dumps(
                            {"result": {"message": "Completed.", "explanation": "done"}}
                        ),
                    )
                ],
                finish_reason="tool_calls",
            )
            for index in range(8)
        ],
        strict_exhaustion=True,
    )


@pytest.fixture
async def adapter_factory(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("ASTER_USER_DIR", str(home / "config"))
    monkeypatch.delenv("ASTER_SETTINGS", raising=False)
    built = []

    def make(factory=completed_llm):
        adapter = CodingACPAdapter(factory)
        adapter.on_connect(RecordingClient())
        built.append(adapter)
        return adapter

    yield make
    for adapter in built:
        with suppress(Exception):
            await adapter.close()


async def session(adapter, session_id):
    return (await adapter._sessions.get(session_id)).value


async def checkpoints(adapter, session_id):
    return await adapter.ext_method("aster/session/checkpoints", {"sessionId": session_id})


async def rewind(adapter, session_id, index):
    return await adapter.ext_method(
        "aster/session/rewind", {"sessionId": session_id, "userMessageIndex": index}
    )


async def test_rewind_restores_model_context_state_and_both_branches_after_restart(
    tmp_path, adapter_factory
):
    clients = []

    def factory():
        client = completed_llm()
        clients.append(client)
        return client

    adapter = adapter_factory(factory)
    initialized = await adapter.initialize(1)
    assert initialized.field_meta["aster"]["conversationRewind"] is True
    original = (await adapter.new_session(str(tmp_path), aster={"prepare": True})).session_id
    assert await checkpoints(adapter, original) == {"checkpoints": []}
    await adapter.prompt(original, [text_block("alpha-earlier-message")])
    source = await session(adapter, original)
    source.agent.vars["decision"] = "before-second"
    source.agent.context["saved_note"] = "context-before-second"
    prior_todo = source.agent.todo.add("before-second task")
    await adapter.prompt(original, [text_block("  beta-discarded-message\nexact spacing  ")])
    source.agent.vars["decision"] = "after-second"
    source.agent.vars["future"] = "future-value"
    source.agent.context["future_context"] = "future-context"
    source.agent.future_attribute = "future-attribute"
    source.agent.todo.complete(prior_todo)
    source.agent.todo.add("future-only task")
    await adapter.prompt(original, [text_block("gamma-discarded-message")])
    listed = await checkpoints(adapter, original)
    assert [item["userMessageIndex"] for item in listed["checkpoints"]] == [0, 1, 2]

    fork = await rewind(adapter, original, 1)
    assert fork["prompt"] == "  beta-discarded-message\nexact spacing  "
    assert fork["sessionId"] != original
    assert await adapter._sessions.ids() == (original,)
    await adapter.load_session(str(tmp_path), fork["sessionId"])
    branch = await session(adapter, fork["sessionId"])
    assert [turn.content for turn in branch.handle.turns() if turn.role == "user"] == [
        "alpha-earlier-message"
    ]
    assert branch.agent.vars["decision"] == "before-second"
    assert "future" not in branch.agent.vars
    assert "future_context" not in branch.agent.context
    assert not hasattr(branch.agent, "future_attribute")
    assert branch.agent.context["saved_note"] == "context-before-second"
    assert branch.agent.todo.get(prior_todo.id).status == "open"
    assert "future-only task" not in branch.agent.todo.status()
    branch.agent.vars["branch_value"] = "saved-on-branch"
    await adapter.prompt(fork["sessionId"], [text_block("delta-replacement-message")])
    model_context = repr(clients[-1].calls[-1].messages)
    assert "alpha-earlier-message" in model_context
    assert "delta-replacement-message" in model_context
    assert "beta-discarded-message" not in model_context
    assert "gamma-discarded-message" not in model_context
    assert "future-value" not in model_context
    assert "future-context" not in model_context
    assert "future-attribute" not in model_context
    assert [turn.content for turn in source.handle.turns() if turn.role == "user"] == [
        "alpha-earlier-message",
        "  beta-discarded-message\nexact spacing  ",
        "gamma-discarded-message",
    ]
    await adapter.close()

    restarted = adapter_factory(factory)
    await restarted.load_session(str(tmp_path), fork["sessionId"])
    restored = await session(restarted, fork["sessionId"])
    assert restored.agent.vars["decision"] == "before-second"
    assert restored.agent.vars["branch_value"] == "saved-on-branch"
    assert [
        item["userMessageIndex"]
        for item in (await checkpoints(restarted, fork["sessionId"]))["checkpoints"]
    ] == [0, 1]
    await restarted.load_session(str(tmp_path), original)
    resumed_source = await session(restarted, original)
    assert resumed_source.agent.vars["decision"] == "after-second"
    assert resumed_source.agent.vars["future"] == "future-value"
    assert resumed_source.agent.future_attribute == "future-attribute"

    first = await rewind(restarted, fork["sessionId"], 0)
    await restarted.load_session(str(tmp_path), first["sessionId"])
    empty = await session(restarted, first["sessionId"])
    assert empty.handle.turns() == []
    assert not empty.handle.info.provisional
    await restarted.close_session(first["sessionId"])
    assert restarted._store(tmp_path).get(first["sessionId"]).turn_count == 0


async def test_legacy_messages_are_not_offered_and_new_turns_gain_checkpoints(
    tmp_path, adapter_factory
):
    adapter = adapter_factory()
    original = (await adapter.new_session(str(tmp_path))).session_id
    current = await session(adapter, original)
    current.handle.record_user_message("old message without checkpoint")
    assert await checkpoints(adapter, original) == {"checkpoints": []}
    with pytest.raises(RequestError) as failure:
        await rewind(adapter, original, 0)
    assert failure.value.code == -32602
    await adapter.prompt(original, [text_block("new checkpointed message")])
    assert await checkpoints(adapter, original) == {
        "checkpoints": [{"userMessageIndex": 1, "prompt": "new checkpointed message"}]
    }


@pytest.mark.parametrize("index", [-1, True, 1.0, "0", None, 999])
async def test_invalid_rewind_does_not_create_a_session(tmp_path, adapter_factory, index):
    adapter = adapter_factory()
    original = (await adapter.new_session(str(tmp_path))).session_id
    await adapter.prompt(original, [text_block("message")])
    with pytest.raises(RequestError) as failure:
        await rewind(adapter, original, index)
    assert failure.value.code == -32602
    assert [info.id for info in adapter._store(tmp_path).list()] == [original]


async def test_checkpoints_and_rewind_reject_busy_or_closed_sessions(tmp_path, adapter_factory):
    adapter = adapter_factory()
    original = (await adapter.new_session(str(tmp_path))).session_id
    await adapter.prompt(original, [text_block("message")])
    runtime = await adapter._sessions.get(original)
    async with runtime.turn():
        for operation in (checkpoints(adapter, original), rewind(adapter, original, 0)):
            with pytest.raises(RequestError) as failure:
                await operation
            assert failure.value.code == -32600
    await adapter.close_session(original)
    with pytest.raises(RequestError) as failure:
        await rewind(adapter, original, 0)
    assert failure.value.code == -32002


async def test_checkpoint_failure_keeps_prompt_and_invalidates_old_state(
    tmp_path, adapter_factory, monkeypatch
):
    from aster.errors.storage import SerializationError

    adapter = adapter_factory()
    original = (await adapter.new_session(str(tmp_path))).session_id
    await adapter.prompt(original, [text_block("first")])
    current = await session(adapter, original)

    def cannot_serialize(*args, **kwargs):
        raise SerializationError("unsupported runtime state")

    monkeypatch.setattr(current.handle.storage, "save_checkpoint", cannot_serialize)
    monkeypatch.setattr(current.handle.storage, "save_current_snapshot", cannot_serialize)
    assert (await adapter.prompt(original, [text_block("second")])).stop_reason == "end_turn"
    assert [turn.content for turn in current.handle.turns() if turn.role == "user"] == [
        "first",
        "second",
    ]
    assert [
        item["userMessageIndex"] for item in (await checkpoints(adapter, original))["checkpoints"]
    ] == [0]
    assert not current.handle.storage.restore_current_snapshot(current.agent)


@pytest.mark.parametrize("placement", ["vars", "todo", "nested"])
async def test_mutated_unserializable_vars_do_not_offer_partial_rewind(
    tmp_path, adapter_factory, placement
):
    from aster.storage.snapshot_vars import SnapshotVars

    adapter = adapter_factory()
    original = (await adapter.new_session(str(tmp_path))).session_id
    await adapter.prompt(original, [text_block("first")])
    current = await session(adapter, original)
    if placement == "todo":
        store = current.agent.todo.add("Keep task state").vars
    elif placement == "nested":
        store = SnapshotVars()
        current.agent.vars["nested"] = {"task": store}
    else:
        store = current.agent.vars
    state = {"important": "keep this", "results": []}
    store["state"] = state
    state["results"].append(object())

    assert (await adapter.prompt(original, [text_block("second")])).stop_reason == "end_turn"
    assert [turn.content for turn in current.handle.turns() if turn.role == "user"] == [
        "first",
        "second",
    ]
    assert store["state"] is state
    assert await checkpoints(adapter, original) == {
        "checkpoints": [{"userMessageIndex": 0, "prompt": "first"}]
    }
    with pytest.raises(RequestError):
        await rewind(adapter, original, 1)
    assert not current.handle.storage.restore_current_snapshot(current.agent)

    # Once the live state becomes serializable again, future messages can rewind.
    state["results"].clear()
    await adapter.prompt(original, [text_block("third")])
    assert [
        item["userMessageIndex"] for item in (await checkpoints(adapter, original))["checkpoints"]
    ] == [0, 2]
    fork = await rewind(adapter, original, 2)
    await adapter.load_session(str(tmp_path), fork["sessionId"])
    branch = await session(adapter, fork["sessionId"])
    restored = (
        branch.agent.todo.list_todos()[0].vars
        if placement == "todo"
        else branch.agent.vars["nested"]["task"]
        if placement == "nested"
        else branch.agent.vars
    )
    assert restored["state"] == {"important": "keep this", "results": []}


async def test_rewind_leaves_working_files_unchanged(tmp_path, adapter_factory):
    adapter = adapter_factory()
    file = tmp_path / "edited.txt"
    file.write_text("before", encoding="utf-8")
    original = (await adapter.new_session(str(tmp_path))).session_id
    await adapter.prompt(original, [text_block("first")])
    file.write_text("after", encoding="utf-8")
    await rewind(adapter, original, 0)
    assert file.read_text(encoding="utf-8") == "after"


async def test_fork_does_not_offer_old_boundary_when_replacement_capture_fails(
    tmp_path, adapter_factory, monkeypatch
):
    from aster.errors.storage import SerializationError

    adapter = adapter_factory()
    original = (await adapter.new_session(str(tmp_path))).session_id
    await adapter.prompt(original, [text_block("first")])
    await adapter.prompt(original, [text_block("second")])
    fork = await rewind(adapter, original, 1)
    await adapter.load_session(str(tmp_path), fork["sessionId"])
    branch = await session(adapter, fork["sessionId"])

    def cannot_serialize(*args, **kwargs):
        raise SerializationError("unsupported current state")

    monkeypatch.setattr(branch.handle.storage, "save_checkpoint", cannot_serialize)
    await adapter.prompt(fork["sessionId"], [text_block("replacement")])
    assert await checkpoints(adapter, fork["sessionId"]) == {
        "checkpoints": [{"userMessageIndex": 0, "prompt": "first"}]
    }
    with pytest.raises(RequestError):
        await rewind(adapter, fork["sessionId"], 1)


async def test_failed_user_record_does_not_leave_a_reusable_old_checkpoint(
    tmp_path, adapter_factory, monkeypatch
):
    from aster.errors.storage import SerializationError

    adapter = adapter_factory()
    original = (await adapter.new_session(str(tmp_path))).session_id
    current = await session(adapter, original)
    actual_record = current.handle.record_user_message

    def cannot_record(*args, **kwargs):
        raise OSError("write failed")

    monkeypatch.setattr(current.handle, "record_user_message", cannot_record)
    with pytest.raises(OSError, match="write failed"):
        await adapter.prompt(original, [text_block("failed to save")])
    assert await checkpoints(adapter, original) == {"checkpoints": []}
    monkeypatch.setattr(current.handle, "record_user_message", actual_record)

    def cannot_serialize(*args, **kwargs):
        raise SerializationError("state has changed")

    monkeypatch.setattr(current.handle.storage, "save_checkpoint", cannot_serialize)
    await adapter.prompt(original, [text_block("saved after retry")])
    assert await checkpoints(adapter, original) == {"checkpoints": []}
    with pytest.raises(RequestError):
        await rewind(adapter, original, 0)


async def test_failed_fork_removes_only_its_partial_database(
    tmp_path, adapter_factory, monkeypatch
):
    adapter = adapter_factory()
    original = (await adapter.new_session(str(tmp_path))).session_id
    await adapter.prompt(original, [text_block("message")])
    current = await session(adapter, original)

    def cannot_fork(*args, **kwargs):
        raise OSError("fork write failed")

    monkeypatch.setattr(current.handle.storage, "fork_checkpoint", cannot_fork)
    with pytest.raises(OSError, match="fork write failed"):
        await rewind(adapter, original, 0)
    assert list((tmp_path / ".aster" / "sessions").glob("*.db")) == [current.handle.path]
    assert await checkpoints(adapter, original) == {
        "checkpoints": [{"userMessageIndex": 0, "prompt": "message"}]
    }
