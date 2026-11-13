# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tree navigation preserves branches and changes the next model's actual context."""

import asyncio
import json
from contextlib import suppress
from unittest.mock import AsyncMock

import pytest
from acp import RequestError, text_block
from acp.interfaces import Client
from aster_acp.server import CodingACPAdapter
from aster_cli.coding.reply_strategy import CodingReplyStrategy
from aster_cli.sessions import SessionTurnFinished

from aster.config import CodeActConfig
from aster.errors import GenerationError
from aster.errors.storage import SerializationError
from aster.strategies.codeact import OUTPUT_TOKENS_EXHAUSTED_MESSAGE
from aster.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


@pytest.fixture
async def tree_adapters(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("ASTER_USER_DIR", str(home / "config"))
    monkeypatch.delenv("ASTER_SETTINGS", raising=False)
    built = []
    clients = []

    def factory(responses=None):
        client = FakeLLMClient(
            responses
            if responses is not None
            else [
                LLMResponse(
                    raw_response=None,
                    parts=(
                        ToolCall(
                            id=f"done-{index}",
                            name="return_result",
                            arguments=json.dumps(
                                {"result": {"message": f"Reply {index}", "explanation": "done"}}
                            ),
                        ),
                    ),
                    finish_reason="tool_calls",
                )
                for index in range(8)
            ],
            strict_exhaustion=True,
        )
        clients.append(client)
        return client

    def make(responses=None):
        adapter = CodingACPAdapter(lambda: factory(responses))
        adapter.on_connect(AsyncMock(spec=Client))
        built.append(adapter)
        return adapter

    yield make, clients
    for adapter in built:
        with suppress(Exception):
            await adapter.close()


async def tree(adapter, session_id):
    return await adapter.ext_method("aster/session/tree", {"sessionId": session_id})


async def navigate(adapter, session_id, node_id, current_id):
    return await adapter.ext_method(
        "aster/session/navigate",
        {"sessionId": session_id, "nodeId": node_id, "expectedCurrentNodeId": current_id},
    )


async def live_session(adapter, session_id):
    return (await adapter._sessions.get(session_id)).value


async def test_message_branches_restore_context_and_survive_restart(tmp_path, tree_adapters):
    make, clients = tree_adapters
    adapter = make()
    initialized = await adapter.initialize(1)
    assert initialized.field_meta["aster"]["sessionTree"] is True
    original = (await adapter.new_session(str(tmp_path), aster={"prepare": True})).session_id
    empty = await tree(adapter, original)
    assert empty["nodes"] == []
    assert empty["currentNodeId"] is None
    source = await live_session(adapter, original)
    source.agent.vars["decision"] = "state-after-alpha"
    await adapter.prompt(original, [text_block("alpha-shared-history")])
    first = await tree(adapter, original)
    first_reply = first["currentNodeId"]
    source.agent.vars["decision"] = "state-before-beta"
    beta_text = "  beta-original-branch\nexact spacing  "
    await adapter.prompt(original, [text_block(beta_text)])
    original_tree = await tree(adapter, original)
    beta = next(node for node in original_tree["nodes"] if "beta-original" in node["preview"])
    beta_reply = original_tree["currentNodeId"]
    calls = sum(len(client.calls) for client in clients)
    working_file = tmp_path / "working-file.txt"
    working_file.write_text("keep current files", encoding="utf-8")

    selected = await navigate(adapter, original, beta["id"], beta_reply)
    assert selected["prompt"] == beta_text
    assert selected["sessionId"] != original
    assert sum(len(client.calls) for client in clients) == calls
    assert await adapter._sessions.ids() == (original,)
    branch_id = selected["sessionId"]
    await adapter.load_session(str(tmp_path), branch_id)
    branch = await live_session(adapter, branch_id)
    assert branch.agent.vars["decision"] == "state-before-beta"
    assert [turn.content for turn in branch.handle.turns() if turn.role == "user"] == [
        "alpha-shared-history"
    ]
    branch.agent.vars["decision"] = "state-after-gamma"
    await adapter.prompt(branch_id, [text_block("gamma-alternate-branch")])
    context = repr(clients[-1].calls[-1].messages)
    assert "alpha-shared-history" in context
    assert "gamma-alternate-branch" in context
    assert "beta-original-branch" not in context

    branched = await tree(adapter, branch_id)
    users = [node for node in branched["nodes"] if node["role"] == "user"]
    assert len(users) == 3
    assert sum("alpha-shared" in node["preview"] for node in users) == 1
    gamma = next(node for node in users if "gamma-alternate" in node["preview"])
    assert gamma["parentId"] == beta["parentId"] == first_reply
    assert beta_reply in {node["id"] for node in branched["nodes"]}
    assert branched["currentNodeId"] != beta_reply
    assert len({node["id"] for node in branched["nodes"]}) == len(branched["nodes"])
    await adapter.close()

    restarted = make()
    await restarted.load_session(str(tmp_path), branch_id)
    restored = await tree(restarted, branch_id)
    assert restored == branched
    restored_branch = await live_session(restarted, branch_id)
    assert restored_branch.agent.vars["decision"] == "state-after-gamma"

    # An assistant target restores its own completed state, rather than the
    # newer state captured immediately before the next user message.
    selected_reply = await navigate(restarted, branch_id, first_reply, restored["currentNodeId"])
    assert selected_reply["prompt"] is None
    reply_branch_id = selected_reply["sessionId"]
    await restarted.load_session(str(tmp_path), reply_branch_id)
    reply_branch = await live_session(restarted, reply_branch_id)
    assert reply_branch.agent.vars["decision"] == "state-after-alpha"
    await restarted.prompt(reply_branch_id, [text_block("delta-from-first-reply")])
    reply_context = repr(clients[-1].calls[-1].messages)
    assert "alpha-shared-history" in reply_context
    assert "delta-from-first-reply" in reply_context
    assert "beta-original-branch" not in reply_context
    assert "gamma-alternate-branch" not in reply_context
    assert working_file.read_text(encoding="utf-8") == "keep current files"
    assert [
        turn.content
        for turn in restarted._store(tmp_path).load_turns(original)
        if turn.role == "user"
    ] == [
        "alpha-shared-history",
        beta_text,
    ]


async def test_navigation_rejects_stale_or_unrelated_nodes_without_creating_branches(
    tmp_path, tree_adapters
):
    make, clients = tree_adapters
    adapter = make()
    original = (await adapter.new_session(str(tmp_path))).session_id
    await adapter.prompt(original, [text_block("first")])
    first = await tree(adapter, original)
    first_user = next(node for node in first["nodes"] if node["role"] == "user")
    await adapter.prompt(original, [text_block("second")])
    current = await tree(adapter, original)
    other = (await adapter.new_session(str(tmp_path))).session_id
    await adapter.prompt(other, [text_block("unrelated-session-input")])
    other_tree = await tree(adapter, other)
    paths = set((tmp_path / ".aster" / "sessions").glob("*.db"))
    calls = sum(len(client.calls) for client in clients)

    for node_id, cursor in (
        (first_user["id"], first["currentNodeId"]),
        ("../../outside", current["currentNodeId"]),
        (other_tree["currentNodeId"], current["currentNodeId"]),
    ):
        with pytest.raises(RequestError) as failure:
            await navigate(adapter, original, node_id, cursor)
        assert failure.value.code == -32602
    assert set((tmp_path / ".aster" / "sessions").glob("*.db")) == paths
    assert await tree(adapter, original) == current
    assert not any("unrelated-session" in node["preview"] for node in current["nodes"])
    assert sum(len(client.calls) for client in clients) == calls


async def test_current_reply_restores_saved_state_and_navigation_respects_busy_runtime(
    tmp_path, tree_adapters
):
    make, clients = tree_adapters
    adapter = make()
    session_id = (await adapter.new_session(str(tmp_path))).session_id
    source = await live_session(adapter, session_id)
    source.agent.vars["decision"] = "saved reply state"
    await adapter.prompt(session_id, [text_block("first")])
    current = await tree(adapter, session_id)
    source.agent.vars["decision"] = "unsaved later state"
    paths = set((tmp_path / ".aster" / "sessions").glob("*.db"))
    result = await navigate(adapter, session_id, current["currentNodeId"], current["currentNodeId"])
    assert result["sessionId"] != session_id
    assert result["prompt"] is None
    assert set((tmp_path / ".aster" / "sessions").glob("*.db")) == paths | {
        adapter._store(tmp_path).path_for(result["sessionId"])
    }
    await adapter.load_session(str(tmp_path), result["sessionId"])
    restored = await live_session(adapter, result["sessionId"])
    assert restored.agent.vars["decision"] == "saved reply state"
    assert source.agent.vars["decision"] == "unsaved later state"
    assert await tree(adapter, result["sessionId"]) == current
    runtime = await adapter._sessions.get(session_id)
    async with runtime.turn():
        for method, params in (
            ("aster/session/tree", {"sessionId": session_id}),
            (
                "aster/session/navigate",
                {
                    "sessionId": session_id,
                    "nodeId": current["currentNodeId"],
                    "expectedCurrentNodeId": current["currentNodeId"],
                },
            ),
        ):
            with pytest.raises(RequestError) as failure:
                await adapter.ext_method(method, params)
            assert failure.value.code == -32600
    assert len(clients[0].calls) == 1
    assert sum(len(client.calls) for client in clients) == 1


async def test_user_parent_then_current_reply_restores_two_distinct_saved_boundaries(
    tmp_path, tree_adapters
):
    make, clients = tree_adapters
    adapter = make()
    session_id = (await adapter.new_session(str(tmp_path))).session_id
    source = await live_session(adapter, session_id)
    source.agent.vars["decision"] = "after alpha"
    await adapter.prompt(session_id, [text_block("alpha")])
    first_reply = (await tree(adapter, session_id))["currentNodeId"]
    source.agent.vars["decision"] = "before beta"
    await adapter.prompt(session_id, [text_block("beta")])
    current = await tree(adapter, session_id)
    beta = next(node for node in current["nodes"] if node["preview"] == "beta")

    before_beta = await navigate(adapter, session_id, beta["id"], current["currentNodeId"])
    await adapter.load_session(str(tmp_path), before_beta["sessionId"])
    parent = await live_session(adapter, before_beta["sessionId"])
    assert parent.agent.vars["decision"] == "before beta"
    assert (await tree(adapter, parent.handle.id))["currentNodeId"] == first_reply

    after_alpha = await navigate(adapter, parent.handle.id, first_reply, first_reply)
    assert after_alpha["sessionId"] != parent.handle.id
    assert after_alpha["prompt"] is None
    await adapter.load_session(str(tmp_path), after_alpha["sessionId"])
    restored = await live_session(adapter, after_alpha["sessionId"])
    assert restored.agent.vars["decision"] == "after alpha"
    assert parent.agent.vars["decision"] == "before beta"
    assert sum(len(client.calls) for client in clients) == 2


async def test_navigation_preserves_running_background_work(tmp_path, tree_adapters):
    make, clients = tree_adapters
    adapter = make()
    session_id = (await adapter.new_session(str(tmp_path))).session_id
    await adapter.prompt(session_id, [text_block("first")])
    current = await tree(adapter, session_id)
    source = await live_session(adapter, session_id)
    started = asyncio.Event()

    async def background_job():
        started.set()
        await asyncio.Event().wait()

    source.agent.queue_manager.queue("checks")
    handle = source.agent.queue_manager.spawn(background_job(), channel="checks", label="Tests")
    try:
        await asyncio.wait_for(started.wait(), 2)
        paths = set((tmp_path / ".aster" / "sessions").glob("*.db"))
        assert await tree(adapter, session_id) == current
        with pytest.raises(RequestError) as failure:
            await navigate(adapter, session_id, current["currentNodeId"], current["currentNodeId"])
        assert failure.value.code == -32600
        assert isinstance(failure.value.data, dict)
        assert "background jobs" in failure.value.data["reason"]
        assert handle.state == "running"
        assert set((tmp_path / ".aster" / "sessions").glob("*.db")) == paths
        assert sum(len(client.calls) for client in clients) == 1
    finally:
        await handle.cancel()


async def test_generation_limit_keeps_progress_visible_without_a_restore_boundary(
    tmp_path, tree_adapters
):
    make, clients = tree_adapters
    codes = [
        "self.vars['stage'] = 'at progress'\nself.message('A progress update')",
        "self.vars['stage'] = 'later unfinished work'\nself.context['later_only'] = 'later context'",
    ]
    adapter = make(
        [
            LLMResponse(
                raw_response=None,
                parts=(
                    ToolCall(
                        id=f"step-{index}",
                        name="execute_python",
                        arguments=json.dumps({"code": code}),
                    ),
                ),
                finish_reason="tool_calls",
            )
            for index, code in enumerate(codes)
        ]
    )
    session_id = (await adapter.new_session(str(tmp_path))).session_id
    source = await live_session(adapter, session_id)
    source.dispatcher._strategy = CodingReplyStrategy(
        config=CodeActConfig(prefill=None, max_iterations=2)
    )
    response = await adapter.prompt(session_id, [text_block("complete both steps")])
    assert response.stop_reason == "max_turn_requests"
    assert source.agent.vars["stage"] == "later unfinished work"
    current = await tree(adapter, session_id)
    progress = next(node for node in current["nodes"] if node["role"] == "assistant")
    assert progress["preview"] == "A progress update"
    assert progress["navigable"] is False
    paths = set((tmp_path / ".aster" / "sessions").glob("*.db"))
    with pytest.raises(RequestError):
        await navigate(adapter, session_id, progress["id"], current["currentNodeId"])
    assert set((tmp_path / ".aster" / "sessions").glob("*.db")) == paths
    await adapter.close()
    restarted = make()
    await restarted.load_session(str(tmp_path), session_id)
    assert await tree(restarted, session_id) == current
    assert sum(len(client.calls) for client in clients) == 2


async def test_cancelled_turn_keeps_progress_and_stop_note_without_restore_boundaries(
    tmp_path, tree_adapters
):
    make, clients = tree_adapters
    adapter = make(
        [
            LLMResponse(
                raw_response=None,
                parts=(
                    ToolCall(
                        id="waiting",
                        name="execute_python",
                        arguments=json.dumps(
                            {
                                "code": (
                                    "import asyncio\nself.message('A progress update')\n"
                                    "self.vars['stage'] = 'after progress'\nawait asyncio.Event().wait()"
                                )
                            }
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            )
        ]
    )
    session_id = (await adapter.new_session(str(tmp_path))).session_id
    source = await live_session(adapter, session_id)
    progress_seen = asyncio.Event()
    source.agent.event_manager.on("AgentMessage", lambda event: progress_seen.set())
    prompt = asyncio.create_task(adapter.prompt(session_id, [text_block("wait until stopped")]))
    try:
        await asyncio.wait_for(progress_seen.wait(), 2)
        await adapter.cancel(session_id)
        assert (await asyncio.wait_for(prompt, 2)).stop_reason == "cancelled"
        current = await tree(adapter, session_id)
        assistants = [node for node in current["nodes"] if node["role"] == "assistant"]
        assert [node["preview"] for node in assistants] == [
            "A progress update",
            "Stopped at your request.",
        ]
        assert all(node["navigable"] is False for node in assistants)
        assert "after:0" not in source.handle.storage.checkpoint_ids()
        await adapter.close()
        restarted = make()
        await restarted.load_session(str(tmp_path), session_id)
        assert await tree(restarted, session_id) == current
        assert sum(len(client.calls) for client in clients) == 1
    finally:
        if not prompt.done():
            prompt.cancel()
        await asyncio.gather(prompt, return_exceptions=True)


@pytest.mark.parametrize("failure", ["max_tokens", "exception"])
async def test_failed_dispatch_does_not_restore_partial_reply(
    tmp_path, tree_adapters, monkeypatch, failure
):
    make, clients = tree_adapters
    adapter = make()
    session_id = (await adapter.new_session(str(tmp_path))).session_id
    source = await live_session(adapter, session_id)

    async def failed_submit(text):
        source.agent.message("Progress before failure")
        source.agent.vars["later"] = "state after the progress message"
        if failure == "max_tokens":
            raise GenerationError(OUTPUT_TOKENS_EXHAUSTED_MESSAGE)
        raise RuntimeError("dispatcher failed")

    monkeypatch.setattr(source.dispatcher, "submit", failed_submit)
    if failure == "max_tokens":
        assert (await adapter.prompt(session_id, [text_block("first")])).stop_reason == "max_tokens"
    else:
        with pytest.raises(RuntimeError, match="dispatcher failed"):
            await adapter.prompt(session_id, [text_block("first")])
    current = await tree(adapter, session_id)
    assert current["nodes"][-1]["preview"] == "Progress before failure"
    assert current["nodes"][-1]["navigable"] is False
    assert "after:0" not in source.handle.storage.checkpoint_ids()
    await adapter.close()
    restarted = make()
    await restarted.load_session(str(tmp_path), session_id)
    assert await tree(restarted, session_id) == current
    assert sum(len(client.calls) for client in clients) == 0


@pytest.mark.parametrize("failed_save", ["after", "current", "both"])
async def test_save_failure_never_substitutes_current_state_for_missing_completed_checkpoint(
    tmp_path, tree_adapters, monkeypatch, failed_save
):
    make, clients = tree_adapters
    adapter = make()
    session_id = (await adapter.new_session(str(tmp_path))).session_id
    source = await live_session(adapter, session_id)
    source.agent.vars["stage"] = "completed state"
    save_checkpoint = source.handle.storage.save_checkpoint

    def failed_checkpoint(key, agent):
        if key.startswith("after:"):
            raise SerializationError("cannot capture completed checkpoint")
        save_checkpoint(key, agent)

    def failed_current(agent):
        raise SerializationError("cannot capture current snapshot")

    if failed_save in {"after", "both"}:
        monkeypatch.setattr(source.handle.storage, "save_checkpoint", failed_checkpoint)
    if failed_save in {"current", "both"}:
        monkeypatch.setattr(source.handle.storage, "save_current_snapshot", failed_current)
    assert (await adapter.prompt(session_id, [text_block("first")])).stop_reason == "end_turn"
    current = await tree(adapter, session_id)
    reply = current["currentNodeId"]
    assert current["nodes"][-1]["navigable"] is (failed_save == "current")
    paths = set((tmp_path / ".aster" / "sessions").glob("*.db"))
    if failed_save == "current":
        selected = await navigate(adapter, session_id, reply, reply)
        await adapter.load_session(str(tmp_path), selected["sessionId"])
        restored = await live_session(adapter, selected["sessionId"])
        assert restored.agent.vars["stage"] == "completed state"
    else:
        with pytest.raises(RequestError):
            await navigate(adapter, session_id, reply, reply)
        assert set((tmp_path / ".aster" / "sessions").glob("*.db")) == paths
    await adapter.close()
    restarted = make()
    await restarted.load_session(str(tmp_path), session_id)
    assert await tree(restarted, session_id) == current
    assert sum(len(client.calls) for client in clients) == 1


async def test_rejected_duplicate_turn_id_preserves_previous_completion(tmp_path, tree_adapters):
    make, clients = tree_adapters
    adapter = make()
    session_id = (await adapter.new_session(str(tmp_path))).session_id
    await adapter.prompt(session_id, [text_block("first")], aster={"turnId": "same-id"})
    current = await tree(adapter, session_id)
    with pytest.raises(RequestError) as failure:
        await adapter.prompt(session_id, [text_block("not accepted")], aster={"turnId": "same-id"})
    assert failure.value.code == -32602
    assert await tree(adapter, session_id) == current
    selected = await navigate(
        adapter, session_id, current["currentNodeId"], current["currentNodeId"]
    )
    await adapter.load_session(str(tmp_path), selected["sessionId"])
    assert await tree(adapter, selected["sessionId"]) == current
    assert sum(len(client.calls) for client in clients) == 1


async def test_completion_metadata_write_failure_cannot_publish_a_legacy_current_boundary(
    tmp_path, tree_adapters, monkeypatch
):
    make, clients = tree_adapters
    adapter = make()
    session_id = (await adapter.new_session(str(tmp_path))).session_id
    source = await live_session(adapter, session_id)
    add_event = source.handle.events.add

    def fail_outcome(event, *, record=True):
        if isinstance(event, SessionTurnFinished):
            raise OSError("completion metadata unavailable")
        return add_event(event, record=record)

    monkeypatch.setattr(source.handle.events, "add", fail_outcome)
    with pytest.raises(OSError, match="completion metadata unavailable"):
        await adapter.prompt(session_id, [text_block("first")])
    current = await tree(adapter, session_id)
    assert current["nodes"][-1]["preview"] == "Reply 0"
    assert current["nodes"][-1]["navigable"] is False
    assert not source.handle.storage.restore_current_snapshot(source.agent)
    assert "after:0" not in source.handle.storage.checkpoint_ids()
    await adapter.close()
    restarted = make()
    await restarted.load_session(str(tmp_path), session_id)
    assert await tree(restarted, session_id) == current
    assert sum(len(client.calls) for client in clients) == 1


@pytest.mark.parametrize("release_on_reply", [False, True], ids=["after-prompt", "during-reply"])
async def test_completed_reply_stays_navigable_after_background_messages_and_next_turn(
    tmp_path, tree_adapters, release_on_reply
):
    make, clients = tree_adapters
    adapter = make()
    session_id = (await adapter.new_session(str(tmp_path))).session_id
    source = await live_session(adapter, session_id)
    source.agent.vars["stage"] = "foreground boundary"
    released = asyncio.Event()
    started = asyncio.Event()

    def release_background(event):
        if release_on_reply and event.content == "Reply 0":
            released.set()

    unsubscribe = source.agent.event_manager.on("AgentMessage", release_background)

    async def background_job():
        started.set()
        await released.wait()
        source.agent.vars["stage"] = "background finished"
        source.agent.context["background_only"] = "late context"
        source.agent.message("The background job finished")

    source.agent.queue_manager.queue("checks")
    job = source.agent.queue_manager.spawn(background_job(), channel="checks", label="Tests")
    try:
        await asyncio.wait_for(started.wait(), 2)
        assert (await adapter.prompt(session_id, [text_block("first")])).stop_reason == "end_turn"
        completed_id = next(
            node["id"]
            for node in (await tree(adapter, session_id))["nodes"]
            if node["preview"] == "Reply 0"
        )
        released.set()
        await asyncio.wait_for(job._task, 2)
        current = await tree(adapter, session_id)
        nodes = {node["id"]: node for node in current["nodes"]}
        background_id = current["currentNodeId"]
        assert background_id != completed_id
        assert nodes[completed_id]["navigable"] is True
        assert nodes[background_id]["navigable"] is False
        selected = await navigate(adapter, session_id, completed_id, background_id)
        await adapter.load_session(str(tmp_path), selected["sessionId"])
        restored = await live_session(adapter, selected["sessionId"])
        assert restored.agent.vars["stage"] == "foreground boundary"
        assert "background_only" not in restored.agent.context

        await adapter.prompt(session_id, [text_block("next user turn")])
        later = await tree(adapter, session_id)
        next_user = next(node for node in later["nodes"] if node["preview"] == "next user turn")
        before_next = await navigate(adapter, session_id, next_user["id"], later["currentNodeId"])
        await adapter.load_session(str(tmp_path), before_next["sessionId"])
        await adapter.close()
        # The remaining sibling must carry its own retained after checkpoint,
        # not silently restore from the original or the earlier reply clone.
        adapter._store(tmp_path).delete(session_id)
        adapter._store(tmp_path).delete(selected["sessionId"])
        restarted = make()
        await restarted.load_session(str(tmp_path), before_next["sessionId"])
        fork_tree = await tree(restarted, before_next["sessionId"])
        selected_again = await navigate(
            restarted, before_next["sessionId"], completed_id, fork_tree["currentNodeId"]
        )
        await restarted.load_session(str(tmp_path), selected_again["sessionId"])
        precise = await live_session(restarted, selected_again["sessionId"])
        assert precise.agent.vars["stage"] == "foreground boundary"
        assert "background_only" not in precise.agent.context
        final_nodes = {
            node["id"]: node for node in (await tree(restarted, precise.handle.id))["nodes"]
        }
        assert final_nodes[background_id]["navigable"] is False
        assert sum(len(client.calls) for client in clients) == 2
    finally:
        unsubscribe()
        await job.cancel()


async def test_tree_search_matches_full_messages_on_inactive_branches(tmp_path, tree_adapters):
    make, clients = tree_adapters
    adapter = make()
    initialized = await adapter.initialize(1)
    assert initialized.field_meta["aster"]["sessionTreeSearch"] is True
    session_id = (await adapter.new_session(str(tmp_path))).session_id
    await adapter.prompt(session_id, [text_block("shared prefix")])
    await adapter.prompt(session_id, [text_block("x" * 600 + " needle_after_preview")])
    current = await tree(adapter, session_id)
    original_user = next(node for node in current["nodes"] if node["preview"] == "x" * 500)
    alternate = await navigate(adapter, session_id, original_user["id"], current["currentNodeId"])
    await adapter.load_session(str(tmp_path), alternate["sessionId"])
    await adapter.prompt(alternate["sessionId"], [text_block("different branch")])
    unfiltered = await tree(adapter, alternate["sessionId"])
    calls = sum(len(client.calls) for client in clients)
    query = " NeEdLe_AfTeR_PrEvIeW "
    searched = await adapter.ext_method(
        "aster/session/tree", {"sessionId": alternate["sessionId"], "query": query}
    )
    assert searched["query"] == query
    assert searched["matchedNodeIds"] == [original_user["id"]]
    assert searched["currentNodeId"] == unfiltered["currentNodeId"]
    assert searched["nodes"] == unfiltered["nodes"]
    assert "needle_after_preview" not in json.dumps(searched["nodes"])
    assert sum(len(client.calls) for client in clients) == calls


@pytest.mark.parametrize(
    "query", [None, 42, True, [], "q" * 4097], ids=["null", "number", "bool", "list", "too-long"]
)
async def test_tree_search_validates_query_before_opening_a_session(query):
    adapter = CodingACPAdapter(lambda: pytest.fail("invalid search must not create a model"))
    with pytest.raises(RequestError) as failure:
        await adapter.ext_method("aster/session/tree", {"sessionId": "unknown", "query": query})
    assert failure.value.code == -32602


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"nodeId": None, "expectedCurrentNodeId": None},
        {"nodeId": True, "expectedCurrentNodeId": None},
        {"nodeId": "", "expectedCurrentNodeId": None},
        {"nodeId": "x" * 257, "expectedCurrentNodeId": None},
        {"nodeId": "node"},
        {"nodeId": "node", "expectedCurrentNodeId": False},
        {"nodeId": "node", "expectedCurrentNodeId": ""},
    ],
)
async def test_navigation_validates_node_and_expected_cursor_before_opening_a_session(params):
    adapter = CodingACPAdapter(lambda: pytest.fail("invalid navigation must not create a model"))
    with pytest.raises(RequestError) as failure:
        await adapter.ext_method("aster/session/navigate", {"sessionId": "unknown", **params})
    assert failure.value.code == -32602
