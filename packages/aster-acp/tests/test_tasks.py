# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task inspector state follows real todos and individual background handles."""

import asyncio
from pathlib import Path

import pytest
from acp.schema import AgentPlanUpdate
from aster_acp._tasks import TaskTracker
from aster_acp.event_bridge import ACPEventBridge
from aster_cli.coding import CodingAgent

from aster.context_blocks.events import ToolCallEvent
from aster.interactive import Waiting
from aster.unifiedllm import FakeLLMClient


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASTER_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("ASTER_PROJECT_DIR", str(tmp_path / "project"))
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.delenv("ASTER_SETTINGS", raising=False)


async def test_plan_snapshot_tracks_active_completed_and_blocked_steps(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    tracker = TaskTracker(agent)
    try:
        first = agent.todo.add("Inspect parser")
        second = agent.todo.add("Fix parser", deps=[first])
        agent.todo.activate(first)
        agent.todo.comment(first, "Located the failing branch.")
        initial = tracker.snapshot("Running tests")
        assert initial["currentTaskId"] == first.id
        assert initial["steps"][0]["status"] == "in_progress"
        assert initial["steps"][1]["blockedBy"] == [first.id]
        agent.todo.complete(first)
        agent.todo.activate(second)
        update = tracker.update("Waiting for model")
        assert [entry.status for entry in update.entries] == ["completed", "in_progress"]
        assert update.field_meta["aster"]["tasks"]["waitReason"] == "Waiting for model"
    finally:
        await agent.close()


async def test_stop_selected_background_job_leaves_other_work_running(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    tracker = TaskTracker(agent)
    one_started, two_started = asyncio.Event(), asyncio.Event()
    one_stopped, two_stopped = asyncio.Event(), asyncio.Event()

    async def run(started, stopped):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    agent.queue_manager.queue("one")
    agent.queue_manager.queue("two")
    first = agent.queue_manager.spawn(run(one_started, one_stopped), channel="one", label="Lint")
    second = agent.queue_manager.spawn(run(two_started, two_stopped), channel="two", label="Tests")
    try:
        await asyncio.wait_for(asyncio.gather(one_started.wait(), two_started.wait()), 2)
        rows = tracker.snapshot()["jobs"]
        selected = next(row for row in rows if row["channel"] == "one")
        assert selected["title"] == "Lint"
        assert await tracker.stop(selected["id"])
        assert one_stopped.is_set()
        assert not two_stopped.is_set()
        assert first.state == "cancelled"
        assert second.state == "running"
        assert await tracker.stop(selected["id"]) is False
    finally:
        await agent.close()


@pytest.mark.parametrize("stop_order", [(0, 1), (1, 0)])
async def test_same_channel_jobs_have_stable_ids_and_stop_independently(
    tmp_path: Path, stop_order: tuple[int, int]
) -> None:
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    tracker = TaskTracker(agent)
    started = [asyncio.Event(), asyncio.Event()]
    stopped = [asyncio.Event(), asyncio.Event()]

    async def run(index: int) -> None:
        started[index].set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped[index].set()

    agent.queue_manager.queue("checks")
    first = agent.queue_manager.spawn(run(0), channel="checks", label="Lint")
    try:
        await asyncio.wait_for(started[0].wait(), 2)
        first_id = tracker.snapshot()["jobs"][0]["id"]
        second = agent.queue_manager.spawn(run(1), channel="checks", label="Tests")
        await asyncio.wait_for(started[1].wait(), 2)
        rows = {row["title"]: row for row in tracker.snapshot()["jobs"]}
        assert set(rows) == {"Lint", "Tests"}
        assert rows["Lint"]["id"] == first_id
        ids = [first_id, rows["Tests"]["id"]]
        assert ids[0] != ids[1]

        assert await tracker.stop(ids[stop_order[0]])
        assert stopped[stop_order[0]].is_set()
        assert not stopped[stop_order[1]].is_set()
        current = {row["id"]: row for row in tracker.snapshot()["jobs"]}
        assert current[ids[stop_order[0]]]["status"] == "cancelled"
        assert current[ids[stop_order[1]]]["canStop"] is True

        assert await tracker.stop(ids[stop_order[1]])
        assert first.state == second.state == "cancelled"
        assert await tracker.stop(ids[0]) is False
        assert await tracker.stop(ids[1]) is False
    finally:
        await agent.close()


async def test_bounded_task_snapshot_keeps_running_jobs_ahead_of_recent_finished_jobs(
    tmp_path: Path,
) -> None:
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    tracker = TaskTracker(agent)
    started = asyncio.Event()

    async def long_job() -> None:
        started.set()
        await asyncio.Event().wait()

    async def finished_job(index: int) -> int:
        return index

    channel = agent.queue_manager.queue("checks")
    agent.queue_manager.spawn(long_job(), channel="checks", label="Long check")
    try:
        await asyncio.wait_for(started.wait(), 2)
        active_id = tracker.snapshot()["jobs"][0]["id"]
        for index in range(105):
            agent.queue_manager.spawn(finished_job(index), channel="checks", label=f"Check {index}")
            assert await asyncio.wait_for(channel.get(), 2) == index

        snapshot = tracker.snapshot()
        assert len(snapshot["jobs"]) == 100
        assert snapshot["truncated"] is True
        assert snapshot["jobs"][0]["id"] == active_id
        assert snapshot["jobs"][0]["canStop"] is True
        titles = {row["title"] for row in snapshot["jobs"]}
        assert "Check 104" in titles
        assert "Check 0" not in titles
        assert await tracker.stop(active_id)
    finally:
        await agent.close()


async def test_stale_job_id_cannot_stop_replacement_on_same_channel(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    tracker = TaskTracker(agent)

    async def run():
        await asyncio.Event().wait()

    agent.queue_manager.queue("checks")
    old = agent.queue_manager.spawn(run(), channel="checks")
    try:
        await asyncio.sleep(0)
        old_id = tracker.snapshot()["jobs"][0]["id"]
        await old.cancel()
        agent.queue_manager.queue("checks", replace=True)
        replacement = agent.queue_manager.spawn(run(), channel="checks")
        await asyncio.sleep(0)
        with pytest.raises(ValueError, match="no longer registered"):
            await tracker.stop(old_id)
        assert replacement.state == "running"
        assert tracker.snapshot()["jobs"][0]["id"] != old_id
    finally:
        await agent.close()


async def test_replacement_output_survives_cancellation_cleanup(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    tracker = TaskTracker(agent)
    started, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    channel = agent.queue_manager.queue("checks")

    async def old_job():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()

    async def replacement_job():
        return "replacement output"

    agent.queue_manager.spawn(old_job(), channel="checks")
    operation = None
    try:
        await asyncio.wait_for(started.wait(), 2)
        old_id = tracker.snapshot()["jobs"][0]["id"]
        operation = asyncio.create_task(tracker.stop(old_id))
        await asyncio.wait_for(cleaning.wait(), 2)
        agent.queue_manager.spawn(replacement_job(), channel="checks")
        await asyncio.sleep(0)
        release.set()
        assert await asyncio.wait_for(operation, 2)
        assert channel.drain() == ["replacement output"]
        assert agent.queue_manager.get_channel("system_messages").drain()
    finally:
        release.set()
        if operation is not None:
            await operation
        await agent.close()


async def test_bridge_emits_structured_plan_and_waiting_reason_without_duplicate_updates(tmp_path):
    class Client:
        def __init__(self):
            self.updates = []

        async def session_update(self, session_id, update, **kwargs):
            self.updates.append(update)

    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = Client()
    bridge = ACPEventBridge(agent, client, "s1", structured_progress=True)
    try:
        first = agent.todo.add("Check import path")
        agent.todo.activate(first)
        bridge.observe_result(Waiting(explanation="Tests are running", on=["checks"]))
        bridge.refresh_tasks()
        await bridge.flush()
        plans = [update for update in client.updates if isinstance(update, AgentPlanUpdate)]
        assert len(plans) == 1
        assert plans[0].entries[0].status == "in_progress"
        assert "Tests are running" in plans[0].field_meta["aster"]["tasks"]["waitReason"]
        assert "checks" in bridge.task_snapshot()["waitReason"]
    finally:
        await bridge.close()
        await agent.close()


async def test_purpose_comment_labels_python_card_without_losing_code(tmp_path):
    class Client:
        def __init__(self):
            self.updates = []

        async def session_update(self, session_id, update, **kwargs):
            self.updates.append(update)

    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = Client()
    bridge = ACPEventBridge(agent, client, "s1")
    code = "# Inspect parser tests\nprint('test_parser.py')"
    try:
        agent.event_manager.add(
            ToolCallEvent(tool_call_id="inspect", name="execute_python", arguments={"code": code})
        )
        await bridge.flush()
        assert client.updates[0].title == "Inspect parser tests"
        assert code in client.updates[0].content[0].content.text
    finally:
        await bridge.close()
        await agent.close()
