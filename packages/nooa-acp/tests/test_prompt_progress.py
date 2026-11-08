# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt progress and delivery timing through the real ACP adapter."""

import asyncio
import json
import os
from collections import defaultdict
from typing import Any

import pytest
from acp import RequestError, text_block
from acp.schema import AgentMessageChunk, ToolCallProgress, ToolCallStart, UserMessageChunk
from nooa_acp.server import CodingACPAdapter

from nooa.context_blocks.events import ToolCallEvent
from nooa.errors import GenerationError
from nooa.strategies.codeact import OUTPUT_TOKENS_EXHAUSTED_MESSAGE
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

_TIMEOUT = 15
_TIMING_PREFIX = "[nooa-prompt] "
_PRIVATE_PROMPT = "PRIVATE_USER_INPUT_952afa"
_PRIVATE_ANSWER = "PRIVATE_ANSWER_f27c14"


def _reply_llm(*answers: str) -> FakeLLMClient:
    responses = []
    for index, answer in enumerate(answers):
        code = f"self.message({answer!r})\nreturn_result(Done(explanation='complete'))"
        responses.append(
            LLMResponse(
                raw_response=None,
                content="",
                tool_calls=[
                    ToolCall(
                        id=f"fake-reply-{index}",
                        name="execute_python",
                        arguments=json.dumps({"code": code}),
                    )
                ],
                finish_reason="tool_calls",
                reasoning=None,
                usage=None,
            )
        )
    return FakeLLMClient(responses, strict_exhaustion=True)


class _Gate:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def wait(self) -> None:
        self.entered.set()
        await self.release.wait()


def _hold_model(monkeypatch, llm: FakeLLMClient, gate: _Gate) -> None:
    original = llm.acall

    async def acall(*args, **kwargs):
        await gate.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(llm, "acall", acall)


def _is_progress(update: Any) -> bool:
    return isinstance(update, ToolCallStart | ToolCallProgress) and update.tool_call_id.startswith(
        "nooa-progress-"
    )


class _RecordingClient:
    def __init__(self, *, answer_gate: _Gate | None = None, reject_progress: bool = False) -> None:
        self.updates: list[tuple[str, Any]] = []
        self.rejected_progress = 0
        self._answer_gate = answer_gate
        self._reject_progress = reject_progress
        self._changed = asyncio.Condition()

    async def session_update(self, session_id: str, update: Any, **kwargs) -> None:
        if self._reject_progress and _is_progress(update):
            self.rejected_progress += 1
            raise RuntimeError("Client declined optional activity")
        if self._answer_gate is not None and isinstance(update, AgentMessageChunk):
            await self._answer_gate.wait()
        async with self._changed:
            self.updates.append((session_id, update))
            self._changed.notify_all()

    async def wait_for_title(self, title: str, session_id: str) -> None:
        async with asyncio.timeout(_TIMEOUT), self._changed:
            await self._changed.wait_for(
                lambda: any(
                    sid == session_id and _is_progress(update) and update.title == title
                    for sid, update in self.updates
                )
            )

    def messages(self, session_id: str) -> list[str]:
        return [
            update.content.text
            for sid, update in self.updates
            if sid == session_id and isinstance(update, AgentMessageChunk)
        ]

    def progress(self, session_id: str) -> list[Any]:
        return [
            update for sid, update in self.updates if sid == session_id and _is_progress(update)
        ]


class _Harness:
    def __init__(self) -> None:
        self.adapters: list[CodingACPAdapter] = []
        self.tasks: list[asyncio.Task] = []
        self.gates: list[_Gate] = []

    def connect(self, llm_factory, client: _RecordingClient) -> CodingACPAdapter:
        adapter = CodingACPAdapter(llm_factory)
        self.adapters.append(adapter)
        adapter.on_connect(client)  # type: ignore[arg-type]
        return adapter

    def prompt(self, adapter: CodingACPAdapter, session_id: str, text: str) -> asyncio.Task:
        task = asyncio.create_task(adapter.prompt(session_id, [text_block(text)]))
        self.tasks.append(task)
        return task

    def gate(self) -> _Gate:
        gate = _Gate()
        self.gates.append(gate)
        return gate

    async def close(self) -> None:
        # Release transport/model gates even when an assertion failed before the
        # normal release. Then cancel prompts before closing their session stores.
        for gate in self.gates:
            gate.release.set()
        for task in self.tasks:
            if not task.done():
                task.cancel()
        try:
            async with asyncio.timeout(_TIMEOUT):
                await asyncio.gather(*self.tasks, return_exceptions=True)
        finally:
            for adapter in reversed(self.adapters):
                async with asyncio.timeout(_TIMEOUT):
                    await adapter.close()


@pytest.fixture
async def acp(tmp_path, monkeypatch):
    """Keep settings, skills and durable sessions inside temporary directories."""
    isolated_home = tmp_path / "isolated-home"
    user_config = isolated_home / "user-config"
    user_config.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(isolated_home))
    monkeypatch.setenv("USERPROFILE", str(isolated_home))
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user_config))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "project-config"))
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    for name in ("NEMO_OO_SETTINGS", "NOOA_PROMPT_TIMING", "NOOA_STARTUP_TIMING"):
        monkeypatch.delenv(name, raising=False)
    harness = _Harness()
    try:
        yield harness
    finally:
        await harness.close()


def _timings(stderr: str) -> list[dict[str, Any]]:
    return [
        json.loads(line.removeprefix(_TIMING_PREFIX))
        for line in stderr.splitlines()
        if line.startswith(_TIMING_PREFIX)
    ]


def _stages(records: list[dict[str, Any]]) -> list[str]:
    return [record["stage"] for record in records]


async def _wait(event: asyncio.Event) -> None:
    await asyncio.wait_for(event.wait(), timeout=_TIMEOUT)


@pytest.mark.parametrize("timing_env", ["NOOA_PROMPT_TIMING", "NOOA_STARTUP_TIMING"])
async def test_waiting_status_precedes_model_result_and_is_not_replayed(
    acp, tmp_path, monkeypatch, capsys, timing_env
):
    monkeypatch.setenv(timing_env, "1")
    model_gate = acp.gate()
    llm = _reply_llm(_PRIVATE_ANSWER)
    _hold_model(monkeypatch, llm, model_gate)
    client = _RecordingClient()
    adapter = acp.connect(lambda: llm, client)
    created = await adapter.new_session(str(tmp_path))
    prompt = acp.prompt(adapter, created.session_id, _PRIVATE_PROMPT)

    await _wait(model_gate.entered)
    await client.wait_for_title("Waiting for model", created.session_id)
    before = capsys.readouterr()
    before_stages = _stages(_timings(before.err))
    assert not prompt.done()
    assert client.messages(created.session_id) == []
    assert "first_status" in before_stages
    assert "model_start" in before_stages
    assert "first_model_response" not in before_stages
    assert "first_answer" not in before_stages

    model_gate.release.set()
    response = await asyncio.wait_for(prompt, timeout=_TIMEOUT)
    after = capsys.readouterr()
    stages = _stages(_timings(before.err + after.err))
    assert response.stop_reason == "end_turn"
    assert client.messages(created.session_id) == [_PRIVATE_ANSWER]
    assert stages.count("first_model_response") == stages.count("first_answer") == 1
    assert stages.index("model_start") < stages.index("first_model_response")
    assert stages.index("first_status") < stages.index("first_answer") < stages.index("complete")
    assert _TIMING_PREFIX not in before.out + after.out

    progress = client.progress(created.session_id)
    assert sum(isinstance(update, ToolCallStart) for update in progress) == 1
    assert progress[0].title == "Preparing response"
    assert progress[0].kind == "other"
    assert len({update.tool_call_id for update in progress}) == 1
    terminal = [update for update in progress if update.status in {"completed", "failed"}]
    assert [(update.title, update.status) for update in terminal] == [
        ("Response ready", "completed")
    ]
    assert all(update.raw_input is None and not update.content for update in progress)
    delivered = [update for sid, update in client.updates if sid == created.session_id]
    answer_index = next(
        i for i, update in enumerate(delivered) if isinstance(update, AgentMessageChunk)
    )
    assert delivered.index(terminal[0]) < answer_index

    await adapter.close_session(created.session_id)
    replay = _RecordingClient()
    restored = acp.connect(lambda: _reply_llm("unused"), replay)
    await restored.load_session(str(tmp_path), created.session_id)
    assert replay.messages(created.session_id) == [_PRIVATE_ANSWER + "\n"]
    assert [
        update.content.text for _, update in replay.updates if isinstance(update, UserMessageChunk)
    ] == [_PRIVATE_PROMPT + "\n"]
    assert replay.progress(created.session_id) == []


async def test_first_answer_waits_for_successful_client_delivery(
    acp, tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("NOOA_PROMPT_TIMING", "1")
    answer_gate = acp.gate()
    client = _RecordingClient(answer_gate=answer_gate)
    adapter = acp.connect(lambda: _reply_llm(_PRIVATE_ANSWER), client)
    created = await adapter.new_session(str(tmp_path))
    prompt = acp.prompt(adapter, created.session_id, _PRIVATE_PROMPT)

    await _wait(answer_gate.entered)
    before = capsys.readouterr()
    stages = _stages(_timings(before.err))
    assert "first_model_response" in stages
    assert "first_status" in stages
    assert "first_answer" not in stages
    assert "complete" not in stages
    assert client.messages(created.session_id) == []
    assert not prompt.done()

    answer_gate.release.set()
    response = await asyncio.wait_for(prompt, timeout=_TIMEOUT)
    stages = _stages(_timings(before.err + capsys.readouterr().err))
    assert response.stop_reason == "end_turn"
    assert client.messages(created.session_id) == [_PRIVATE_ANSWER]
    assert stages.count("first_answer") == 1
    assert (
        stages.index("first_model_response")
        < stages.index("first_answer")
        < stages.index("complete")
    )


async def test_cancelled_wait_does_not_count_stop_note_and_next_prompt_works(
    acp, tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("NOOA_PROMPT_TIMING", "1")
    model_gate = acp.gate()
    llm = _reply_llm("Recovered answer")
    _hold_model(monkeypatch, llm, model_gate)
    client = _RecordingClient()
    adapter = acp.connect(lambda: llm, client)
    created = await adapter.new_session(str(tmp_path))
    prompt = acp.prompt(adapter, created.session_id, "cancel this wait")
    await _wait(model_gate.entered)
    await client.wait_for_title("Waiting for model", created.session_id)

    await asyncio.wait_for(adapter.cancel(created.session_id), timeout=_TIMEOUT)
    cancelled = await asyncio.wait_for(prompt, timeout=_TIMEOUT)
    cancelled_records = _timings(capsys.readouterr().err)
    assert cancelled.stop_reason == "cancelled"
    assert client.messages(created.session_id) == ["Stopped at your request."]
    assert "first_answer" not in _stages(cancelled_records)
    assert "first_model_response" not in _stages(cancelled_records)
    assert cancelled_records[-1]["stage"] == "complete"
    assert cancelled_records[-1]["outcome"] == "cancelled"
    terminal = [
        update for update in client.progress(created.session_id) if update.status == "failed"
    ]
    assert [(update.title, update.status) for update in terminal] == [("Cancelled", "failed")]

    model_gate.release.set()
    resumed = await asyncio.wait_for(
        acp.prompt(adapter, created.session_id, "try again"), timeout=_TIMEOUT
    )
    resumed_records = _timings(capsys.readouterr().err)
    assert resumed.stop_reason == "end_turn"
    assert client.messages(created.session_id) == ["Stopped at your request.", "Recovered answer"]
    assert _stages(resumed_records).count("first_answer") == 1
    assert {record["turn_id"] for record in cancelled_records}.isdisjoint(
        record["turn_id"] for record in resumed_records
    )
    assert resumed_records[-1]["outcome"] == "end_turn"


async def test_rejected_activity_does_not_discard_answer_or_report_first_status(
    acp, tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("NOOA_PROMPT_TIMING", "1")
    client = _RecordingClient(reject_progress=True)
    adapter = acp.connect(lambda: _reply_llm(_PRIVATE_ANSWER), client)
    created = await adapter.new_session(str(tmp_path))
    response = await asyncio.wait_for(
        acp.prompt(adapter, created.session_id, _PRIVATE_PROMPT), timeout=_TIMEOUT
    )

    records = _timings(capsys.readouterr().err)
    assert client.rejected_progress > 0
    assert response.stop_reason == "end_turn"
    assert client.messages(created.session_id) == [_PRIVATE_ANSWER]
    assert "first_status" not in _stages(records)
    assert _stages(records).count("first_answer") == 1
    assert records[-1]["stage"] == "complete"
    assert records[-1]["outcome"] == "end_turn"


@pytest.mark.parametrize(
    ("fault", "expected_error", "outcome", "title"),
    [
        pytest.param(
            RuntimeError("Dispatcher unavailable"),
            RuntimeError,
            "failed",
            "Response failed",
            id="ordinary-error",
        ),
        pytest.param(
            GenerationError("Generation unavailable"),
            RequestError,
            "failed",
            "Response failed",
            id="generation-error",
        ),
        pytest.param(
            GenerationError(OUTPUT_TOKENS_EXHAUSTED_MESSAGE),
            None,
            "max_tokens",
            "Stopped at generation limit",
            id="output-token-limit",
        ),
    ],
)
async def test_failure_before_answer_finishes_activity_and_allows_next_prompt(
    acp, tmp_path, monkeypatch, capsys, fault, expected_error, outcome, title
):
    monkeypatch.setenv("NOOA_PROMPT_TIMING", "1")
    client = _RecordingClient()
    adapter = acp.connect(lambda: _reply_llm("Recovered answer"), client)
    created = await adapter.new_session(str(tmp_path))
    session = (await adapter._sessions.get(created.session_id)).value

    async def fail_submit(text):
        raise fault

    with monkeypatch.context() as fault_patch:
        fault_patch.setattr(session.dispatcher, "submit", fail_submit)
        if expected_error is None:
            response = await asyncio.wait_for(
                acp.prompt(adapter, created.session_id, "fail before answering"), timeout=_TIMEOUT
            )
            assert response.stop_reason == outcome
        else:
            with pytest.raises(expected_error):
                await asyncio.wait_for(
                    acp.prompt(adapter, created.session_id, "fail before answering"),
                    timeout=_TIMEOUT,
                )

    failed_records = _timings(capsys.readouterr().err)
    assert client.messages(created.session_id) == []
    assert "first_answer" not in _stages(failed_records)
    assert "first_model_response" not in _stages(failed_records)
    completed = [record for record in failed_records if record["stage"] == "complete"]
    assert len(completed) == 1 and completed[0]["outcome"] == outcome
    activity = client.progress(created.session_id)
    assert sum(isinstance(update, ToolCallStart) for update in activity) == 1
    failed_id = activity[0].tool_call_id
    terminal = [update for update in activity if update.status in {"completed", "failed"}]
    assert [(update.title, update.status) for update in terminal] == [(title, "failed")]

    response = await asyncio.wait_for(
        acp.prompt(adapter, created.session_id, "try again"), timeout=_TIMEOUT
    )
    resumed_records = _timings(capsys.readouterr().err)
    assert response.stop_reason == "end_turn"
    assert client.messages(created.session_id) == ["Recovered answer"]
    assert _stages(resumed_records).count("first_answer") == 1
    assert resumed_records[-1]["outcome"] == "end_turn"
    assert {record["turn_id"] for record in failed_records}.isdisjoint(
        record["turn_id"] for record in resumed_records
    )
    assert [
        update
        for update in client.progress(created.session_id)
        if update.tool_call_id == failed_id and update.status in {"completed", "failed"}
    ] == terminal


async def test_late_events_from_old_prompt_do_not_change_new_prompt_progress(
    acp, tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("NOOA_PROMPT_TIMING", "1")
    late_gate, model_gate = acp.gate(), acp.gate()
    llm = _reply_llm("First answer", "Second answer")
    client = _RecordingClient()
    adapter = acp.connect(lambda: llm, client)
    created = await adapter.new_session(str(tmp_path))
    session = (await adapter._sessions.get(created.session_id)).value
    original_submit = session.dispatcher.submit
    callbacks = []

    async def late_events():
        await late_gate.wait()
        session.agent.event_manager.add(
            ToolCallEvent(
                tool_call_id="old-prompt-prefill",
                name="execute_python",
                arguments={"code": "pass"},
                metadata={"prefill": True},
            ),
            record=False,
        )
        session.agent.event_manager.add(LLMResponse(content="Late old response"), record=False)

    async def submit_with_late_callback(text):
        # create_task inherits the actual old prompt's context. Its callback
        # survives that prompt finishing while the next prompt owns the session.
        callback = asyncio.create_task(late_events())
        callbacks.append(callback)
        acp.tasks.append(callback)
        return await original_submit(text)

    with monkeypatch.context() as callback_patch:
        callback_patch.setattr(session.dispatcher, "submit", submit_with_late_callback)
        first = await asyncio.wait_for(
            acp.prompt(adapter, created.session_id, "first request"), timeout=_TIMEOUT
        )
    assert first.stop_reason == "end_turn"
    await _wait(late_gate.entered)
    old_records = _timings(capsys.readouterr().err)
    client.updates.clear()

    _hold_model(monkeypatch, llm, model_gate)
    second = acp.prompt(adapter, created.session_id, "second request")
    await _wait(model_gate.entered)
    await client.wait_for_title("Waiting for model", created.session_id)
    before_records = _timings(capsys.readouterr().err)
    before_activity = client.progress(created.session_id)
    assert {record["turn_id"] for record in old_records}.isdisjoint(
        record["turn_id"] for record in before_records
    )
    assert before_activity[-1].title == "Waiting for model"

    late_gate.release.set()
    await asyncio.wait_for(callbacks[0], timeout=_TIMEOUT)
    await session.bridge.flush()
    assert _timings(capsys.readouterr().err) == []
    assert client.progress(created.session_id) == before_activity
    assert "first_model_response" not in _stages(before_records)
    assert client.messages(created.session_id) == []
    assert not second.done()

    model_gate.release.set()
    assert (await asyncio.wait_for(second, timeout=_TIMEOUT)).stop_reason == "end_turn"
    records = before_records + _timings(capsys.readouterr().err)
    assert len({record["turn_id"] for record in records}) == 1
    assert _stages(records).count("first_model_response") == 1
    assert _stages(records).count("first_answer") == 1
    assert records[-1]["outcome"] == "end_turn"
    assert client.messages(created.session_id) == ["Second answer"]


async def test_concurrent_sessions_and_later_turn_have_independent_content_free_timings(
    acp, tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("NOOA_PROMPT_TIMING", "1")
    alpha_gate, beta_gate = acp.gate(), acp.gate()
    alpha_llm = _reply_llm(_PRIVATE_ANSWER, _PRIVATE_ANSWER)
    beta_llm = _reply_llm(_PRIVATE_ANSWER)
    _hold_model(monkeypatch, alpha_llm, alpha_gate)
    _hold_model(monkeypatch, beta_llm, beta_gate)
    llms = iter([alpha_llm, beta_llm])
    client = _RecordingClient()
    adapter = acp.connect(lambda: next(llms), client)
    alpha = await adapter.new_session(str(tmp_path))
    beta = await adapter.new_session(str(tmp_path))

    alpha_task = acp.prompt(adapter, alpha.session_id, _PRIVATE_PROMPT)
    await _wait(alpha_gate.entered)
    await client.wait_for_title("Waiting for model", alpha.session_id)
    records = _timings(capsys.readouterr().err)
    (alpha_id,) = {record["turn_id"] for record in records}

    beta_task = acp.prompt(adapter, beta.session_id, _PRIVATE_PROMPT)
    await _wait(beta_gate.entered)
    await client.wait_for_title("Waiting for model", beta.session_id)
    beta_records = _timings(capsys.readouterr().err)
    (beta_id,) = {record["turn_id"] for record in beta_records}
    assert alpha_id != beta_id
    records.extend(beta_records)

    beta_gate.release.set()
    assert (await asyncio.wait_for(beta_task, timeout=_TIMEOUT)).stop_reason == "end_turn"
    beta_done = _timings(capsys.readouterr().err)
    assert beta_done and {record["turn_id"] for record in beta_done} == {beta_id}
    assert not alpha_task.done()
    records.extend(beta_done)

    alpha_gate.release.set()
    assert (await asyncio.wait_for(alpha_task, timeout=_TIMEOUT)).stop_reason == "end_turn"
    alpha_done = _timings(capsys.readouterr().err)
    assert alpha_done and {record["turn_id"] for record in alpha_done} == {alpha_id}
    records.extend(alpha_done)

    assert (
        await asyncio.wait_for(
            acp.prompt(adapter, alpha.session_id, _PRIVATE_PROMPT), timeout=_TIMEOUT
        )
    ).stop_reason == "end_turn"
    records.extend(_timings(capsys.readouterr().err))
    by_turn = defaultdict(list)
    for record in records:
        by_turn[record["turn_id"]].append(record)
        assert record["pid"] == os.getpid()
        assert record["backend"] == "inprocess"
        assert set(record) <= {"pid", "turn_id", "backend", "stage", "elapsed_s", "call", "outcome"}
    assert len(by_turn) == 3
    for turn in by_turn.values():
        stages = _stages(turn)
        assert stages[0] == "received" and stages[-1] == "complete"
        for stage in (
            "accepted",
            "first_status",
            "first_model_response",
            "first_answer",
            "complete",
        ):
            assert stages.count(stage) == 1
        assert [record["call"] for record in turn if record["stage"] == "model_start"] == [1]
        assert [record["call"] for record in turn if record["stage"] == "model_end"] == [1]
        elapsed = [record["elapsed_s"] for record in turn]
        assert elapsed == sorted(elapsed) and elapsed[0] >= 0
        assert turn[-1]["outcome"] == "end_turn"
    diagnostic_values = "\n".join(str(value) for record in records for value in record.values())
    for private_value in (
        _PRIVATE_PROMPT,
        _PRIVATE_ANSWER,
        alpha.session_id,
        beta.session_id,
        str(tmp_path),
    ):
        assert private_value not in diagnostic_values
    assert client.messages(alpha.session_id) == [_PRIVATE_ANSWER, _PRIVATE_ANSWER]
    assert client.messages(beta.session_id) == [_PRIVATE_ANSWER]


async def test_prompt_diagnostics_are_silent_by_default(acp, tmp_path, capsys):
    client = _RecordingClient()
    adapter = acp.connect(lambda: _reply_llm(_PRIVATE_ANSWER), client)
    created = await adapter.new_session(str(tmp_path))
    response = await asyncio.wait_for(
        acp.prompt(adapter, created.session_id, _PRIVATE_PROMPT), timeout=_TIMEOUT
    )

    captured = capsys.readouterr()
    assert response.stop_reason == "end_turn"
    assert _TIMING_PREFIX not in captured.out + captured.err
    assert client.messages(created.session_id) == [_PRIVATE_ANSWER]
    assert client.progress(created.session_id)
