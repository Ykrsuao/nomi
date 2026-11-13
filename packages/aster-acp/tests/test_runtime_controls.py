# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution approvals and live model changes exercise actual session state."""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from acp.schema import RequestPermissionResponse
from aster_acp.runtime_controls import RuntimeControls
from aster_cli.coding import CodingAgent

from aster.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASTER_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("ASTER_PROJECT_DIR", str(tmp_path / "project"))
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.delenv("ASTER_SETTINGS", raising=False)


class PermissionClient:
    def __init__(self, choices: list[str]) -> None:
        self.choices = list(choices)
        self.requests: list[dict[str, Any]] = []
        self.started = asyncio.Event()
        self.release: asyncio.Event | None = None

    async def request_permission(self, session_id, tool_call, options, **kwargs):
        self.requests.append({"session": session_id, "tool": tool_call, "options": options})
        self.started.set()
        if self.release is not None:
            await self.release.wait()
        choice = self.choices.pop(0)
        if choice == "error":
            raise ConnectionError("transport closed")
        outcome = (
            {"outcome": "cancelled"}
            if choice == "cancelled"
            else {"outcome": "selected", "optionId": choice}
        )
        return RequestPermissionResponse.model_validate({"outcome": outcome})


def controls_for(agent, choices, **kwargs):
    client = PermissionClient(choices)
    updates = []
    controls = RuntimeControls(
        agent,
        client,
        "s1",
        publish=updates.append,
        model_names=lambda: ["alpha", "beta"],
        **kwargs,
    )
    agent._execution_control = controls
    return controls, client, updates


async def test_allow_once_requests_each_cell_and_session_grant_is_live_only(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    controls, client, updates = controls_for(agent, ["allow_once", "allow_session"])
    try:
        assert await controls.authorize("print('one')", tool_call_id="one") is None
        assert await controls.authorize("print('two')", tool_call_id="two") is None
        assert await controls.authorize("print('three')") is None
        assert len(client.requests) == 2
        assert controls.status()["approval"] == "session"
        assert client.requests[0]["tool"].raw_input == {"code": "print('one')"}
        assert client.requests[0]["tool"].tool_call_id == "one"
        assert [option.kind for option in client.requests[0]["options"]] == [
            "allow_once",
            "allow_always",
            "reject_once",
        ]
        assert updates[-1].field_meta["aster"]["runtime"]["approvalPending"] is False
        replacement, replacement_client, _ = controls_for(agent, ["reject"])
        assert await replacement.authorize("print('four')") is not None
        assert len(replacement_client.requests) == 1
    finally:
        await agent.close()


async def test_approval_preview_shows_command_and_preserves_complete_python_cell(tmp_path):
    code = (
        "# Inspect the selected workspace.\n"
        'result = await self.shell.run("Get-Location", timeout=20)\n'
        "print(result)"
    )
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    controls, client, _ = controls_for(agent, ["reject"])
    try:
        assert "not run" in (await controls.authorize(code))
        tool = client.requests[0]["tool"]
        preview = tool.field_meta["aster"]["approvalPreview"]
        assert preview["summary"] == "Run shell command"
        assert "Command: Get-Location" in preview["details"]
        assert "Timeout: 20 seconds" in preview["details"]
        assert preview["complete"] is True
        assert tool.title == "Run shell command?"
        assert tool.raw_input == {"code": code}
        assert tool.content[0].content.text == code
        assert tool.field_meta["aster"]["approvalScope"] == "python_execution"
        assert client.requests[0]["options"][1].name == "Allow all code for this session"
        assert controls.approval == "ask"
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "code,summary,details",
    [
        (
            'r = await self.shell.run("cd && ls -a | head -20", timeout=20)\nprint(r)',
            "Run shell command",
            ["Command: cd && ls -a | head -20", "Timeout: 20 seconds"],
        ),
        (
            'await self.shell.run(command="python -", stdin="print(42)")',
            "Run shell command",
            ["Command: python -", "Standard input:\nprint(42)"],
        ),
        (
            'result = await self.shell.read(path="src/app.py", lines=(3, 8))\npprint(result)',
            "Read file",
            ["Read file: src/app.py", "Lines: 3–8"],
        ),
        (
            'await self.shell.write_file("src/app.py", "print(42)\\n")',
            "Write file",
            ["Write file: src/app.py", "Full file content:\nprint(42)\n"],
        ),
        (
            'await self.shell.replace("src/app.py", "before", new="after")',
            "Edit file",
            ["Edit file: src/app.py", "Find:\nbefore", "Replace with:\nafter"],
        ),
        (
            'await self.shell.read("a.py")\nawait self.shell.read("b.py")',
            "Run 2 operations",
            ["Read file: a.py", "Read file: b.py"],
        ),
    ],
)
async def test_approval_preview_represents_literal_operations(tmp_path, code, summary, details):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    controls, client, _ = controls_for(agent, ["reject"])
    try:
        assert await controls.authorize(code) is not None
        preview = client.requests[0]["tool"].field_meta["aster"]["approvalPreview"]
        assert preview == {"summary": summary, "details": details, "complete": True}
        assert client.requests[0]["tool"].raw_input == {"code": code}
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "code",
    [
        "await self.shell.run(command)",
        'await self.shell.run(f"Get-Content {path}")',
        'await self.shell.run("Get-Location", timeout=choose_timeout())',
        'await self.shell.run("Get-Location", **options)',
        'await self.shell.run("Get-Location", unexpected=True)',
        'await self.shell.run("Get-Location")\n__import__("os").system("whoami")',
        'await self.shell.write_file("app.py", generate_content())',
        'await self.shell.replace(match, "new text")',
        'await self.shell.replace("app.py", "old", new=render())',
        'def deferred():\n    self.shell.run("Get-Location")',
        'self.shell = replacement\nawait self.shell.run("Get-Location")',
        'self = await self.shell.read("app.py")',
        'print = await self.shell.read("app.py")',
        'self.shell.run("Get-Location")',
        'result = await self.shell.run("Get-Location")\nprint(result, file=target)',
        'print("Hello!")',
        "await self.shell.run(",
        "pass\n" * 4100,
        f"await self.shell.run({('echo ' + 'x' * 2000)!r})",
        f"await self.shell.write_file('app.py', {('x' * 2000)!r})",
    ],
)
async def test_partial_or_unknown_approval_preview_requires_full_code(tmp_path, code):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    controls, client, _ = controls_for(agent, ["reject"])
    try:
        assert await controls.authorize(code) is not None
        tool = client.requests[0]["tool"]
        preview = tool.field_meta["aster"]["approvalPreview"]
        assert preview["complete"] is False
        assert preview["summary"] == "Run Python code"
        assert tool.raw_input == {"code": code}
        assert controls.approval == "ask"
    finally:
        await agent.close()


async def test_approval_preview_never_executes_code_and_bounds_only_the_preview(tmp_path):
    target = tmp_path / "preview-must-not-execute.txt"
    code = f"from pathlib import Path\nPath({str(target)!r}).write_text('bad')\n" + "\n".join(
        f"await self.shell.run({('x' * 4000 + str(i))!r})" for i in range(10)
    )
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    controls, client, _ = controls_for(agent, ["reject"])
    try:
        assert await controls.authorize(code) is not None
        tool = client.requests[0]["tool"]
        preview = tool.field_meta["aster"]["approvalPreview"]
        assert not target.exists()
        assert preview["complete"] is False
        assert len(preview["details"]) <= 8
        assert sum(map(len, preview["details"])) <= 6000
        assert all(len(detail) <= 1500 for detail in preview["details"])
        assert "full-code review" in preview["details"][-1]
        assert tool.raw_input == {"code": code}
        assert tool.content[0].content.text == code
    finally:
        await agent.close()


@pytest.mark.parametrize("choice", ["reject", "cancelled", "unknown", "error"])
async def test_rejected_or_missing_approval_never_allows_execution(tmp_path, choice):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    controls, _, _ = controls_for(agent, [choice])
    try:
        reason = await controls.authorize("dangerous()")
        assert reason and "not run" in reason
        assert controls.approval == "ask"
    finally:
        await agent.close()


async def test_cancelling_an_approval_unwinds_and_does_not_grant_permission(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    controls, client, _ = controls_for(agent, ["allow_session"])
    client.release = asyncio.Event()
    operation = asyncio.create_task(controls.authorize("work()"))
    try:
        await asyncio.wait_for(client.started.wait(), 2)
        assert controls.status()["approvalPending"]
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation
        assert not controls.status()["approvalPending"]
        assert controls.approval == "ask"
    finally:
        client.release.set()
        await agent.close()


async def test_plan_mode_disables_code_without_asking_and_revokes_session_grant(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    controls, client, _ = controls_for(agent, [])
    try:
        await controls.configure(approval="session")
        await controls.configure(mode="plan")
        assert "not run" in (await controls.authorize("open('x', 'w')"))
        assert not client.requests
        assert controls.status()["executionEnabled"] is False
        await controls.configure(mode="code")
        assert controls.approval == "ask"
        with pytest.raises(ValueError, match="Code mode"):
            await controls.configure(mode="plan", approval="session")
        assert controls.mode == "code"
    finally:
        await agent.close()


async def test_model_switch_changes_real_agent_and_closes_previous_client(tmp_path):
    class ClosingModel(FakeLLMClient):
        def __init__(self, name):
            super().__init__()
            self.model = name
            self.closed = False

        async def aclose(self):
            self.closed = True
            await super().aclose()

    previous = ClosingModel("alpha")
    candidate = ClosingModel("beta")
    agent = CodingAgent(llm=previous, cwd=tmp_path)
    controls, _, _ = controls_for(agent, [], model_factory=lambda name: candidate)
    try:
        result = await controls.configure(model="beta", mode="plan")
        assert agent.llm is candidate
        assert previous.closed
        assert result["model"] == "beta"
        assert result["mode"] == "plan"
        with pytest.raises(ValueError, match="configured"):
            await controls.configure(model="unknown", mode="code")
        assert agent.llm is candidate
        assert controls.mode == "plan"
    finally:
        await agent.close()


async def test_runtime_change_rejected_while_background_work_is_active(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    controls, _, _ = controls_for(agent, [])
    running = asyncio.Event()

    async def job():
        running.set()
        await asyncio.Event().wait()

    agent.queue_manager.queue("check")
    handle = agent.queue_manager.spawn(job(), channel="check")
    try:
        await asyncio.wait_for(running.wait(), 2)
        with pytest.raises(ValueError, match="background jobs"):
            await controls.configure(mode="plan")
        assert controls.mode == "code"
        await handle.cancel()
        await controls.configure(mode="plan")
        assert controls.mode == "plan"
    finally:
        await agent.close()


@pytest.mark.parametrize("change", [{"model": "beta"}, {"mode": "plan"}, {"approval": "session"}])
async def test_runtime_change_waits_for_older_job_after_newest_on_same_channel_finishes(
    tmp_path: Path, change: dict[str, str]
) -> None:
    closed = asyncio.Event()

    class OriginalModel(FakeLLMClient):
        async def aclose(self) -> None:
            closed.set()
            await super().aclose()

    original = OriginalModel()
    agent = CodingAgent(llm=original, cwd=tmp_path)
    controls, _, _ = controls_for(agent, [], model_factory=lambda name: FakeLLMClient())
    started = asyncio.Event()

    async def long_job() -> None:
        started.set()
        await asyncio.Event().wait()

    async def short_job() -> str:
        return "finished"

    channel = agent.queue_manager.queue("checks")
    first = agent.queue_manager.spawn(long_job(), channel="checks")
    try:
        await asyncio.wait_for(started.wait(), 2)
        newest = agent.queue_manager.spawn(short_job(), channel="checks")
        assert await asyncio.wait_for(channel.get(), 2) == "finished"
        assert newest.state == "done"
        assert first.state == "running"
        before = controls.status()

        with pytest.raises(ValueError, match="background jobs"):
            await controls.configure(**change)
        assert controls.status() == before
        assert agent.llm is original
        assert not closed.is_set()

        await first.cancel()
        configured = await controls.configure(**change)
        assert all(configured[key] == value for key, value in change.items())
    finally:
        await agent.close()


async def test_cancelled_model_cleanup_still_publishes_effective_settings(tmp_path):
    closing = asyncio.Event()

    class PreviousModel(FakeLLMClient):
        async def aclose(self):
            closing.set()
            await asyncio.Event().wait()

    agent = CodingAgent(llm=PreviousModel(), cwd=tmp_path)
    replacement = FakeLLMClient()
    controls, _, updates = controls_for(agent, [], model_factory=lambda name: replacement)
    change = asyncio.create_task(controls.configure(model="beta", mode="plan"))
    try:
        await asyncio.wait_for(closing.wait(), 2)
        assert agent.llm is replacement
        state = updates[-1].field_meta["aster"]["runtime"]
        assert state["model"] == "beta"
        assert state["mode"] == "plan"
        change.cancel()
        with pytest.raises(asyncio.CancelledError):
            await change
        assert controls.status()["model"] == state["model"]
    finally:
        change.cancel()
        await agent.close()


def _response(name, arguments, index):
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id=f"call-{index}", name=name, arguments=json.dumps(arguments))],
        finish_reason="tool_calls",
    )


@pytest.mark.parametrize(
    "mode,choice,exists",
    [("plan", "reject", False), ("code", "reject", False), ("code", "allow_once", True)],
)
async def test_real_codeact_cannot_write_without_execution_approval(tmp_path, mode, choice, exists):
    target = tmp_path / "execution.txt"
    llm = FakeLLMClient(
        [
            _response(
                "execute_python",
                {"code": f"from pathlib import Path\nPath({str(target)!r}).write_text('executed')"},
                1,
            ),
            _response(
                "return_result", {"result": {"message": "Finished.", "explanation": "done"}}, 2
            ),
        ],
        strict_exhaustion=True,
    )
    agent = CodingAgent(llm=llm, cwd=tmp_path)
    controls, client, _ = controls_for(agent, [choice])
    try:
        await controls.configure(mode=mode)
        await asyncio.wait_for(agent.handle({"user_messages": ["Perform the requested work"]}), 10)
        assert target.exists() is exists
        assert len(client.requests) == (0 if mode == "plan" else 1)
    finally:
        await agent.close()


async def test_controlled_return_result_does_not_eval_constructor_expressions(tmp_path):
    target = tmp_path / "coercion.txt"
    expression = f"Done(message=str(__import__('pathlib').Path({str(target)!r}).write_text('bad')), explanation='coerced')"
    llm = FakeLLMClient(
        [
            _response("return_result", {"result": expression}, 1),
            _response(
                "return_result", {"result": {"message": "A plan.", "explanation": "planned"}}, 2
            ),
        ],
        strict_exhaustion=True,
    )
    agent = CodingAgent(llm=llm, cwd=tmp_path)
    controls, client, _ = controls_for(agent, [])
    try:
        await controls.configure(mode="plan")
        await asyncio.wait_for(agent.handle({"user_messages": ["Plan the work"]}), 10)
        assert not target.exists()
        assert not client.requests
    finally:
        await agent.close()
