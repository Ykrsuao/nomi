# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native host behavior, without an opt-in flag or a VM runtime."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from nooa import Agent
from nooa.storage.sqlite import (
    SessionAlreadyActiveError,
    SQLiteStorageManager,
    delete_sqlite_database,
)
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


def test_import_and_cli_need_no_platform_override():
    env = {k: v for k, v in os.environ.items() if k != "NOOA_SKIP_PLATFORM_CHECK"}
    for args in (["-c", "import nooa"], ["-m", "nooa_cli", "--help"]):
        result = subprocess.run([sys.executable, *args], env=env, capture_output=True)
        assert result.returncode == 0, result.stderr.decode(errors="replace")


def test_file_storage_lock_releases_on_close(tmp_path):
    path = tmp_path / "会话 with spaces.db"
    first = SQLiteStorageManager(path)
    try:
        with pytest.raises(SessionAlreadyActiveError):
            SQLiteStorageManager(path)
        with pytest.raises(SessionAlreadyActiveError):
            delete_sqlite_database(path)
    finally:
        first.close()
    reopened = SQLiteStorageManager(path, must_exist=True)
    reopened.close()
    assert delete_sqlite_database(path)


async def test_inprocess_state_tools_and_return_result():
    def response(code):
        return LLMResponse(
            raw_response=None,
            content="",
            tool_calls=[
                ToolCall(id=code[:8], name="execute_python", arguments=json.dumps({"code": code}))
            ],
            finish_reason="tool_calls",
        )

    llm = FakeLLMClient(
        scripted_responses=[
            response("values = [1, 2, 3]\ndef bump(v):\n    return v + 1"),
            response("self.record(bump(sum(values)))\nreturn_result(self.answer)"),
        ]
    )

    class Calculator(Agent, llm=llm):
        answer: int = 0

        def record(self, value: int) -> None:
            self.answer = value

        async def calculate(self) -> int:
            """Compute and record the answer."""
            ...

    agent = Calculator()
    assert await agent.calculate() == 7
    assert agent.answer == 7


@pytest.mark.skipif(sys.platform != "win32", reason="native PowerShell")
async def test_native_shell_persists_state_and_handles_timeout(tmp_path):
    from nooa.tools.shell_tools import ShellTools

    shell = ShellTools(cwd=str(tmp_path))
    try:
        assert (await shell.run("$answer = 21; Write-Output '你好' ")).stdout.strip() == "你好"
        assert (await shell.run("Write-Output ($answer * 2)")).stdout.strip() == "42"
        result = await shell.run("Write-Output 'partial'; Start-Sleep -Seconds 20", timeout=0.3)
        assert result.timed_out and result.returncode == 124
        assert "partial" in result.stdout
        assert (await shell.run("Write-Output 'recovered'")).stdout.strip() == "recovered"
    finally:
        await shell.close()


@pytest.mark.skipif(sys.platform != "win32", reason="native PowerShell")
async def test_native_monitor_streams_lines():
    from nooa.runtime.producers import monitor

    assert [line async for line in monitor("Write-Output 'one'; Write-Output '二'")] == [
        "one",
        "二",
    ]


@pytest.mark.skipif(sys.platform != "win32", reason="native PowerShell")
async def test_native_stdin_preserves_bytes_and_temporary_cwd(tmp_path):
    from nooa.tools.shell_tools import ShellTools

    shell = ShellTools(cwd=str(tmp_path))
    child = tmp_path / "child with spaces"
    child.mkdir()
    try:
        command = f"& '{sys.executable}' -c 'import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())'"
        result = await shell.run(command, stdin="first\nsecond", cwd=str(child))
        assert result.stdout == "first\nsecond"
        assert shell.cwd == tmp_path
    finally:
        await shell.close()
