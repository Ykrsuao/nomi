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


@pytest.mark.skipif(sys.platform != "win32", reason="native PowerShell")
@pytest.mark.parametrize("executable", ["pwsh", "powershell.exe"])
async def test_native_shell_user_variables_cannot_change_control_state(
    tmp_path, monkeypatch, executable
):
    import shutil

    from nooa.tools import _powershell_session

    path = shutil.which(executable)
    if path is None:
        pytest.skip(f"{executable} is not installed")
    monkeypatch.setattr(_powershell_session, "powershell_executable", lambda: path)
    target = tmp_path / "keep-me.txt"
    target.write_text("keep this file", encoding="utf-8")
    shell = _powershell_session.PowerShellSession(cwd=tmp_path)
    try:
        quoted = str(target).replace("'", "''")
        result = await shell.run(f"$inputFile = '{quoted}'; Write-Output 'assigned'")
        assert result == ("assigned\r\n", "", 0)
        assert target.read_text(encoding="utf-8") == "keep this file"
        result = await shell.run(
            "$rc = 7; $control = 'user control'; $request = 'user request'; "
            "$commandReader = 'user reader'; $inputReader = 'user input'; "
            "$inputStream = 'user stream'; $originalInput = 'user handle'; "
            "function Get-ReviewValue { $rc }; Write-Output 'stored'"
        )
        assert result == ("stored\r\n", "", 0)
        result = await shell.run("Get-ReviewValue; Write-Output $control; Test-Path $inputFile")
        assert result[0].splitlines() == ["7", "user control", "True"]
        assert result[1:] == ("", 0)
        failed = await shell.run("Write-Error 'expected failure'")
        assert failed[2] == 1 and "expected failure" in failed[1]
        native = await shell.run(f"& '{sys.executable}' -c 'import sys; sys.exit(7)'")
        assert native[2] == 7, native
        assert await shell.run("Write-Output 'recovered'") == ("recovered\r\n", "", 0)
    finally:
        await shell.close()


@pytest.mark.skipif(sys.platform != "win32", reason="native PowerShell")
@pytest.mark.parametrize("executable", ["pwsh", "powershell.exe"])
@pytest.mark.parametrize("returncode", [0, 7])
async def test_native_shell_preserves_native_exit_code_with_stderr(
    tmp_path, monkeypatch, executable, returncode
):
    import shutil

    from nooa.tools import _powershell_session

    path = shutil.which(executable)
    if path is None:
        pytest.skip(f"{executable} is not installed")
    monkeypatch.setattr(_powershell_session, "powershell_executable", lambda: path)
    shell = _powershell_session.PowerShellSession(cwd=tmp_path)
    try:
        python = str(sys.executable).replace("'", "''")
        result = await shell.run(
            f"& '{python}' -c 'import sys; print(42); "
            f"print(43, file=sys.stderr); sys.exit({returncode})'; Write-Output 'continued'"
        )
        assert result[0].splitlines() == ["42", "continued"]
        assert "43" in result[1]
        assert result[2] == returncode, result
        assert await shell.run("Write-Output 'recovered'") == ("recovered\r\n", "", 0)
    finally:
        await shell.close()


@pytest.mark.skipif(sys.platform != "win32", reason="native PowerShell")
async def test_native_search_anchors_handle_single_quotes(tmp_path):
    from nooa.tools.shell_tools import ShellTools

    target = tmp_path / "a'b.txt"
    target.write_text("quoted\n", encoding="utf-8")
    shell = ShellTools(cwd=str(tmp_path))
    try:
        result = await shell.run('rg -n quoted "a\'b.txt"')
        assert result.returncode == 0
        assert result.matches is not None and len(result.matches) == 1
        await shell.replace(result.matches[0], "updated\n")
        assert target.read_text(encoding="utf-8") == "updated\n"
    finally:
        await shell.close()
