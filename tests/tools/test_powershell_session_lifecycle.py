# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PowerShell command serialization across restarts and lifecycle operations."""

from __future__ import annotations

import asyncio
import sys

import pytest

from aster.tools._powershell_session import PowerShellSession


@pytest.mark.parametrize("entrypoint", ["run", "run_with_timeout_flag", "run_stream"])
async def test_loop_change_keeps_commands_serialized(tmp_path, monkeypatch, entrypoint):
    """A second caller must wait while the first command restarts its shell."""
    session = PowerShellSession(cwd=tmp_path)
    session._started = True
    session._started_on_loop = object()
    command_entered = asyncio.Event()
    allow_command = asyncio.Event()
    active = peak = 0

    async def execute(command, timeout, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        command_entered.set()
        await allow_command.wait()
        session._started_on_loop = asyncio.get_running_loop()
        active -= 1
        return command, "", 0, False

    async def invoke(command):
        if entrypoint == "run_stream":
            return [chunk async for chunk in session.run_stream(command)]
        return await getattr(session, entrypoint)(command)

    monkeypatch.setattr(session, "_run", execute)
    first = asyncio.create_task(invoke("first"))
    await asyncio.wait_for(command_entered.wait(), timeout=1)
    second = asyncio.create_task(invoke("second"))
    try:
        # run_stream schedules its command task one turn after taking the lock.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        entered_concurrently = peak
    finally:
        allow_command.set()
        results = await asyncio.wait_for(
            asyncio.gather(first, second, return_exceptions=True), timeout=1
        )

    assert entered_concurrently == 1
    if entrypoint == "run_stream":
        assert results == [[("__done__", "0,0")], [("__done__", "0,0")]]
    elif entrypoint == "run_with_timeout_flag":
        assert results == [("first", "", 0, False), ("second", "", 0, False)]
    else:
        assert results == [("first", "", 0), ("second", "", 0)]


@pytest.mark.skipif(sys.platform != "win32", reason="native PowerShell")
@pytest.mark.parametrize("operation", ["start", "reset", "close"])
@pytest.mark.parametrize("streaming", [False, True], ids=["run", "run_stream"])
async def test_lifecycle_waits_for_running_command(tmp_path, monkeypatch, operation, streaming):
    """Public lifecycle operations cannot replace or stop an active command."""
    session = PowerShellSession(cwd=tmp_path)
    await session.start()
    process = session._process
    reader = session._control_reader
    read_response = reader.readline
    command_entered = asyncio.Event()
    allow_response = asyncio.Event()

    async def readline():
        command_entered.set()
        await allow_response.wait()
        return await read_response()

    async def execute():
        if streaming:
            return [chunk async for chunk in session.run_stream("Write-Output 'first'")]
        return await session.run("Write-Output 'first'")

    monkeypatch.setattr(reader, "readline", readline)
    first = asyncio.create_task(execute())
    await asyncio.wait_for(command_entered.wait(), timeout=3)
    lifecycle = asyncio.create_task(getattr(session, operation)())
    try:
        await asyncio.sleep(0)
        preserved_process = session._process is process
        lifecycle_waited = not lifecycle.done()
    finally:
        allow_response.set()
        results = await asyncio.wait_for(
            asyncio.gather(first, lifecycle, return_exceptions=True), timeout=10
        )
        await session.close()

    assert preserved_process
    assert lifecycle_waited
    assert results[1] is None
    if streaming:
        assert results[0][-1] == ("__done__", "0,0")
        assert "".join(chunk for name, chunk in results[0] if name == "stdout") == "first\r\n"
        assert not any(name == "stderr" for name, _ in results[0])
    else:
        assert results[0] == ("first\r\n", "", 0)


@pytest.mark.skipif(sys.platform != "win32", reason="native PowerShell")
async def test_startup_command_and_reset_do_not_reenter_lock(tmp_path):
    session = PowerShellSession(cwd=tmp_path, init_command="$answer = 42")
    try:
        for _ in range(2):
            assert await asyncio.wait_for(session.run("Write-Output $answer"), timeout=10) == (
                "42\r\n",
                "",
                0,
            )
            await asyncio.wait_for(session.reset(), timeout=10)
    finally:
        await session.close()


@pytest.mark.skipif(sys.platform != "win32", reason="native PowerShell")
async def test_failed_startup_command_cleans_up_without_reentering_lock(tmp_path):
    session = PowerShellSession(cwd=tmp_path, init_command="Write-Error 'setup failed'")
    try:
        with pytest.raises(RuntimeError, match="Shell init command failed"):
            await asyncio.wait_for(session.run("Write-Output 'unreachable'"), timeout=10)
        assert session._process is None
        assert session._job is None
        session._init_command = "$answer = 42"
        assert await asyncio.wait_for(session.run("Write-Output $answer"), timeout=10) == (
            "42\r\n",
            "",
            0,
        )
    finally:
        await session.close()
