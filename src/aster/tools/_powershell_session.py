# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent native PowerShell with a separate, authenticated control pipe."""

from __future__ import annotations

import asyncio
import base64
import codecs
import json
import secrets
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from aster.agentdoc import TruncatingStringIO
from aster.tools._windows_process import WindowsJob, powershell_executable

_BOOTSTRAP = r"""
[Console]::InputEncoding = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$ProgressPreference = 'SilentlyContinue'
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class AsterInputHandle {
    [DllImport("kernel32.dll")] public static extern IntPtr GetStdHandle(int n);
    [DllImport("kernel32.dll")] public static extern bool SetStdHandle(int n, IntPtr h);
    [DllImport("kernel32.dll")] public static extern bool SetHandleInformation(IntPtr h, uint mask, uint flags);
}
'@
$OutputEncoding = [Console]::OutputEncoding
$ProgressPreference = 'SilentlyContinue'
$PSNativeCommandUseErrorActionPreference = $false
$commandReader = [Console]::In
$originalInput = [AsterInputHandle]::GetStdHandle(-10)
$client = [Net.Sockets.TcpClient]::new('127.0.0.1', PORT)
$control = [IO.StreamWriter]::new($client.GetStream(), [Text.UTF8Encoding]::new($false))
$control.AutoFlush = $true
$control.WriteLine('TOKEN')
# Keep persistent user variables in a separate runspace. In particular, an
# ordinary assignment to $inputFile or $control must never alter cleanup/IPC.
$userRunspace = [RunspaceFactory]::CreateRunspace($Host)
$userRunspace.Open()
$userRunspace.SessionStateProxy.Path.SetLocation((Get-Location).Path) | Out-Null
$userRunspace.SessionStateProxy.SetVariable('OutputEncoding', [Console]::OutputEncoding)
$userRunspace.SessionStateProxy.SetVariable('ErrorActionPreference', 'Stop')
$userRunspace.SessionStateProxy.SetVariable('ProgressPreference', 'SilentlyContinue')
$userRunspace.SessionStateProxy.SetVariable('PSNativeCommandUseErrorActionPreference', $false)
while ($null -ne ($line = $commandReader.ReadLine())) {
    $request = ConvertFrom-Json $line
    if ($request.op -eq 'close') { break }
    $script = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($request.code))
    $savedLocation = $userRunspace.SessionStateProxy.Path.CurrentLocation.Path
    $userRunspace.SessionStateProxy.SetVariable('LASTEXITCODE', 0)
    $rc = 0
    $ErrorActionPreference = 'Stop'
    $inputFile = [IO.Path]::GetTempFileName()
    $inputStream = $null
    $inputReader = $null
    $userShell = $null
    try {
        if ($request.cwd) {
            $userRunspace.SessionStateProxy.Path.SetLocation([string]$request.cwd) | Out-Null
        }
        if ($null -ne $request.input) {
            [IO.File]::WriteAllBytes($inputFile, [Convert]::FromBase64String($request.input))
        }
        $inputStream = [IO.File]::Open($inputFile, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
        $handle = $inputStream.SafeFileHandle.DangerousGetHandle()
        [void][AsterInputHandle]::SetHandleInformation($handle, 1, 1)
        [void][AsterInputHandle]::SetStdHandle(-10, $handle)
        $inputReader = [IO.StreamReader]::new($inputStream, [Text.UTF8Encoding]::new($false))
        [Console]::SetIn($inputReader)
        $userShell = [PowerShell]::Create()
        $userShell.Runspace = $userRunspace
        [void]$userShell.AddScript($script, $false)
        # Let Out-Default route native stderr without treating it as a script
        # failure. In Windows PowerShell, the hosted error stream also applies
        # ErrorActionPreference=Stop to native stderr before Invoke returns.
        $userShell.Commands.Commands[0].MergeMyResults(
            [System.Management.Automation.Runspaces.PipelineResultTypes]::Error,
            [System.Management.Automation.Runspaces.PipelineResultTypes]::Output
        )
        [void]$userShell.AddCommand('Out-Default')
        [void]$userShell.Invoke()
        $nativeExitCode = $userRunspace.SessionStateProxy.GetVariable('LASTEXITCODE')
        if ($userShell.Streams.Error.Count) {
            $rc = 1
            foreach ($record in $userShell.Streams.Error) {
                [Console]::Error.WriteLine($record.ToString())
            }
        } elseif ($nativeExitCode) { $rc = $nativeExitCode }
        elseif ($userShell.HadErrors) { $rc = 1 }
    } catch {
        [Console]::Error.WriteLine($_.ToString())
        $rc = 1
    } finally {
        [void][AsterInputHandle]::SetStdHandle(-10, $originalInput)
        [Console]::SetIn($commandReader)
        if ($userShell) { $userShell.Dispose() }
        if ($inputReader) { $inputReader.Dispose() }
        if ($inputStream) { $inputStream.Dispose() }
        [IO.File]::Delete($inputFile)
        if ($request.cwd) {
            $userRunspace.SessionStateProxy.Path.SetLocation($savedLocation) | Out-Null
        }
    }
    $control.WriteLine((@{code=$rc;cwd=$userRunspace.SessionStateProxy.Path.CurrentLocation.Path} | ConvertTo-Json -Compress))
}
$userRunspace.Dispose()
$control.Dispose()
$client.Dispose()
"""


class PowerShellSession:
    """The shell-session interface on Windows; cwd and variables persist.

    Commands and lifecycle operations on one event loop are serialized. A
    session may move to another loop once the previous loop's commands have
    finished; concurrent use across loops or threads is not supported.
    """

    def __init__(
        self,
        cwd: str | Path = ".",
        init_command: str | None = None,
        *,
        keep_background_on_close: bool = False,
    ) -> None:
        self._cwd = Path(cwd).resolve()
        self._init_command = init_command
        self._keep_background_on_close = keep_background_on_close
        self._started = False
        self._started_on_loop: asyncio.AbstractEventLoop | None = None
        self._process: Any = None
        self._job: WindowsJob | None = None
        self._control_reader: Any = None
        self._control_writer: Any = None
        self._readers: list[asyncio.Task[None]] = []
        self._lock = asyncio.Lock()
        self._lock_loop: asyncio.AbstractEventLoop | None = None
        self._events: Any = None
        self._buffers: dict[str, TruncatingStringIO] = {}
        self._start_count = 0
        self._last_command = ""
        from aster.tools.shell_lifecycle import _current_scope

        scope = _current_scope.get()
        if scope:
            scope.adopt(self)

    async def __aenter__(self) -> Any:
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def start(self) -> None:
        self._ensure_lock_on_current_loop()
        async with self._lock:
            await self._start_unlocked()

    async def _start_unlocked(self) -> None:
        """Start or restart the shell while the caller holds the command lock."""
        loop = asyncio.get_running_loop()
        if self._started_on_loop is not None and self._started_on_loop is not loop:
            await self._stop_unlocked(False)
        if self._started:
            return
        self._started_on_loop = loop
        connected = loop.create_future()
        token = secrets.token_hex(24)

        async def accept(reader: Any, writer: Any) -> None:
            try:
                if (await asyncio.wait_for(reader.readline(), 5)).decode().strip() != token:
                    writer.close()
                    return
                if connected.done():
                    writer.close()
                    return
                connected.set_result((reader, writer))
            except Exception:
                writer.close()

        server = await asyncio.start_server(accept, "127.0.0.1", 0, limit=8192)
        port = server.sockets[0].getsockname()[1]
        code = _BOOTSTRAP.replace("PORT", str(port)).replace("TOKEN", token)
        encoded = base64.b64encode(code.encode("utf-16-le")).decode()
        try:
            self._process = await asyncio.create_subprocess_exec(
                powershell_executable(),
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-OutputFormat",
                "Text",
                "-EncodedCommand",
                encoded,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._cwd),
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            self._job = WindowsJob(self._process.pid)
            self._control_reader, self._control_writer = await asyncio.wait_for(connected, 20)
            self._buffers = {
                name: TruncatingStringIO(limit=30_000) for name in ("stdout", "stderr")
            }
            self._readers = [
                asyncio.create_task(self._drain(name, getattr(self._process, name)))
                for name in ("stdout", "stderr")
            ]
            self._started = True
            self._started_on_loop = loop
            self._start_count += 1
        except BaseException:
            await self._stop_unlocked(False)
            raise
        finally:
            server.close()
            # Python 3.13 waits for accepted clients too. The control client
            # belongs to the live session and is closed by _stop_unlocked().
        if self._init_command:
            result = await self._run(self._init_command, 30.0)
            if result[2]:
                await self._stop_unlocked(False)
                raise RuntimeError(f"Shell init command failed: {result[1]}")

    async def _drain(self, name: str, reader: Any) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        while chunk := await reader.read(8192):
            text = decoder.decode(chunk)
            self._buffers[name].write(text)
            if self._events is not None:
                await self._events.put((name, text))

    def _ensure_lock_on_current_loop(self) -> None:
        """Adopt the new loop's lock before awaiting shell startup or cleanup.

        Shell restarts must not replace a lock that a command already holds.
        Commands on the previous loop must have finished before switching.
        """
        loop = asyncio.get_running_loop()
        if self._lock_loop is None:
            self._lock_loop = loop
        elif self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop

    async def run(
        self,
        command: str,
        timeout: float = 30.0,
        *,
        stdin: str | None = None,
        cwd: str | Path | None = None,
    ) -> tuple[str, str, int]:
        out, err, code, _ = await self.run_with_timeout_flag(command, timeout, stdin=stdin, cwd=cwd)
        return out, err, code

    async def run_with_timeout_flag(
        self,
        command: str,
        timeout: float = 30.0,
        *,
        stdin: str | None = None,
        cwd: str | Path | None = None,
    ) -> tuple[str, str, int, bool]:
        self._ensure_lock_on_current_loop()
        async with self._lock:
            return await self._run(command, timeout, stdin=stdin, cwd=cwd)

    async def _run(
        self,
        command: str,
        timeout: float,
        *,
        stdin: str | None = None,
        cwd: str | Path | None = None,
    ) -> tuple[str, str, int, bool]:
        await self._start_unlocked()
        self._buffers = {name: TruncatingStringIO(limit=30_000) for name in ("stdout", "stderr")}
        self._last_command = command
        request = {
            "code": base64.b64encode(command.encode()).decode(),
            "cwd": str(cwd) if cwd else None,
            "input": base64.b64encode(stdin.encode()).decode() if stdin is not None else None,
        }
        try:
            self._process.stdin.write((json.dumps(request) + "\n").encode())
            await self._process.stdin.drain()
            response = await asyncio.wait_for(self._control_reader.readline(), timeout)
            if not response:
                raise EOFError("PowerShell exited before completing the command")
            status = json.loads(response)
            self._cwd = Path(status["cwd"])
            await asyncio.sleep(0.05)
            return (
                self._buffers["stdout"].getvalue(),
                self._buffers["stderr"].getvalue(),
                int(status["code"]),
                False,
            )
        except TimeoutError:
            result = (
                self._buffers["stdout"].getvalue(),
                self._buffers["stderr"].getvalue(),
                124,
                True,
            )
            await self._stop_unlocked(False)
            return result
        except asyncio.CancelledError:
            await self._stop_unlocked(False)
            raise
        except BaseException:
            await self._stop_unlocked(False)
            raise

    async def run_stream(
        self,
        command: str,
        timeout: float = 30.0,
        *,
        stdin: str | None = None,
        cwd: str | Path | None = None,
    ) -> AsyncIterator[tuple[str, str]]:
        self._ensure_lock_on_current_loop()
        async with self._lock:
            self._events = asyncio.Queue(maxsize=128)
            task = asyncio.create_task(self._run(command, timeout, stdin=stdin, cwd=cwd))
            try:
                while not task.done() or not self._events.empty():
                    receive = asyncio.create_task(self._events.get())
                    done, _ = await asyncio.wait(
                        (task, receive), return_when=asyncio.FIRST_COMPLETED
                    )
                    if receive in done:
                        yield receive.result()
                    else:
                        receive.cancel()
                        await asyncio.gather(receive, return_exceptions=True)
                _, _, code, timed = await task
                yield "__done__", f"{code},{int(timed)}"
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                self._events = None

    async def reset(self) -> None:
        self._ensure_lock_on_current_loop()
        async with self._lock:
            await self._stop_unlocked(False)
            await self._start_unlocked()

    async def _stop_unlocked(self, keep: bool) -> None:
        """Release process resources without replacing the caller's lock."""
        same_loop = self._started_on_loop in (None, asyncio.get_running_loop())
        process, job = self._process, self._job
        self._process = self._job = None
        self._started = False
        self._started_on_loop = None
        if job:
            if keep:
                job.keep_alive(True)
            job.close()
        if process:
            if keep and same_loop and process.returncode is None:
                try:
                    process.stdin.write(b'{"op":"close"}\n')
                    await process.stdin.drain()
                except (BrokenPipeError, ConnectionError):
                    pass
            elif process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            if process.stdin:
                try:
                    process.stdin.close()
                except RuntimeError:
                    if same_loop:
                        raise
            if same_loop:
                await asyncio.wait_for(process.wait(), 5)
        if self._control_writer:
            try:
                self._control_writer.close()
            except RuntimeError:
                if same_loop:
                    raise
            self._control_writer = None
        self._control_reader = None
        if not keep or not same_loop:
            for reader in self._readers:
                if not reader.get_loop().is_closed():
                    reader.cancel()
            if same_loop:
                await asyncio.gather(*self._readers, return_exceptions=True)
            self._readers.clear()

    async def close(self) -> None:
        self._ensure_lock_on_current_loop()
        async with self._lock:
            await self._stop_unlocked(self._keep_background_on_close)
