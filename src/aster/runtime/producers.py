# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Async producer helpers for QueueManager.spawn().

Each function returns a coroutine or async generator suitable for
``qm.spawn(producer, channel="name")``.

Usage::

    from aster.runtime.producers import monitor, after, cron

    qm.spawn(monitor("make test"), channel="ci", buffer=100)
    qm.spawn(after(300), channel="wakeup")
    qm.spawn(cron(60), channel="ticks")
"""

import asyncio
import os
import signal

from aster._platform import is_native_windows


async def after(delay: float) -> None:
    """One-shot timer: sleep *delay* seconds, then return None."""
    await asyncio.sleep(delay)
    return None


async def cron(interval_seconds: float):
    """Yield a tick counter at fixed intervals.

    Yields an incrementing integer every *interval_seconds*.
    """
    tick = 0
    while True:
        await asyncio.sleep(interval_seconds)
        tick += 1
        yield tick


async def tail(path: str, *, poll_interval: float = 0.5):
    """Tail a file, yielding new lines as they appear.

    Starts from the current end of file. Polls every
    *poll_interval* seconds. A line that is still being written is
    held until its newline arrives, so a poll landing mid-write does
    not split one line across two yields.
    """
    fh = open(path)
    try:
        fh.seek(0, 2)
        partial = ""
        while True:
            chunk = fh.readline()
            if chunk:
                partial += chunk
                if partial.endswith("\n"):
                    yield partial.rstrip("\n")
                    partial = ""
            else:
                await asyncio.sleep(poll_interval)
    finally:
        fh.close()


async def run_job(coro, job_id: str):
    """Wrap a coroutine result with a job_id tag.

    Returns ``{"job_id": job_id, "result": <awaited value>}``.
    """
    result = await coro
    return {"job_id": job_id, "result": result}


async def monitor(cmd: str):
    """Stream stdout lines from a shell command as they appear.

    Yields each line (stripped) as it's written to stdout.
    Output is decoded as UTF-8 with undecodable bytes replaced, so a
    command that emits non-UTF-8 output does not terminate the stream.
    stderr is merged into stdout.  Uses ``start_new_session=True``
    for process-group isolation so multiple concurrent monitors
    (and the agent itself) don't contend for ptys or interfere
    with each other. On cancellation or early close, the entire process
    group is killed, even if its leader has already exited, to prevent
    orphaned children.

    Native Windows uses PowerShell and a Job Object for process-tree ownership;
    Linux/macOS use the host shell and a process group.
    """
    job = None
    if is_native_windows():
        import base64
        import subprocess

        from aster.tools._windows_process import WindowsJob, powershell_executable

        bootstrap = (
            "[Console]::InputEncoding=[Text.UTF8Encoding]::new($false); "
            "[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); "
            "$OutputEncoding=[Console]::OutputEncoding; "
            "& ([ScriptBlock]::Create([Console]::In.ReadToEnd())) | Out-Default"
        )
        proc = await asyncio.create_subprocess_exec(
            powershell_executable(),
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-InputFormat",
            "Text",
            "-OutputFormat",
            "Text",
            "-EncodedCommand",
            base64.b64encode(bootstrap.encode("utf-16-le")).decode(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            job = WindowsJob(proc.pid)
            proc.stdin.write(cmd.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()
        except BaseException:
            if job:
                job.close()
            proc.kill()
            await proc.wait()
            raise
    else:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
    assert proc.stdout is not None
    completed = False
    try:
        async for line in proc.stdout:
            yield line.decode("utf-8", errors="replace").rstrip("\r\n")
        await proc.wait()
        completed = True
    finally:
        # A child can retain stdout after the group leader exits. Its exit
        # status alone must not suppress cleanup of an interrupted stream.
        if not completed:
            try:
                if job:
                    job.terminate()
                else:
                    os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                if proc.returncode is None:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
            await proc.wait()
        if job:
            job.close()
