# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep ACP wire output and task-local execution output independent."""

import asyncio
import io
import sys
from contextlib import nullcontext

import pytest
from aster_acp._stdio import diagnostic_stdout, protocol_streams

from aster.agent import Agent
from aster.runtime.actor import ActorRuntime, _stderr_buffer_var, _stdout_buffer_var
from aster.runtime.stream_wrappers import ContextVarStream
from aster.unifiedllm import FakeLLMClient


@pytest.mark.parametrize("fail", [False, True])
def test_diagnostics_restore_the_callers_stdout_after_normal_or_failed_service(monkeypatch, fail):
    output = io.StringIO()
    diagnostics = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "stderr", diagnostics)

    print("before service")
    expected_error = pytest.raises(RuntimeError, match="service failed") if fail else nullcontext()
    with expected_error, diagnostic_stdout():
        print("provider diagnostic", flush=True)
        if fail:
            raise RuntimeError("service failed")

    assert sys.stdout is output
    assert sys.stderr is diagnostics
    print("after service")
    assert output.getvalue() == "before service\nafter service\n"
    assert diagnostics.getvalue() == "provider diagnostic\n"


async def test_diagnostics_preserve_concurrent_cell_capture_with_existing_stream_wrappers(
    monkeypatch,
):
    """A previous agent cell may already have installed both capture wrappers."""

    class CaptureAgent(Agent, llm=FakeLLMClient()):
        pass

    runtimes = [ActorRuntime(CaptureAgent()), ActorRuntime(CaptureAgent())]
    output = io.StringIO()
    diagnostics = io.StringIO()
    original_stdout = ContextVarStream(output, _stdout_buffer_var, "stdout")
    original_stderr = ContextVarStream(diagnostics, _stderr_buffer_var, "stderr")
    monkeypatch.setattr(sys, "stdout", original_stdout)
    monkeypatch.setattr(sys, "stderr", original_stderr)
    # ActorRuntime also installs a persistent stdin wrapper; keep it test-local.
    monkeypatch.setattr(sys, "stdin", sys.stdin)

    with diagnostic_stdout():
        print("provider initialization")
        results = await asyncio.gather(
            *(
                runtime.execute_code(
                    "import asyncio\nimport sys\n"
                    f"print('stdout-{index}')\n"
                    "await asyncio.sleep(0)\n"
                    f"print('stderr-{index}', file=sys.stderr)",
                    validate=False,
                )
                for index, runtime in enumerate(runtimes)
            )
        )
        print("provider request")

    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr
    assert output.getvalue() == ""
    assert diagnostics.getvalue() == "provider initialization\nprovider request\n"
    for index, result in enumerate(results):
        assert result.error is None
        assert result.stdout == f"stdout-{index}\n"
        assert result.stderr == f"stderr-{index}\n"


async def test_windows_protocol_writer_keeps_original_binary_output_after_redirection(monkeypatch):
    """Use the real SDK transport that looks up sys.stdout on every write."""
    import acp.stdio
    from acp.core import DEFAULT_STDIO_BUFFER_LIMIT_BYTES

    binary_output = io.BytesIO()
    output = io.TextIOWrapper(binary_output, encoding="utf-8", write_through=True)
    diagnostics = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "stderr", diagnostics)
    monkeypatch.setattr(sys, "platform", "win32")
    reader = asyncio.StreamReader()
    sdk_writer = asyncio.StreamWriter(
        acp.stdio._StdoutTransport(),
        acp.stdio._WritePipeProtocol(),
        None,
        asyncio.get_running_loop(),
    )
    limits = []

    async def streams(*, limit):
        limits.append(limit)
        return reader, sdk_writer

    monkeypatch.setattr(acp.stdio, "stdio_streams", streams)
    protocol_reader, writer = await protocol_streams()
    try:
        assert protocol_reader is reader
        # AgentSideConnection requires this concrete type at runtime.
        assert isinstance(writer, asyncio.StreamWriter)
        with diagnostic_stdout():
            print("provider output", flush=True)
            writer.write(b'{"jsonrpc":"2.0","id":1,"result":{}}\n')
            await writer.drain()

        assert sys.stdout is output
        assert binary_output.getvalue() == b'{"jsonrpc":"2.0","id":1,"result":{}}\n'
        assert diagnostics.getvalue() == "provider output\n"
        assert limits == [DEFAULT_STDIO_BUFFER_LIMIT_BYTES]
    finally:
        writer.close()
    assert writer.is_closing()
    assert not output.closed
