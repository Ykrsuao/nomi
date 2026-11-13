# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep Python diagnostics separate from the ACP protocol's stdout pipe."""

from __future__ import annotations

import asyncio
import sys
from contextlib import AbstractContextManager, redirect_stdout
from typing import Any, BinaryIO


class _DiagnosticOutput:
    """Forward to stderr without pretending to be its task-local capture wrapper."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def diagnostic_stdout() -> AbstractContextManager[_DiagnosticOutput]:
    # The runtime installs separate ContextVarStream wrappers for stdout and
    # stderr. Reusing stderr's wrapper as stdout would mix captured cell output.
    return redirect_stdout(_DiagnosticOutput(sys.stderr))


class _BoundStdoutWriter(asyncio.StreamWriter):
    """Pin Windows protocol writes before sys.stdout becomes a diagnostic stream."""

    def __init__(self, writer: asyncio.StreamWriter, output: BinaryIO) -> None:
        self._writer = writer
        self._output = output

    def write(self, data: bytes) -> None:
        # The SDK's Windows transport also writes synchronously, but looks up
        # sys.stdout for every message. That lookup is unsafe after redirection.
        self._output.write(data)
        self._output.flush()

    async def drain(self) -> None:
        await self._writer.drain()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._writer, name)


async def protocol_streams() -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    from acp.core import DEFAULT_STDIO_BUFFER_LIMIT_BYTES
    from acp.stdio import stdio_streams

    output = sys.stdout.buffer
    reader, writer = await stdio_streams(limit=DEFAULT_STDIO_BUFFER_LIMIT_BYTES)
    if sys.platform == "win32":
        writer = _BoundStdoutWriter(writer, output)
    # POSIX's SDK writer already owns its pipe and preserves async backpressure.
    return reader, writer
