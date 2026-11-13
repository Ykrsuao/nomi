# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guest signal exits preserve resource failures without hiding unrelated kills."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import msgpack
import pytest

from aster.runtime.sandbox.config import SandboxConfig
from aster.runtime.sandbox.errors import CellTimeoutError, WorkerDiedError
from aster.runtime.sandbox.executor import SandboxedExecutor
from aster.runtime.sandbox.protocol import frame


@pytest.mark.parametrize(
    ("signum", "cpu_seconds", "cpu_limit", "error_type"),
    [
        pytest.param(9, 2.0, 1, CellTimeoutError, id="hard-cpu-limit"),
        pytest.param(24, 2.0, 1, CellTimeoutError, id="soft-cpu-limit"),
        pytest.param(9, 0.5, 1, WorkerDiedError, id="kill-before-budget"),
        pytest.param(9, 2.0, 0, WorkerDiedError, id="kill-without-cpu-limit"),
        pytest.param(15, 2.0, 1, WorkerDiedError, id="unrelated-signal"),
    ],
)
async def test_guest_signal_exit_cause(signum, cpu_seconds, cpu_limit, error_type):
    executor = SandboxedExecutor(
        object(), SandboxConfig(max_cpu_seconds=cpu_limit), cell_timeout=None
    )
    payload = msgpack.packb({"type": "worker_exit", "signal": signum, "cpu_seconds": cpu_seconds})
    assert isinstance(payload, bytes)

    async def events():
        yield SimpleNamespace(event_type="stdout", data=frame(payload))
        yield SimpleNamespace(event_type="exited", code=-1)

    executor._stream = events()
    executor._messages = asyncio.Queue()
    executor._cpu_baseline = 0.5
    try:
        await executor._read_events()
        error = executor._messages.get_nowait()
        assert type(error) is error_type
        assert executor._worker_failure is error
    finally:
        await executor.aclose()
