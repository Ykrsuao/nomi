# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VM teardown must let concurrent sessions keep making progress.

The asynchronous runtime destroy may take time to finish. A simulated slow VM
checks event-loop progress and both public close entry points without booting a VM.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from aster import Agent
from aster.runtime.sandbox.config import SandboxConfig
from aster.runtime.sandbox.executor import SandboxedExecutor
from aster.unifiedllm import FakeLLMClient


class _Agent(Agent, llm=FakeLLMClient()):
    pass


class _SlowVM:
    """Stands in for a runtime whose asynchronous cleanup takes time."""

    DESTROY_S = 0.5

    def __init__(self) -> None:
        self.destroyed = False
        self.calls = 0

    async def destroy(self, *, force: bool, timeout: float) -> None:
        assert force and timeout > 0
        self.calls += 1
        await asyncio.sleep(self.DESTROY_S)
        self.destroyed = True


@pytest.mark.asyncio
async def test_async_teardown_does_not_block_the_event_loop():
    ex = SandboxedExecutor(_Agent(), SandboxConfig(require=False), cell_timeout=10.0)
    vm = _SlowVM()
    ex._vm = vm

    ticks = 0

    async def heartbeat() -> None:
        nonlocal ticks
        for _ in range(40):
            await asyncio.sleep(0.01)
            ticks += 1

    hb = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)  # let the heartbeat task start
    t0 = time.monotonic()
    await ex.aclose()
    teardown_s = time.monotonic() - t0
    # Capture progress before waiting for the heartbeat itself to finish.
    ticks_during_teardown = ticks
    await hb

    assert vm.calls == 1 and vm.destroyed
    assert teardown_s >= _SlowVM.DESTROY_S * 0.8
    assert ticks_during_teardown >= 20, (
        f"event loop was blocked during teardown (only {ticks_during_teardown} ticks)"
    )


def test_close_sync_waits_for_vm_cleanup_without_a_running_loop():
    """The synchronous public entry point completes the asynchronous cleanup."""
    ex = SandboxedExecutor(_Agent(), SandboxConfig(require=False), cell_timeout=10.0)
    vm = _SlowVM()
    ex._vm = vm
    ex.close_sync()
    assert vm.calls == 1 and vm.destroyed
    assert ex._vm is None and ex._closed


async def test_failed_destroy_can_be_retried():
    class FlakyVM(_SlowVM):
        async def destroy(self, *, force, timeout):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("temporary runtime failure")
            self.destroyed = True

    ex = SandboxedExecutor(_Agent(), SandboxConfig(), cell_timeout=10)
    vm = FlakyVM()
    ex._vm = vm
    with pytest.raises(TimeoutError):
        await ex.aclose()
    assert ex._vm is vm
    await ex.aclose()
    assert vm.calls == 2 and vm.destroyed
    assert ex._vm is None


async def test_cancelled_close_shares_inflight_cleanup_with_next_close():
    ex = SandboxedExecutor(_Agent(), SandboxConfig(), cell_timeout=10)
    vm = _SlowVM()
    ex._vm = vm
    closing = asyncio.create_task(ex.aclose())
    while vm.calls == 0:
        await asyncio.sleep(0)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    await ex.aclose()
    assert vm.destroyed and vm.calls == 1
