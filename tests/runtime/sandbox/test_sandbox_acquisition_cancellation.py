# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cancellation must retain VMs whose SDK acquisition has not returned yet."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace

import pytest

from aster.runtime.sandbox import microsandbox_executor as backend
from aster.runtime.sandbox.config import SandboxConfig
from aster.runtime.sandbox.errors import SandboxUnavailable

_TIMEOUT = 10
_IMAGE = "python:synthetic-acquisition-test"


class _VM:
    def __init__(self, name: str, runtime: _Runtime):
        self.name = name
        self.runtime = runtime

    async def exec(self, *args, **kwargs):
        raise AssertionError("a cancelled acquisition must not start environment preparation")

    async def exec_stream(self, *args, **kwargs):
        raise AssertionError("a cancelled acquisition must not start the guest interpreter")

    async def destroy(self, *, force: bool, timeout: float | None = None):
        assert force is True
        assert timeout is None or timeout > 0
        self.runtime.destroy_calls += 1
        self.runtime.destroy_started.set()
        await self.runtime.finish_destroy.wait()
        self.runtime.live.pop(self.name)
        self.runtime.destroyed.append(self.name)


class _Runtime:
    """Native allocation keeps running when its Python SDK waiter is cancelled.

    The external resource exists before the return gate opens. Cancelling the
    SDK coroutine therefore cannot undo allocation; its eventual handle must
    still be collected and destroyed by the executor.
    """

    def __init__(self, path: str):
        self.path = path
        self.entered = asyncio.Event()
        self.finish_acquisition = asyncio.Event()
        self.acquisition_finished = asyncio.Event()
        self.destroy_started = asyncio.Event()
        self.finish_destroy = asyncio.Event()
        self.finish_destroy.set()
        self.native_tasks: list[asyncio.Task] = []
        self.live: dict[str, _VM] = {}
        self.destroyed: list[str] = []
        self.destroy_calls = 0
        self.snapshot = SimpleNamespace(image_ref=_IMAGE)
        self.sdk = ModuleType("microsandbox")
        self.sdk.Sandbox = SimpleNamespace(create=self.create, restore=self.restore)
        self.sdk.Snapshot = SimpleNamespace(open=self.open_snapshot, list=self.list_snapshots)
        self.sdk.SecurityProfile = SimpleNamespace(DEFAULT="default")
        self.sdk.Network = SimpleNamespace(allow_all=lambda: "all", none=lambda: "none")
        self.sdk.NetworkPolicy = self.sdk.Network
        self.sdk.Volume = SimpleNamespace(bind=lambda path, **kwargs: (path, kwargs))
        self.sdk.Stdin = SimpleNamespace(pipe=lambda: None)

    async def open_snapshot(self, selector):
        if self.path == "restore":
            return self.snapshot
        raise FileNotFoundError(selector)

    async def list_snapshots(self):
        return []

    async def create(self, name, **kwargs):
        assert self.path in {"create", "environment"}
        assert name.startswith("aster-prepare-" if self.path == "environment" else "aster-")
        return await self._acquire(name)

    async def restore(self, snapshot, *, name, **kwargs):
        assert self.path == "restore"
        assert snapshot is self.snapshot
        return await self._acquire(name)

    async def _acquire(self, name):
        async def allocate():
            vm = _VM(name, self)
            self.live[name] = vm
            self.entered.set()
            await self.finish_acquisition.wait()
            self.acquisition_finished.set()
            return vm

        native = asyncio.create_task(allocate())
        self.native_tasks.append(native)
        return await asyncio.shield(native)


@dataclass
class _Acquisition:
    executor: backend.SandboxedExecutor
    runtime: _Runtime
    tasks: list[asyncio.Task] = field(default_factory=list)

    def task(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.append(task)
        return task


@pytest.fixture(params=["create", "restore", "environment"])
async def acquisition(request, tmp_path, monkeypatch):
    runtime = _Runtime(request.param)
    monkeypatch.setitem(sys.modules, "microsandbox", runtime.sdk)
    monkeypatch.setenv("MSB_HOME", str(tmp_path / "isolated-runtime"))
    # This SDK double stops before serialization; ordinary CI omits sandbox extras.
    original_find_spec = backend.importlib.util.find_spec

    def find_spec(name, package=None):
        if name == "cloudpickle":
            return ModuleSpec(name, loader=None)
        return original_find_spec(name, package)

    monkeypatch.setattr(backend.importlib.util, "find_spec", find_spec)
    executor = backend.SandboxedExecutor(
        object(),
        SandboxConfig(
            workspace=str(tmp_path / "workspace"),
            image=_IMAGE,
            prepare_environment=request.param != "create",
        ),
        cell_timeout=10,
    )
    # Dependency discovery is separate from ownership of a pending VM.
    monkeypatch.setattr(executor, "_dependencies", lambda: ["cloudpickle==3.1.2"])
    harness = _Acquisition(executor, runtime)
    try:
        yield harness
    finally:
        runtime.finish_acquisition.set()
        runtime.finish_destroy.set()
        for task in harness.tasks:
            if not task.done():
                task.cancel()
        async with asyncio.timeout(_TIMEOUT):
            await asyncio.gather(*harness.tasks, return_exceptions=True)
            await executor.aclose()
            await asyncio.gather(*runtime.native_tasks, return_exceptions=True)


async def _checkpoint():
    """Let an already queued cancellation run without a timed sleep."""
    reached = asyncio.Event()
    asyncio.get_running_loop().call_soon(reached.set)
    await reached.wait()


@pytest.mark.parametrize("cancellations", [1, 2], ids=["cancel-once", "cancel-twice"])
async def test_cancelled_acquisition_destroys_the_late_vm(acquisition, cancellations):
    executor, runtime = acquisition.executor, acquisition.runtime
    async with asyncio.timeout(_TIMEOUT):
        starting = acquisition.task(executor.run_cell("return 1"))
        await runtime.entered.wait()
        assert len(runtime.live) == 1
        assert not runtime.acquisition_finished.is_set()

        for _ in range(cancellations):
            starting.cancel()
            await _checkpoint()
        runtime.finish_acquisition.set()
        await runtime.acquisition_finished.wait()

        with pytest.raises(asyncio.CancelledError):
            await starting
        await executor.aclose()

        assert not runtime.live, "cancellation lost the handle to a VM that completed later"
        assert runtime.destroy_calls == 1
        assert len(runtime.destroyed) == 1


async def test_cancelled_close_retains_cleanup_of_the_late_vm(acquisition):
    executor, runtime = acquisition.executor, acquisition.runtime
    runtime.finish_destroy.clear()
    async with asyncio.timeout(_TIMEOUT):
        starting = acquisition.task(executor.run_cell("return 1"))
        await runtime.entered.wait()
        starting.cancel()
        await _checkpoint()
        runtime.finish_acquisition.set()
        await runtime.acquisition_finished.wait()

        closing = acquisition.task(executor.aclose())
        destroying = acquisition.task(runtime.destroy_started.wait())
        # If the late handle was lost, close returns immediately. Fail on that
        # ownership error instead of waiting for a destruction that cannot start.
        await asyncio.wait({closing, destroying}, return_when=asyncio.FIRST_COMPLETED)
        assert runtime.destroy_started.is_set(), "close omitted the late acquisition's VM"
        assert not closing.done(), "close returned before VM destruction finished"

        closing.cancel()
        await _checkpoint()
        starting.cancel()
        replacement_close = acquisition.task(executor.aclose())
        runtime.finish_destroy.set()

        with pytest.raises(asyncio.CancelledError):
            await closing
        with pytest.raises(asyncio.CancelledError):
            await starting
        await replacement_close

        assert not runtime.live
        assert runtime.destroy_calls == 1
        assert len(runtime.destroyed) == 1


def _return_native_futures(monkeypatch, runtime: _Runtime) -> list[asyncio.Future]:
    """Expose a native-style Future from the synchronous SDK entry points."""
    futures = []

    def native_future(operation):
        task = asyncio.create_task(operation)
        runtime.native_tasks.append(task)
        future = asyncio.shield(task)
        futures.append(future)
        return future

    def create(name, **kwargs):
        return native_future(runtime.create(name, **kwargs))

    def restore(snapshot, *, name, **kwargs):
        return native_future(runtime.restore(snapshot, name=name, **kwargs))

    monkeypatch.setattr(runtime.sdk.Sandbox, "create", create)
    monkeypatch.setattr(runtime.sdk.Sandbox, "restore", restore)
    return futures


async def test_sdk_native_future_remains_owned_after_cancellation(acquisition, monkeypatch):
    executor, runtime = acquisition.executor, acquisition.runtime
    futures = _return_native_futures(monkeypatch, runtime)
    async with asyncio.timeout(_TIMEOUT):
        starting = acquisition.task(executor.run_cell("return 1"))
        await runtime.entered.wait()
        assert len(futures) == 1
        assert isinstance(futures[0], asyncio.Future)
        assert not isinstance(futures[0], asyncio.Task)

        starting.cancel()
        await _checkpoint()
        assert not futures[0].cancelled(), "caller cancellation reached the SDK's native Future"
        runtime.finish_acquisition.set()

        with pytest.raises(asyncio.CancelledError):
            await starting
        await executor.aclose()

        assert futures[0].done() and not futures[0].cancelled()
        assert not runtime.live
        assert runtime.destroy_calls == 1


async def _start_until_timeout(acquisition: _Acquisition, monkeypatch):
    executor, runtime = acquisition.executor, acquisition.runtime
    monkeypatch.setattr(backend, "_CLEANUP_TIMEOUT_S", 0.05)
    monkeypatch.setattr(
        executor, "_config", executor._config.model_copy(update={"startup_timeout_s": 0.05})
    )
    starting = acquisition.task(executor.run_cell("return 1"))
    await runtime.entered.wait()
    # Observe only: wait_for() would inject cancellation on its own deadline
    # and could accidentally make an unbounded product startup appear bounded.
    completed, _ = await asyncio.wait({starting}, timeout=1.0)
    assert starting in completed, "startup did not return while native acquisition stayed pending"
    with pytest.raises(SandboxUnavailable, match="microsandbox initialization failed") as failure:
        await starting
    assert isinstance(failure.value.__cause__, TimeoutError)
    assert not runtime.finish_acquisition.is_set()
    assert not runtime.acquisition_finished.is_set()
    assert len(runtime.live) == 1


async def test_startup_timeout_retains_late_vm_until_cleanup(acquisition, monkeypatch):
    executor, runtime = acquisition.executor, acquisition.runtime
    async with asyncio.timeout(_TIMEOUT):
        await _start_until_timeout(acquisition, monkeypatch)

        retry = acquisition.task(executor.run_cell("return 2"))
        completed, _ = await asyncio.wait({retry}, timeout=1.0)
        assert retry in completed, "a second run waited indefinitely for the pending cleanup"
        with pytest.raises((TimeoutError, SandboxUnavailable)):
            await retry
        assert len(runtime.native_tasks) == 1, "retry started another VM before cleanup finished"
        assert len(runtime.live) == 1

        # Closing also has a finite wait, and must leave the shared cleanup
        # alive when the native SDK still cannot deliver its handle.
        closing = acquisition.task(executor.aclose())
        completed, _ = await asyncio.wait({closing}, timeout=1.0)
        assert closing in completed, "close waited indefinitely for native acquisition"
        with pytest.raises(TimeoutError):
            await closing
        assert all(not task.cancelled() for task in runtime.native_tasks)
        assert not runtime.destroy_started.is_set()

        runtime.finish_acquisition.set()
        await runtime.acquisition_finished.wait()
        # Cleanup must resume from the late handle without another close call.
        await runtime.destroy_started.wait()
        assert not runtime.live
        assert runtime.destroy_calls == 1
        await executor.aclose()
        assert len(runtime.destroyed) == 1


async def test_failed_late_vm_destroy_can_be_retried_by_close(acquisition, monkeypatch):
    executor, runtime = acquisition.executor, acquisition.runtime
    original_destroy = _VM.destroy

    async def fail_once(vm, *, force, timeout=None):
        if runtime.destroy_calls == 0:
            assert force is True
            runtime.destroy_calls += 1
            runtime.destroy_started.set()
            raise OSError("synthetic late VM destruction failed")
        await original_destroy(vm, force=force, timeout=timeout)

    monkeypatch.setattr(_VM, "destroy", fail_once)
    async with asyncio.timeout(_TIMEOUT):
        await _start_until_timeout(acquisition, monkeypatch)
        closing = acquisition.task(executor.aclose())
        runtime.finish_acquisition.set()

        with pytest.raises(OSError, match="synthetic late VM destruction failed"):
            await closing
        assert runtime.destroy_calls == 1
        assert len(runtime.live) == 1
        assert runtime.destroyed == []

        await executor.aclose()

        assert not runtime.live
        assert runtime.destroy_calls == 2
        assert len(runtime.destroyed) == 1


async def test_cleanup_deadline_does_not_replace_caller_cancellation(acquisition, monkeypatch):
    executor, runtime = acquisition.executor, acquisition.runtime
    monkeypatch.setattr(backend, "_CLEANUP_TIMEOUT_S", 0.05)
    async with asyncio.timeout(_TIMEOUT):
        starting = acquisition.task(executor.run_cell("return 1"))
        await runtime.entered.wait()
        starting.cancel()

        # Keep the native return gate shut while the product's cleanup grace
        # period expires. Observing completion must not inject another cancel.
        completed, _ = await asyncio.wait({starting}, timeout=1.0)
        assert starting in completed, "caller cancellation exceeded the cleanup grace period"
        assert not runtime.finish_acquisition.is_set()
        assert not runtime.acquisition_finished.is_set()
        assert len(runtime.live) == 1
        assert all(not task.cancelled() for task in runtime.native_tasks)
        with pytest.raises(asyncio.CancelledError):
            await starting

        runtime.finish_acquisition.set()
        await runtime.acquisition_finished.wait()
        # Neither a new run nor close should be needed to claim the late VM.
        await runtime.destroy_started.wait()

        assert not runtime.live
        assert runtime.destroy_calls == 1
        assert len(runtime.destroyed) == 1
