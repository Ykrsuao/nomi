# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Authenticate parent admission failures without trusting worker exception names."""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import replace
from types import SimpleNamespace

import pytest

from aster import Agent
from aster.errors.formatting import format_error_for_llm
from aster.events import ExecutionResult
from aster.runtime.sandbox import wire
from aster.runtime.sandbox.config import SandboxConfig
from aster.runtime.sandbox.errors import CellSerializationError, SandboxExecutionError
from aster.runtime.sandbox.executor import SandboxedExecutor
from aster.runtime.sandbox.paths import PathMap
from aster.runtime.sandbox.protocol import FrameReader, frame
from aster.runtime.sandbox.serialization import (
    ResultDTO,
    _parent_error_token,
    dto_from_wire,
    dto_to_result,
    dto_to_wire,
    effective_error_limit,
    result_to_dto,
)
from aster.runtime.sandbox.worker import ParentToolError, _raise_broker_error
from aster.unifiedllm import AdmissionQueueFullError, FakeLLMClient


class _AdmissionAgent(Agent):
    """Expose a real policy exception and an unrelated tool failure."""

    saved_token: str = ""
    custom_message: str = ""

    def reject(self) -> None:
        """Simulate a parent-side policy rejection without a provider call."""
        raise AdmissionQueueFullError("pending queue is full")

    def ordinary(self) -> None:
        """Keep unrelated tool failures recoverable."""
        raise ValueError("ordinary tool failure")

    def empty(self) -> None:
        """An empty policy message must still cross the boundary faithfully."""
        raise AdmissionQueueFullError()

    def long(self) -> None:
        """Exercise authentication after parent-side message truncation."""
        raise AdmissionQueueFullError("pending " * 10_000)

    def lookalike(self) -> None:
        """A matching class name is not a real admission exception."""
        raise type("AdmissionQueueFullError", (ValueError,), {})("lookalike")

    def custom(self) -> None:
        """Preserve a custom controller's exact rejection message."""
        raise AdmissionQueueFullError(self.custom_message)

    async def slow(self) -> None:
        """Allow a broker timeout to retire the current cell."""
        await asyncio.sleep(10)


async def _parent_failure(key: bytes, *, method: str = "reject", max_error: int = 16_384) -> dict:
    """Exercise the actual parent RPC exception handler without starting a VM."""
    executor = object.__new__(SandboxedExecutor)
    executor._agent = _AdmissionAgent(llm=FakeLLMClient())
    executor._max_error = effective_error_limit(max_error)
    try:
        return await executor._dispatch_tool_call(
            {"kind": "call", "path": [method], "args": (), "kwargs": {}},
            parent_error_key=key,
        )
    finally:
        await executor._agent.aclose()


def _worker_failure(response: dict, *, max_error: int = 16_384):
    """Run real worker reconstruction and msgpack result transport."""
    with pytest.raises(Exception) as caught:
        _raise_broker_error(response)
    dto = result_to_dto(
        ExecutionResult(error=caught.value, stdout="before rejection\n"),
        error_formatter=lambda *args, **kwargs: "worker traceback and source",
        max_error=max_error,
    )
    codec = wire.Codec()
    return dto_from_wire(codec.loads(codec.dumps(dto_to_wire(dto))))


async def test_authenticated_parent_rejection_keeps_type_and_diagnostic():
    """The parent verifies provenance while retaining the worker's cell output."""
    key = secrets.token_bytes(32)
    response = await _parent_failure(key)
    assert len(response["parent_error_token"]) == 64
    dto = _worker_failure(response)
    result = dto_to_result(dto, parent_error_key=key)
    assert isinstance(result.error, SandboxExecutionError)
    assert isinstance(result.error.original_error, AdmissionQueueFullError)
    assert str(result.error.original_error) == "pending queue is full"
    assert format_error_for_llm(result.error) == "worker traceback and source"
    assert result.stdout == "before rejection\n"


@pytest.mark.parametrize(
    "windows, host, message",
    [
        (True, "E:/repo", "E:/repo/queue"),
        (True, "E:/repo", "/workspace/queue"),
        (False, "/srv/repo", "/srv/repo//queue"),
    ],
)
async def test_path_mapping_keeps_authenticated_error_text(monkeypatch, windows, host, message):
    """Real send/read framing preserves signatures while ordinary paths still map."""
    import pickle

    executor = _executor()
    executor._agent.custom_message = message
    executor._paths = PathMap([(host, "/workspace")], windows=windows)
    executor._codec.path_transform = executor._paths.to_host
    executor._messages = asyncio.Queue()
    sent = []

    async def write(data):
        sent.append(data)

    executor._sink = SimpleNamespace(write=write)
    monkeypatch.setattr(executor, "_modules", lambda: [])
    key = secrets.token_bytes(32)
    try:
        response = await executor._dispatch_tool_call(
            {"kind": "call", "path": ["custom"], "args": (), "kwargs": {}},
            parent_error_key=key,
        )
        response.update(type="tool_result", tool_call_id=1, call_hint=host + "/help")
        await executor._send(response)
        # Only the trusted host-to-guest bytes use pickle, exactly as the guest does.
        received = pickle.loads(next(FrameReader().feed(sent[0])))
        assert received["error"] == message
        assert received["call_hint"] == "/workspace/help"
        dto = _worker_failure(received)
        dto.stdout = "/workspace/output"

        async def events():
            yield SimpleNamespace(
                event_type="stdout",
                data=frame(
                    wire.Codec().dumps({"type": "response", "id": 1, "result": dto_to_wire(dto)})
                ),
            )

        executor._stream = events()
        await executor._read_events()
        reply = await executor._messages.get()
        result = dto_to_result(dto_from_wire(reply["result"]), parent_error_key=key)
        assert isinstance(result.error.original_error, AdmissionQueueFullError)
        assert str(result.error.original_error) == message
        assert result.stdout == executor._paths.to_host("/workspace/output")
    finally:
        await executor._agent.aclose()


@pytest.mark.parametrize("tamper", ["token", "unicode-token", "message", "type", "stale", "no-key"])
async def test_worker_cannot_promote_unauthenticated_rejections(tamper):
    """Forged, altered and prior-cell errors remain ordinary sandbox errors."""
    key = secrets.token_bytes(32)
    dto = _worker_failure(await _parent_failure(key))
    assert dto.error is not None
    if tamper == "token":
        dto.error = replace(dto.error, parent_error_token="0" * 64)
    elif tamper == "unicode-token":
        dto.error = replace(dto.error, parent_error_token="\ud800" * 64)
    elif tamper == "message":
        dto.error = replace(dto.error, message="fabricated policy failure")
    elif tamper == "type":
        dto.error = replace(dto.error, type_name="ValueError")
    elif tamper == "stale":
        key = secrets.token_bytes(32)
    else:
        key = None
    result = dto_to_result(dto, parent_error_key=key)
    assert isinstance(result.error, SandboxExecutionError)
    assert not isinstance(result.error.original_error, AdmissionQueueFullError)


@pytest.mark.parametrize("method", ["ordinary", "lookalike"])
async def test_ordinary_parent_error_receives_no_admission_token(method):
    """Ordinary errors retain their original recoverable transport behavior."""
    key = secrets.token_bytes(32)
    response = await _parent_failure(key, method=method)
    assert "parent_error_token" not in response
    result = dto_to_result(_worker_failure(response), parent_error_key=key)
    expected = ValueError if method == "ordinary" else Exception
    assert isinstance(result.error.original_error, expected)
    assert not isinstance(result.error.original_error, AdmissionQueueFullError)


@pytest.mark.parametrize("method", ["reject", "empty", "long"])
@pytest.mark.parametrize("budget", [8, 16_384])
async def test_parent_error_authentication_respects_message_budget(method, budget):
    """Empty and truncated error text is signed exactly as the worker receives it."""
    key = secrets.token_bytes(32)
    response = await _parent_failure(key, method=method, max_error=budget)
    assert len(response["error"]) <= effective_error_limit(budget)
    dto = _worker_failure(response, max_error=budget)
    result = dto_to_result(dto, parent_error_key=key)
    assert isinstance(result.error.original_error, AdmissionQueueFullError)
    assert str(result.error.original_error) == response["error"]


async def test_authentication_does_not_depend_on_worker_diagnostic():
    """A valid parent error still propagates when no worker traceback is available."""
    key = secrets.token_bytes(32)
    dto = _worker_failure(await _parent_failure(key))
    dto.error = replace(dto.error, diagnostic="")
    assert isinstance(dto_to_result(dto, parent_error_key=key).error, AdmissionQueueFullError)


def test_error_name_alone_does_not_create_a_policy_exception():
    """A worker-side lookalike does not acquire parent-side authority."""
    dto = _worker_failure({"error_type": "AdmissionQueueFullError", "error": "lookalike"})
    result = dto_to_result(dto, parent_error_key=secrets.token_bytes(32))
    assert isinstance(result.error.original_error, Exception)
    assert not isinstance(result.error.original_error, AdmissionQueueFullError)
    with pytest.raises(ParentToolError):
        _raise_broker_error({"error_type": "AdmissionQueueFullError", "error": "lookalike"})


@pytest.mark.parametrize("token", [None, 1, b"token", "x" * 65])
def test_malformed_parent_error_tokens_are_refused(token):
    """Admission provenance metadata obeys the existing bounded DTO contract."""
    with pytest.raises(CellSerializationError, match="malformed result"):
        dto_from_wire({"error": {"type_name": "E", "message": "m", "parent_error_token": token}})


def test_hostile_error_token_attribute_does_not_break_result_transport():
    """A user exception cannot abort serialization through its metadata property."""

    class HostileError(ValueError):
        @property
        def _aster_parent_error_token(self):
            raise KeyboardInterrupt("hostile metadata")

    dto = result_to_dto(
        ExecutionResult(error=HostileError("original failure")),
        error_formatter=lambda *args, **kwargs: "worker diagnostic",
    )
    assert dto.error.message == "original failure"
    assert dto.error.parent_error_token == ""
    assert dto.error.diagnostic == "worker diagnostic"


@pytest.mark.parametrize("retire", ["completion", "cancellation", "timeout", "worker-failure"])
async def test_cell_keys_rotate_after_completion_or_recovery(monkeypatch, retire):
    """Real async dispatch binds each response to its cell and keeps keys host-side."""
    executor = _executor()
    started = asyncio.Event()
    keys = []
    replies = []
    random_bytes = secrets.token_bytes

    async def start_worker():
        executor._vm = object()
        executor._messages = asyncio.Queue()
        executor._worker_failure = None

    async def dispose(**kwargs):
        executor._vm = None

    def create_key(size):
        assert executor._vm is not None
        key = random_bytes(size)
        keys.append(key)
        return key

    async def send(message):
        # Neither the startup namespace nor any worker request carries the key.
        assert all(repr(key) not in repr(message) for key in keys)
        if message.get("op") == "run":
            executor._messages.put_nowait(
                {
                    "type": "tool_call",
                    "tool_call_id": 1,
                    "kind": "call",
                    "path": ["reject"],
                    "payload": wire.Codec().dumps(((), {})),
                }
            )
        else:
            assert message["type"] == "tool_result"
            assert message["parent_error_token"] == _parent_error_token(message["error"], keys[-1])
            replies.append(message)
            started.set()
            if len(replies) == 1:
                if retire in {"cancellation", "timeout"}:
                    return
                if retire == "worker-failure":
                    from aster.runtime.sandbox.errors import WorkerDiedError

                    executor._messages.put_nowait(WorkerDiedError("worker stopped"))
                    return
                result = ResultDTO()
            else:
                result = _worker_failure(replies[0])
            executor._messages.put_nowait(
                {"type": "response", "id": executor._req_id, "result": dto_to_wire(result)}
            )

    monkeypatch.setattr(executor, "_start", start_worker)
    monkeypatch.setattr(executor, "_send", send)
    monkeypatch.setattr(executor, "_dispose", dispose)
    monkeypatch.setattr(secrets, "token_bytes", create_key)
    try:
        if retire == "timeout":
            executor._cell_timeout = 0.01
            executor._config = executor._config.model_copy(update={"timeout_grace_s": 0})
        first = asyncio.create_task(executor.run_cell("first"))
        await asyncio.wait_for(started.wait(), 5)
        if retire == "cancellation":
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
        else:
            result = await first
            assert (result.error is None) == (retire == "completion")
        executor._cell_timeout = None
        result = await executor.run_cell("second")
        assert len(keys) == 2 and keys[0] != keys[1]
        assert isinstance(result.error, SandboxExecutionError)
        assert not isinstance(result.error.original_error, AdmissionQueueFullError)
    finally:
        await executor._agent.aclose()


def _executor(workspace=None, **kwargs):
    """Use the microsandbox backend with a per-test writable workspace."""
    return SandboxedExecutor(
        _AdmissionAgent(llm=FakeLLMClient()),
        SandboxConfig(workspace=str(workspace) if workspace else None, **kwargs),
        cell_timeout=15,
    )


@pytest.mark.sandbox
@pytest.mark.parametrize(
    "after",
    ["pass", "raise ValueError('recoverable')", "raise saved"],
)
async def test_worker_may_catch_or_reraise_a_parent_rejection(after, tmp_path):
    """Catch/rethrow semantics match in-process execution and preserve stdout."""
    executor = _executor(tmp_path)
    try:
        result = await executor.run_cell(
            "from aster.unifiedllm import AdmissionQueueFullError\n"
            "try:\n    self.reject()\n"
            "except AdmissionQueueFullError as exc:\n    saved = exc\n"
            "print('caught rejection')\n" + after
        )
        assert "caught rejection" in result.stdout
        if after == "pass":
            assert result.error is None
        else:
            expected = AdmissionQueueFullError if after == "raise saved" else ValueError
            assert isinstance(result.error, SandboxExecutionError)
            assert isinstance(result.error.original_error, expected)
        assert (await executor.run_cell("1 + 1")).returned_value == 2
    finally:
        await executor.aclose()
        await executor._agent.aclose()


@pytest.mark.sandbox
@pytest.mark.parametrize("retire", ["next-cell", "timeout", "crash", "broker-timeout", "close"])
async def test_worker_rejection_authority_is_retired_between_cells(retire, tmp_path):
    """A saved token is invalid in later cells, including after worker replacement."""
    executor = _executor(tmp_path, broker_timeout_s=0.05)
    try:
        first = await executor.run_cell(
            "try:\n    self.reject()\nexcept Exception as exc:\n"
            "    saved = exc\n    self.saved_token = exc._aster_parent_error_token"
        )
        assert first.error is None
        assert len(executor._agent.saved_token) == 64
        if retire == "close":
            token = executor._agent.saved_token
            await executor.aclose()
            await executor._agent.aclose()
            executor = _executor(tmp_path)
            executor._agent.saved_token = token
        elif retire != "next-cell":
            code = {
                "timeout": "while True: pass",
                "crash": "import os\nos._exit(3)",
                "broker-timeout": "await self.slow()",
            }[retire]
            if retire == "timeout":
                executor._cell_timeout = 0.05
            assert (await executor.run_cell(code)).error is not None
            executor._cell_timeout = 15
        result = await executor.run_cell(
            "from aster.unifiedllm import AdmissionQueueFullError\n"
            "error = AdmissionQueueFullError('pending queue is full')\n"
            "error._aster_parent_error_token = self.saved_token\nraise error"
        )
        assert result.error is not None
        assert not isinstance(result.error.original_error, AdmissionQueueFullError)
        assert (await executor.run_cell("2 + 2")).returned_value == 4
    finally:
        await executor.aclose()
        await executor._agent.aclose()


@pytest.mark.sandbox
async def test_many_caught_rejections_need_no_parent_exception_registry(tmp_path):
    """Repeated caught errors keep working; an early rethrow is still authenticated."""
    executor = _executor(tmp_path)
    executor._cell_timeout = 30
    try:
        result = await executor.run_cell(
            "for i in range(1000):\n"
            "    try:\n        self.reject()\n"
            "    except Exception as exc:\n"
            "        if i == 0:\n            saved = exc\n"
            "raise saved"
        )
        assert isinstance(result.error.original_error, AdmissionQueueFullError)
        assert (await executor.run_cell("3 + 3")).returned_value == 6
    finally:
        await executor.aclose()
        await executor._agent.aclose()
