# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real microVM tests through the executor and Agent interfaces."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.runtime.sandbox.config import SandboxConfig
from nooa.runtime.sandbox.executor import SandboxedExecutor
from nooa.strategies.codeact import CodeActStrategy
from nooa.strategies.codeact_v2 import CodeActV2
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

pytestmark = pytest.mark.sandbox


class Service:
    value = 2

    def multiply(self, value):
        self.value = value * 3
        return self.value

    async def bump_async(self):
        self.value += 1
        return self.value

    def read_path(self, path):
        return Path(path).read_text(encoding="utf-8")


async def test_real_vm_keeps_namespace_and_brokers_state(tmp_path):
    service = Service()
    executor = SandboxedExecutor(service, SandboxConfig(workspace=str(tmp_path)), cell_timeout=15)
    try:
        first = await executor.run_cell("values = [1, 2, 3]\ndef bump(v):\n    return v + 1")
        assert first.error is None, first.error
        second = await executor.run_cell(
            "print(self.multiply(bump(sum(values))))", execution_count=2
        )
        assert second.error is None, second.error
        assert second.stdout.strip() == "21"
        assert service.value == 21
    finally:
        await executor.aclose()


@pytest.mark.parametrize("strategy_type", [CodeActStrategy, CodeActV2])
async def test_real_codeact_returns_guest_result(tmp_path, strategy_type):
    def response(code):
        return LLMResponse(
            raw_response=None,
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                ToolCall(
                    id=str(len(code)),
                    name="python_cell" if strategy_type is CodeActV2 else "execute_python",
                    arguments=json.dumps({"code": code}),
                )
            ],
        )

    final_cell = "return_result(value * 7)"
    if strategy_type is CodeActV2:
        final_cell = (
            "assert python_cell_state()['cell_locals'] == {'seed': 'int', 'value': 'int'}\n"
            + final_cell
        )
    llm = FakeLLMClient(scripted_responses=[response("value = seed"), response(final_cell)])

    class Calculator(Agent, llm=llm):
        @strategy(
            strategy_type(
                config=CodeActConfig(
                    execution_backend="sandbox", sandbox=SandboxConfig(workspace=str(tmp_path))
                )
            )
        )
        async def calculate(self, seed: int) -> int:
            """Compute the answer."""
            ...

    calculator = Calculator()
    try:
        assert await calculator.calculate(6) == 42
    finally:
        await calculator.aclose()


async def test_async_tools_are_lazy_and_host_paths_round_trip(tmp_path):
    path = tmp_path / "输入 with spaces.txt"
    path.write_text("你好", encoding="utf-8")
    service = Service()
    executor = SandboxedExecutor(
        service,
        SandboxConfig(workspace=str(tmp_path)),
        cell_timeout=15,
        framework_builtins={"input_path": path},
    )
    try:
        result = await executor.run_cell(
            "pending = self.bump_async()\nassert self.value == 2\nawait pending\n"
            "assert self.value == 3\nprint(input_path.read_text(encoding='utf-8'))\nprint(self.read_path(input_path))"
        )
        assert result.error is None, result.error
        assert result.stdout.splitlines() == ["你好", "你好"]
    finally:
        await executor.aclose()


async def test_guest_file_and_network_limits(tmp_path):
    from nooa.runtime.sandbox.config import FileRule

    workspace = tmp_path / "job"
    secret = tmp_path / "read-only.txt"
    secret.write_text("allowed", encoding="utf-8")
    executor = SandboxedExecutor(
        Service(),
        SandboxConfig(workspace=str(workspace), allow=(FileRule(path=str(secret)),)),
        cell_timeout=15,
        framework_builtins={"allowed_file": secret},
    )
    try:
        result = await executor.run_cell(
            "from pathlib import Path\nimport socket\n"
            "assert allowed_file.read_text() == 'allowed'\n"
            "(Path.cwd() / 'result.txt').write_text('saved')\n"
            "try:\n    allowed_file.write_text('changed')\nexcept OSError:\n    pass\nelse:\n    raise AssertionError('read-only file was writable')\n"
            "try:\n    socket.socket(socket.AF_INET, socket.SOCK_STREAM)\nexcept PermissionError:\n    pass\nelse:\n    raise AssertionError('internet socket was allowed')"
        )
        assert result.error is None, result.error
        assert secret.read_text() == "allowed"
        assert (workspace / "result.txt").read_text() == "saved"
    finally:
        await executor.aclose()


async def test_timeout_restarts_empty_and_close_wakes_unbounded_cell(tmp_path):
    from nooa.runtime.sandbox.errors import CellTimeoutError, WorkerDiedError

    executor = SandboxedExecutor(
        Service(), SandboxConfig(workspace=str(tmp_path), timeout_grace_s=0), cell_timeout=0.3
    )
    try:
        assert (await executor.run_cell("remembered = 9")).error is None
        killed = await executor.run_cell("while True:\n    pass")
        assert isinstance(killed.error, CellTimeoutError)
        fresh = await executor.run_cell("assert 'remembered' not in globals()\nprint('fresh')")
        assert fresh.error is None, fresh.error
        assert fresh.stdout.strip() == "fresh"
        await executor.aclose()
        executor = SandboxedExecutor(
            Service(), SandboxConfig(workspace=str(tmp_path)), cell_timeout=None
        )
        assert (await executor.run_cell("1")).error is None
        pending = asyncio.create_task(executor.run_cell("while True:\n    pass"))
        await asyncio.sleep(0.1)
        await asyncio.wait_for(executor.aclose(), 10)
        result = await asyncio.wait_for(pending, 2)
        assert isinstance(result.error, WorkerDiedError)
        with pytest.raises(WorkerDiedError):
            await executor.run_cell("1")
    finally:
        await executor.aclose()


async def test_output_capture_is_bounded(tmp_path):
    from nooa.config.truncation_config import CaptureConfig, TruncationConfig

    capture = TruncationConfig(capture=CaptureConfig(max_stdout=64, max_stderr=32))
    executor = SandboxedExecutor(
        Service(),
        SandboxConfig(workspace=str(tmp_path)),
        cell_timeout=15,
        truncation_config=capture,
    )
    try:
        result = await executor.run_cell("print('HEAD' + 'x'*10000 + 'TAIL')")
        assert result.error is None, result.error
        assert len(result.stdout) < 1000
        assert "HEAD" in result.stdout and "TAIL" in result.stdout
        assert "max_string" in result.truncation_reasons
    finally:
        await executor.aclose()


async def test_cpu_and_memory_limits(tmp_path):
    from nooa.runtime.sandbox.errors import CellTimeoutError

    cpu = SandboxedExecutor(
        Service(),
        # Let the kernel limit fire before the host watchdog can observe usage.
        SandboxConfig(workspace=str(tmp_path), max_cpu_seconds=1, rss_poll_s=60),
        cell_timeout=None,
    )
    try:
        result = await cpu.run_cell("while True:\n    pass")
        assert isinstance(result.error, CellTimeoutError), result.error
    finally:
        await cpu.aclose()
    memory = SandboxedExecutor(
        Service(), SandboxConfig(workspace=str(tmp_path), max_memory_mb=32), cell_timeout=10
    )
    try:
        result = await memory.run_cell("data = bytearray(128 * 1024 * 1024)")
        assert result.error is not None
        assert "Memory" in str(result.error)
    finally:
        await memory.aclose()


async def test_local_parent_module_and_imported_numpy_function(tmp_path):
    import sys
    import types

    from numpy import array

    parent = types.ModuleType("nooa_local_parent_fixture")
    exec(
        "class Parent:\n    value = 5\n    def increment(self, number):\n        return number + self.value",
        parent.__dict__,
    )
    sys.modules[parent.__name__] = parent

    class Child(parent.Parent):
        pass

    executor = SandboxedExecutor(
        Child(),
        SandboxConfig(workspace=str(tmp_path)),
        cell_timeout=15,
        framework_builtins={"array": array},
    )
    try:
        result = await executor.run_cell("result = array([self.increment(1), 7])\nreturn result")
        assert result.error is None, result.error
        assert result.returned_value.tolist() == [6, 7]
    finally:
        await executor.aclose()
        sys.modules.pop(parent.__name__)
