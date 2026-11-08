# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted user model definitions must survive a fresh interpreter."""

from __future__ import annotations

import subprocess
import sys
import types

import pytest

from nooa.runtime.sandbox.bootstrap import dumps_namespace
from nooa.runtime.sandbox.config import SandboxConfig
from nooa.runtime.sandbox.executor import SandboxedExecutor

pytestmark = pytest.mark.sandbox


def test_user_model_is_reconstructed_with_its_validation(monkeypatch):
    module = types.ModuleType("nooa_bootstrap_validation_fixture")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(
        "from pydantic import BaseModel\n"
        "class Model(BaseModel):\n"
        "    value: int\n"
        "class Holder:\n"
        "    def accept(self, value: Model) -> Model:\n"
        "        return value\n",
        module.__dict__,
    )
    executor = SandboxedExecutor(module.Holder(), SandboxConfig(), cell_timeout=5)
    try:
        payload = dumps_namespace(module.Model, modules=executor._modules())
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import cloudpickle,sys\n"
                "from pydantic import ValidationError\n"
                "model=cloudpickle.loads(sys.stdin.buffer.read())\n"
                "assert model(value='42').value == 42\n"
                "try:\n"
                "    model(value='invalid')\n"
                "except ValidationError:\n"
                "    pass\n"
                "else:\n"
                "    raise AssertionError('Model validation was lost')\n",
            ],
            input=payload,
            capture_output=True,
            timeout=15,
        )
        assert result.returncode == 0, result.stderr.decode(errors="replace")
    finally:
        executor.close_sync()


async def test_codeact_v2_bootstrap_does_not_capture_host_task(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from nooa.runtime.sandbox.paths import PathMap
    from nooa.strategies.codeact_v2 import CodeActV2

    module = types.ModuleType("review_v2_agent")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec("class Service: pass", module.__dict__)
    agent = module.Service()
    call = SimpleNamespace(
        return_type=int,
        kwargs={},
        method_name="answer",
        bound_parameters=lambda: {},
        task=asyncio.current_task(),
    )
    builtins = CodeActV2()._build_builtins(SimpleNamespace(agent=agent), call)
    executor = SandboxedExecutor(
        agent,
        SandboxConfig(),
        cell_timeout=5,
        framework_builtins={**builtins, "_call": call},
    )
    executor._paths = PathMap([])
    executor._guest_allow = []
    try:
        payload = dumps_namespace(executor._bootstrap(), modules=executor._modules())
        assert payload
    finally:
        await executor.aclose()


async def test_guest_cell_state_reports_rebound_inputs_and_new_locals():
    from nooa.runtime.sandbox.cell_core import run_cell_source
    from nooa.runtime.sandbox.guest import cell_state_function

    namespace = {"question": "input", "module_constant": 99}
    namespace["python_cell_state"] = cell_state_function(namespace, ["question"])
    result = await run_cell_source(
        "import json as json_alias\nquestion = 7\nanswers = [1, 2]", namespace
    )
    assert result.error is None, result.error
    result = await run_cell_source("python_cell_state()", namespace, execution_count=2)
    assert result.error is None, result.error
    assert result.returned_value == {
        "cell_locals": {"question": "int", "answers": "list"},
        "cell_imports": {"json_alias": "json"},
    }
