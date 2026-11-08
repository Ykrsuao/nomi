# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted user model definitions must survive a fresh interpreter."""

from __future__ import annotations

import importlib
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


@pytest.mark.parametrize("transport", ["interpreter", "microvm"])
async def test_local_helper_dependency_loads_in_fresh_interpreter(tmp_path, monkeypatch, transport):
    from nooa.runtime.sandbox.paths import PathMap

    sources = {
        "review_leaf": "def value():\n    return 42\n",
        "review_helper": "from review_leaf import value\ndef helper():\n    return value()\n",
        "review_agent": "from review_helper import helper\nclass Service: pass\n",
    }
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for name, source in sources.items():
        (source_dir / f"{name}.py").write_text(source, encoding="utf-8")
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.syspath_prepend(str(source_dir))
    module = importlib.import_module("review_agent")
    executor = SandboxedExecutor(
        module.Service(), SandboxConfig(workspace=str(tmp_path / "workspace")), cell_timeout=5
    )
    try:
        if transport == "microvm":
            result = await executor.run_cell("helper()")
            assert result.error is None, result.error
            assert result.returned_value == 42
            return
        executor._paths = PathMap([])
        executor._guest_allow = []
        payload = dumps_namespace(executor._bootstrap(), modules=executor._modules())
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import cloudpickle,sys; data=cloudpickle.loads(sys.stdin.buffer.read()); assert data['globals']['helper']() == 42",
            ],
            input=payload,
            capture_output=True,
            timeout=15,
        )
        assert result.returncode == 0, result.stderr.decode(errors="replace")
    finally:
        await executor.aclose()
        for name in sources:
            sys.modules.pop(name, None)


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
