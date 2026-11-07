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
