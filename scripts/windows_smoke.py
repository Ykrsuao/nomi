# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Installed-wheel smoke: native host features, no platform flag or hypervisor."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run_cli(*args):
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"ASTER_SKIP_PLATFORM_CHECK", "PYTHONUTF8", "PYTHONIOENCODING"}
    }
    result = subprocess.run(
        [sys.executable, "-m", "aster_cli", *args], env=env, capture_output=True
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    return result


async def check_host(directory):
    from aster import Agent, strategy
    from aster.runtime.producers import monitor
    from aster.storage.sqlite import (
        SessionAlreadyActiveError,
        SQLiteStorageManager,
        delete_sqlite_database,
    )
    from aster.strategies.predict import PredictStrategy
    from aster.tools.shell_tools import ShellTools
    from aster.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

    predict = FakeLLMClient(
        scripted_responses=[
            LLMResponse(
                raw_response=None, content='{"value": 7}', tool_calls=[], finish_reason="stop"
            )
        ]
    )

    class Extractor(Agent, llm=predict):
        @strategy(PredictStrategy())
        async def extract(self) -> int:
            """Return seven."""
            ...

    assert await Extractor().extract() == 7
    llm = FakeLLMClient(
        scripted_responses=[
            LLMResponse(
                raw_response=None,
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="code",
                        name="execute_python",
                        arguments=json.dumps({"code": "value = 6\nreturn_result(value * 7)"}),
                    )
                ],
            )
        ]
    )

    class Calculator(Agent, llm=llm):
        async def calculate(self) -> int:
            """Compute the answer."""
            ...

    assert await Calculator().calculate() == 42
    path = directory / "session.db"
    first = SQLiteStorageManager(path)
    try:
        try:
            SQLiteStorageManager(path)
        except SessionAlreadyActiveError:
            pass
        else:
            raise AssertionError("file storage was not exclusively locked")
    finally:
        first.close()
    reopened = SQLiteStorageManager(path, must_exist=True)
    reopened.close()
    assert delete_sqlite_database(path)
    shell = ShellTools(cwd=str(directory))
    try:
        command = (
            "$value=21; Write-Output 'ready'"
            if sys.platform == "win32"
            else "value=21; printf 'ready\\n'"
        )
        assert (await shell.run(command)).stdout.strip() == "ready"
        command = "Write-Output ($value * 2)" if sys.platform == "win32" else "echo $((value * 2))"
        assert (await shell.run(command)).stdout.strip() == "42"
        if sys.platform == "win32":
            executable = sys.executable.replace("'", "''")
            for returncode in (0, 7):
                result = await shell.run(
                    f"& '{executable}' -c 'import sys; print(43, file=sys.stderr); "
                    f"sys.exit({returncode})'; Write-Output 'continued'"
                )
                assert result.stdout.strip() == "continued", result
                assert result.stderr.strip() == "43", result
                assert result.returncode == returncode, result
    finally:
        await shell.close()
    command = "Write-Output 'line'" if sys.platform == "win32" else "printf 'line\\n'"
    assert [line async for line in monitor(command)] == ["line"]
    assert "microsandbox" not in sys.modules


def main():
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
    import aster

    print("aster", aster.__version__, sys.platform)
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        asyncio.run(check_host(directory))
        run_cli("--help")
        run_cli(
            "connect",
            "fake-model",
            "--provider",
            "custom",
            "--endpoint",
            "http://127.0.0.1:9/v1",
            "--api-style",
            "chat",
            "--api-key-env",
            "ASTER_SMOKE_FAKE_KEY",
            "--as",
            "fake",
            "--no-probe",
            "--no-catalogue",
            "--yes",
            "--output",
            str(directory / "llm_config.yaml"),
        )
        import yaml

        assert (
            "fake"
            in yaml.safe_load((directory / "llm_config.yaml").read_text(encoding="utf-8"))["models"]
        )
    print("SMOKE OK")


if __name__ == "__main__":
    main()
