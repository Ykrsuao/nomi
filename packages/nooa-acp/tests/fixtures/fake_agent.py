# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic ACP subprocess used by protocol smoke tests."""

import asyncio
import sys

from nooa_acp.server import serve

from nooa.unifiedllm import FakeLLMClient


def pause_first_prefill() -> None:
    """Hold setup before execution so cancellation is reproducible, even on a fast host."""
    from nooa.strategies.codeact import CodeActStrategy

    original = CodeActStrategy._execute_code
    paused = False

    async def execute(self, *args, **kwargs):
        nonlocal paused
        if not paused and kwargs.get("output_metadata", {}).get("prefill"):
            # The prefill event has already been enqueued, but no execution
            # environment is ready yet. Set the flag before yielding to the pump.
            paused = True
            await asyncio.Event().wait()
        return await original(self, *args, **kwargs)

    CodeActStrategy._execute_code = execute


def llm_factory() -> FakeLLMClient:
    if "--shell" in sys.argv:
        # Blocks inside a real shell command, so cancellation exercises
        # ActivityShellTools.run rather than a bare asyncio wait.
        return FakeLLMClient.with_tool_call(
            "execute_python",
            {"code": "await self.shell.run('sleep 30', timeout=30)"},
        )
    if "--blocking" in sys.argv:
        return FakeLLMClient.with_tool_call(
            "execute_python",
            {
                "code": ("self.message('Sandbox ready')\n" if "--sandbox" in sys.argv else "")
                + "await asyncio.Event().wait()"
            },
        )
    return FakeLLMClient.with_tool_call(
        "execute_python",
        {
            "code": (
                "import sys\nself.message('Interpreter platform: ' + sys.platform)\n"
                if "--sandbox" in sys.argv
                else ""
            )
            + (
                "self.message('NOOA ACP smoke test passed.')\n"
                "return_result(Done(explanation='smoke test complete'))"
            )
        },
    )


if "--pause-prefill" in sys.argv:
    pause_first_prefill()

asyncio.run(serve(llm_factory, sandbox="--sandbox" in sys.argv))
