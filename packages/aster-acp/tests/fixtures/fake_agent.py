# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic ACP subprocess used by protocol smoke tests."""

import asyncio
import json
import sys

from aster_acp.server import serve

from aster.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


def pause_first_prefill() -> None:
    """Hold setup before execution so cancellation is reproducible, even on a fast host."""
    from aster.strategies.codeact import CodeActStrategy

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
    if "--noisy" in sys.argv:
        print("PROVIDER_INITIALIZATION_DIAGNOSTIC", flush=True)
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
    code = (
        "import sys\nself.message('Interpreter platform: ' + sys.platform)\n"
        if "--sandbox" in sys.argv
        else ""
    )
    code += "print(sum([20, 22]))"
    return FakeLLMClient(
        [
            LLMResponse(
                raw_response=None,
                content="",
                tool_calls=[
                    ToolCall(id=f"smoke-{index}", name=name, arguments=json.dumps(arguments))
                ],
                finish_reason="tool_calls",
            )
            for index, (name, arguments) in enumerate(
                [
                    ("execute_python", {"code": code}),
                    (
                        "return_result",
                        {
                            "result": {
                                "message": "ASTER ACP smoke test passed.",
                                "explanation": "smoke test complete",
                            }
                        },
                    ),
                ]
                * (8 if "--rewind" in sys.argv else 1)
            )
        ],
        strict_exhaustion=True,
    )


if "--pause-prefill" in sys.argv:
    pause_first_prefill()

if "--noisy" in sys.argv:
    original_acall = FakeLLMClient.acall

    async def noisy_acall(self, *args, **kwargs):
        print("PROVIDER_REQUEST_DIAGNOSTIC", flush=True)
        return await original_acall(self, *args, **kwargs)

    FakeLLMClient.acall = noisy_acall

if "--cli" in sys.argv:
    from aster_acp.cli import command

    import aster.secrets
    import aster.unifiedllm

    def load_configuration():
        print("CONFIGURATION_DIAGNOSTIC", flush=True)

    aster.secrets.load_secrets_into_env = load_configuration
    aster.unifiedllm.get_llm_client = lambda *args, **kwargs: llm_factory()
    command(["--model", "test-model", "--no-sandbox"])
else:
    asyncio.run(serve(llm_factory, sandbox="--sandbox" in sys.argv))
