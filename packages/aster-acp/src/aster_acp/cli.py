# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command-line entry points for the ASTER ACP agent."""

import asyncio
import os
from typing import TYPE_CHECKING

import click

from aster_acp._startup import mark, phase
from aster_acp._stdio import diagnostic_stdout

if TYPE_CHECKING:
    from aster.unifiedllm import UnifiedLLM


@click.command()
@click.option(
    "--model",
    envvar="ASTER_MODEL",
    required=True,
    help="LiteLLM model name or configured ASTER model alias. Or set ASTER_MODEL.",
)
@click.option(
    "--client-type",
    type=click.Choice(("completion", "responses")),
    default=None,
    help="Override the configured ASTER LLM client type.",
)
@click.option(
    "--sandbox/--no-sandbox",
    envvar="ASTER_SANDBOX",
    default=False,
    help="Execute CodeAct Python in a microsandbox microVM.",
)
def command(model: str, client_type: str | None, sandbox: bool) -> None:
    """Serve the ASTER coding agent over ACP on standard input/output."""
    # Imports and configuration may print provider diagnostics before serve()
    # has created the protocol connection. Keep those off the JSON-RPC pipe too.
    with diagnostic_stdout():
        with phase("framework_import"):
            from aster.secrets import load_secrets_into_env
        with phase("llm_import"):
            from aster.unifiedllm import get_llm_client
        with phase("acp_import"):
            from aster_acp.server import serve

        with phase("configuration"):
            load_secrets_into_env()
            # Session startup must not wait for the SDK's remote pricing catalogue.
            os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        with phase("provider_code_cache"):
            from aster_acp._provider_code import activate_provider_code_cache

            activate_provider_code_cache()
    nvidia_api_key = os.getenv("NVIDIA_API_KEY") if model.startswith("nvidia_nim/") else None

    def llm_factory() -> "UnifiedLLM":
        overrides = {"api_key": nvidia_api_key} if nvidia_api_key else {}
        return get_llm_client(model, client_type=client_type, **overrides)

    operation = serve(llm_factory, sandbox=True) if sandbox else serve(llm_factory)
    mark("server_start")
    asyncio.run(operation)


def main() -> None:
    command()
