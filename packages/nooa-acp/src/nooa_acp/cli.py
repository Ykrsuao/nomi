# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command-line entry points for the NOOA ACP agent."""

import asyncio
import os
from typing import TYPE_CHECKING

import click

from nooa_acp._startup import mark, phase

if TYPE_CHECKING:
    from nooa.unifiedllm import UnifiedLLM


@click.command()
@click.option(
    "--model",
    envvar="NOOA_MODEL",
    required=True,
    help="LiteLLM model name or configured NOOA model alias. Or set NOOA_MODEL.",
)
@click.option(
    "--client-type",
    type=click.Choice(("completion", "responses")),
    default=None,
    help="Override the configured NOOA LLM client type.",
)
@click.option(
    "--sandbox/--no-sandbox",
    envvar="NOOA_SANDBOX",
    default=False,
    help="Execute CodeAct Python in a microsandbox microVM.",
)
def command(model: str, client_type: str | None, sandbox: bool) -> None:
    """Serve the NOOA coding agent over ACP on standard input/output."""
    with phase("framework_import"):
        from nooa.secrets import load_secrets_into_env
    with phase("llm_import"):
        from nooa.unifiedllm import get_llm_client
    with phase("acp_import"):
        from nooa_acp.server import serve

    with phase("configuration"):
        load_secrets_into_env()
    nvidia_api_key = os.getenv("NVIDIA_API_KEY") if model.startswith("nvidia_nim/") else None

    def llm_factory() -> "UnifiedLLM":
        overrides = {"api_key": nvidia_api_key} if nvidia_api_key else {}
        return get_llm_client(model, client_type=client_type, **overrides)

    operation = serve(llm_factory, sandbox=True) if sandbox else serve(llm_factory)
    mark("server_start")
    asyncio.run(operation)


def main() -> None:
    command()
