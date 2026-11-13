# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch Aster without importing its runtime for help."""

import importlib.util
import sys
from pathlib import Path

import click


@click.command()
@click.option(
    "--working-dir",
    "-w",
    "workspace",
    type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path),
    default=".",
    show_default=True,
    help="Workspace whose instructions, settings, and sessions the agent uses.",
)
@click.option(
    "--model",
    envvar="ASTER_MODEL",
    default=None,
    help=(
        "Model alias or LiteLLM model name. Defaults to ASTER_MODEL, "
        "tui.default_model, or the only configured chat model."
    ),
)
@click.option(
    "--client-type",
    type=click.Choice(("completion", "responses")),
    default=None,
    help="Override the configured LLM client type.",
)
@click.option(
    "--sandbox/--no-sandbox",
    envvar="ASTER_SANDBOX",
    default=False,
    help="Execute CodeAct Python in a microsandbox microVM.",
)
@click.option("--resume", metavar="ID", help="Reopen a saved session by ID or unique prefix.")
@click.option("--continue", "continue_last", is_flag=True, help="Reopen the most recent session.")
def command(
    workspace: Path,
    model: str | None,
    client_type: str | None,
    sandbox: bool,
    resume: str | None,
    continue_last: bool,
) -> None:
    """Open Aster, the interactive terminal coding agent.

    The conversation prepares in the background while you type. Enter sends a
    message or queues it until ready; Alt+Enter inserts a newline. Esc recalls a
    message waiting for preparation, stops a running task, or closes a panel or
    suggestions while keeping the draft. Ctrl+F searches; Ctrl+E manages the queue.
    Press Ctrl+C twice within 2 seconds to exit; the first press shows confirmation.
    Ctrl+D exits when input is empty; Ctrl+Q exits at any time. Use /help for commands.
    """
    if resume is not None and continue_last:
        raise click.UsageError("Use either --resume or --continue, not both.")
    if model is not None and not model.strip():
        raise click.BadParameter("The model cannot be blank.", param_hint="--model")
    if resume is not None and not resume.strip():
        raise click.BadParameter("The session ID cannot be blank.", param_hint="--resume")
    if not all(stream is not None and stream.isatty() for stream in (sys.stdin, sys.stdout)):
        raise click.UsageError("Aster requires an interactive terminal for both input and output.")
    if any(importlib.util.find_spec(name) is None for name in ("acp", "aster_acp")):
        raise click.ClickException(
            "Install Aster's Python dependencies in this checkout with: uv sync --extra tui"
        )

    if model is None:
        from aster_cli._tui_model import resolve_tui_model

        try:
            model = resolve_tui_model(workspace)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc

    from aster_cli._tui_launcher import launch_tui

    try:
        returncode = launch_tui(
            workspace=workspace,
            model=model,
            client_type=client_type,
            sandbox=sandbox,
            resume=resume,
            continue_last=continue_last,
        )
    except (OSError, RuntimeError) as exc:
        raise click.ClickException(str(exc) or type(exc).__name__) from exc
    if returncode:
        raise click.exceptions.Exit(returncode)
