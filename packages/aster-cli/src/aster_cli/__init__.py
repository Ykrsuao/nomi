# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aster CLI — extensible command-line toolkit.

Usage:
    aster eval --config config.yaml  # Run an eval-pipeline job
    aster start-dev                  # Start the viewer
    aster traces delete              # Delete traces
    aster completion install         # Set up shell completions

Adding new commands:
    In this package: drop a .py file in aster_cli/commands/ — see
    commands/_template.py
    From another package: register an entry point in the
    "aster_cli.commands" group — see aster_cli/commands/__init__.py

Shell completion:
    eval "$(_ASTER_COMPLETE=bash_source aster)"    # bash
    eval "$(_ASTER_COMPLETE=zsh_source aster)"     # zsh
"""

import sys

import click

from .commands import discover_commands
from .commands.tui import command as tui_command
from .completion import completion

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


# `completion` just emits a shell script and doesn't need the ~1.5s core
# import cost or secrets preload. TUI loads secrets in its agent process,
# after selecting the workspace, and can draw before the framework imports.
_SKIP_SECRETS_PRELOAD = {"completion", "tui"}


@click.group(
    name="aster",
    context_settings=CONTEXT_SETTINGS,
    invoke_without_command=True,
    no_args_is_help=False,
    subcommand_metavar="[COMMAND] [ARGS]...",
    params=list(tui_command.params),
)
@click.version_option(package_name="aster-cli", prog_name="Aster")
@click.pass_context
def cli(ctx: click.Context, **terminal_options: object) -> None:
    """Open Aster's terminal workspace, or run a subcommand.

    Run aster with no command to start coding in the current directory.
    Use -w to choose a workspace, --model to choose a model, and connect
    to configure a provider. Terminal options also work with aster tui.
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(tui_command, **terminal_options)
        return
    explicit_options = [
        param.opts[0]
        for param in tui_command.params
        if isinstance(param, click.Option)
        and param.name is not None
        and ctx.get_parameter_source(param.name) is click.core.ParameterSource.COMMANDLINE
    ]
    if explicit_options:
        raise click.UsageError(
            "Use terminal options without a subcommand, or after 'aster tui': "
            + ", ".join(explicit_options)
        )
    if ctx.invoked_subcommand in _SKIP_SECRETS_PRELOAD:
        return
    # Load secrets.yaml into the process env before any subcommand runs so
    # all commands (config show, eval, …) see the same API keys.
    # Non-clobbering (shell env wins) and best-effort: a broken or missing
    # secrets file must never block the CLI — but log the failure at debug
    # so it's recoverable rather than silently swallowed.
    try:
        from aster.secrets import load_secrets_into_env

        load_secrets_into_env()
    except Exception:
        import logging

        logging.getLogger(__name__).debug(
            "Failed to preload secrets.yaml into the environment", exc_info=True
        )


# -- Auto-discover and register all commands (built-ins + entry-point plugins) --
for _name, _cmd in discover_commands():
    cli.add_command(_cmd, name=_name)

# -- Built-in infrastructure commands (not in commands/ because they're meta) -
# Registered *after* discovery on purpose: `completion` is not part of
# discover_commands(), so this ordering is what stops a third-party plugin
# named "completion" from taking the name.
cli.add_command(completion)


def _is_utf8(stream) -> bool:
    encoding = getattr(stream, "encoding", None) or ""
    return encoding.lower().replace("-", "").replace("_", "") == "utf8"


def _use_utf8_output() -> None:
    """Use UTF-8 for redirected output and piped input on native Windows.

    Redirected stdout/stderr and piped stdin use the ANSI code page there (for
    example cp1252), which cannot encode the CLI's check marks and arrows and
    would crash the command midway. Consoles already use UTF-8, so this changes
    nothing there.
    """
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and not _is_utf8(stream) and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    stdin = sys.stdin
    try:
        piped = stdin is not None and not stdin.isatty()
    except (OSError, ValueError):
        piped = False
    if piped and not _is_utf8(stdin) and hasattr(stdin, "reconfigure"):
        stdin.reconfigure(encoding="utf-8", errors="replace")


def main(*, command: click.Command | None = None) -> None:
    """Run a console command with the shared Windows UTF-8 setup."""
    _use_utf8_output()
    (command or cli)(prog_name="aster")


def aster_main() -> None:
    """Run Aster through the shared console entry point."""
    main()
