# aster CLI — Adding New Commands

## Quick Start

```bash
cd src/aster_cli/commands/
cp _template.py mycommand.py    # copy the template
# edit mycommand.py              # add your logic
aster mycommand              # it just works
```

That's it. No registration, no config files, no editing other files.

## How It Works

The CLI auto-discovers every `.py` file in `src/aster_cli/commands/` at startup.
Files starting with `_` are ignored (private helpers, templates).

### The Contract

Your command module must export **one thing**: a module-level variable named `command`
that is a Click command or group.

```python
# commands/mycommand.py
import click

@click.command()
@click.argument("name")
def command(name: str):
    """One-line description shown in `aster --help`."""
    click.echo(f"Hello, {name}!")
```

The filename becomes the subcommand name: `mycommand.py` → `aster mycommand`.

### Override the Name

Add `NAME` at module level if the filename doesn't match what you want:

```python
NAME = "my-command"  # aster my-command (instead of aster mycommand)
```

## Patterns

### Simple Command (leaf, no subcommands)

```python
import click

@click.command()
@click.argument("target", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True)
def command(target: str, verbose: bool):
    """Do something to TARGET."""
    click.echo(f"Processing {target}")
```

### Group with Subcommands

For `aster things list` / `aster things create`:

```python
import click

@click.group()
def command():
    """Manage things."""

@command.command()
def list():
    """List all things."""
    click.echo("thing-1\nthing-2")

@command.command()
@click.argument("name")
def create(name: str):
    """Create a new thing."""
    click.echo(f"Created: {name}")
```

### Passthrough to Another CLI

For wrapping an existing tool without duplicating its options:

```python
import click

@click.command(
    add_help_option=False,
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def command(args: tuple[str, ...]):
    """Run some-other-tool (all args forwarded)."""
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "some_other_tool", *args])
```

## Commands From Another Package

If your command lives in a *different* installed package (not in this repo),
register it in the `aster_cli.commands` entry-point group instead of dropping a
file here. The entry-point name becomes the subcommand name:

```toml
# pyproject.toml of your package
[project.entry-points."aster_cli.commands"]
tui  = "my_package.cli.tui:command"
term = "my_package.cli.term:command"
```

`my_package.cli.tui:command` must be a `click.Command` or `click.Group` — the
same contract as an in-repo command module.

- **Built-ins win name collisions.** A plugin can't shadow `eval`, `config`,
  or anything else shipped here; it's logged and skipped.
- **A broken plugin is skipped, not fatal.** An entry point that fails to
  import, or that resolves to a non-`click.Command`, logs a warning and is
  left out. `aster` keeps working.
- Plugins register in entry-point-name order, so `aster --help` is stable
  regardless of install order.
- The **Performance Rule** below applies with extra force: every registered
  entry point is loaded on *every* `aster` invocation, including `aster --help`.
  Keep heavy imports inside the handler.
- **Importing `aster_cli` must not import `aster`** (`test_connect_entrypoint.py`
  checks this). `main()` switches redirected output to UTF-8 on Windows.
  New console entry points should go through `main()`.

This mirrors the existing `aster.skills` and `aster.bundled_configs` groups.

## Shared Utilities

Common helpers live in `src/aster_cli/_common.py`:

```python
from aster_cli._common import find_project_root, format_size

root = find_project_root()           # Path to project root (where pyproject.toml is)
format_size(1_500_000)               # "1.4 MB"
```

For API keys / env vars, use `aster.secrets.load_secrets_into_env()`
(reads the layered `secrets.yaml`); for any layered config file use
`aster.layered_config.load_layered_yaml(filename, env_var)`.

## Performance Rule

**Keep imports lazy.** The CLI starts in ~0.3s because it doesn't import the
heavy `aster` framework at module level. Only import heavy dependencies
inside your command handler function:

```python
@click.command()
def command():
    """Do something that needs the framework."""
    # These imports happen only when the command actually runs,
    # not when `aster --help` loads the CLI.
    from aster import Agent
    import pandas as pd
```

## File Layout

```
src/aster_cli/
├── __init__.py              # Root CLI group + auto-discovery wiring
├── __main__.py              # python -m aster_cli
├── _common.py               # Shared utilities
├── completion.py            # Shell completion (bash/zsh/fish)
├── AGENTS.md                # ← You are here
└── commands/
    ├── __init__.py          # Auto-discovery engine
    ├── _template.py         # Copy-paste starter for new commands
    ├── eval.py              # aster eval ...
    ├── traces.py            # aster traces cleanup/list/stats
    └── start_dev.py         # aster start-dev
```

## Shell Completion

New commands get tab-completion automatically. To enable it:

```bash
aster completion install   # auto-detects your shell

# Or manually:
eval "$(_ASTER_COMPLETE=zsh_source aster)"   # zsh
eval "$(_ASTER_COMPLETE=bash_source aster)"  # bash
```
