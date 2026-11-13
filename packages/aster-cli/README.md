# Aster CLI (`aster-cli`)

The `aster` command launches the terminal coding workspace. Its subcommands
handle model setup, configuration, traces, evaluations, and editor integrations.
The terminal and ACP hosts share the Aster Python framework and saved sessions.

## Install

Run from the root of this Aster checkout:

```bash
uv sync --extra cli
uv sync --extra tui        # include the shared ACP runtime
```

The `aster-cli` package depends on `aster` (the core framework). To add the data
science libraries available to generated code, run
`uv sync --package aster-cli --extra datascience` from the workspace root.

> **Windows:** the CLI runs natively. Host shell commands use PowerShell; sandbox execution uses the optional microsandbox runtime. See the [main README](../../README.md#installation).

## Usage

```bash
uv run aster              # open the terminal workspace
uv run aster --help
uv run aster connect      # set up a model alias
uv run aster start-dev    # launch the trace viewer
uv run aster eval ...     # eval pipeline runner
uv run aster traces ...   # inspect/manage trace files
```

`uv sync --extra acp` adds the `aster acp` subcommand and the `aster-acp` entry
point for editor integrations. See the [ACP setup guide](../aster-acp/README.md)
and [framework documentation](../../docs/README.md).

## Terminal chat

`aster` launches the Aster Rust / Ratatui terminal workspace and
connects it to the Python ACP agent. `aster tui` explicitly opens the same interface. Build the frontend once from this repository checkout:

```bash
uv sync --extra tui
cargo build --locked --release --manifest-path packages/aster-tui/Cargo.toml
uv run --extra tui aster -w .
```

For `aster-cli[tui]` or `aster[tui]` installed from this checkout, place the compiled
`aster` executable (`aster.exe` on Windows) on PATH or set `ASTER_TUI_BIN` to its
absolute path.
Python wheels do not currently bundle the native binary. In a checkout the
launcher finds `packages/aster-tui/target/release` automatically. Rebuild after
changing Rust sources; startup does not download or compile anything.
Use `uv run --extra tui aster connect` to create a model alias if needed.
`--model` takes precedence over `ASTER_MODEL` and the saved
`tui.default_model` in layered `settings.yaml`, in that order. Without an explicit
choice, Aster uses the only configured chat model; multiple aliases require a
default or `--model`. To save a project default, add this to `.aster/settings.yaml`:

```yaml
tui:
  default_model: my-model
```

Once the Python launcher is on PATH, run `aster` without `uv run` or environment
activation. `aster` and `aster tui` start a new conversation with an empty composer
and queue. The selected workspace supplies the agent's
instructions, settings, secrets, and saved sessions. The interface appears while
the agent process starts and prepares the conversation in the background. You can
type immediately; Enter queues a message if preparation has not finished and sends
it when ready. Esc recalls a
waiting message while preparation continues. If another draft already exists,
both are preserved in the composer and paused queue. History is recorded when a
prompt is dispatched; unused prepared sessions are cleaned up on normal close.

The full-screen conversation view supports independent scrolling, Markdown
replies, persistent tool cards, and expandable file diffs. The conversation fills
the terminal width with a compact composer and grayscale canvas; Alt+B opens a sidebar on wide terminals for saved
conversations and recent tools. `/` shows suggestions above the composer;
Tab completes a command and Esc hides suggestions while keeping the draft. Ctrl+K searches commands,
Ctrl+R selects a saved conversation, Ctrl+T inspects complete tool output, and
Ctrl+G opens Git workspace changes, separating staged, unstaged and untracked files.
Tab switches to the conversation's saved edits; Ctrl+R refreshes Git. Ctrl+F searches the
conversation, including collapsed tool output, and jumps to the chosen result.
Replies support fenced code, lists, quotes, links and Unicode-aligned tables. Final
answers returned through the structured result tool appear as provider fragments
arrive; validation completes the same message without duplicating it. Replies
computed inside Python still arrive together. Cancelled previews are marked incomplete.

The composer grows for multiline input and preserves your draft while tools run
or panels open. Enter during a running prompt adds a message to the visible queue;
successful turns send queued messages in order. Cancellation or errors pause the
queue. Ctrl+E manages it and Ctrl+S continues. Drafts, attachment snapshots, unsent
messages and interrupted input are saved separately for each conversation under
`<workspace>/.aster/tui-drafts/`. Use `--continue`, `--resume ID` or `/sessions` to
reopen a saved conversation with its input. `/drafts` lists drafts containing
unsent input, newest first, including drafts saved before a first message was sent.
Select one and press Enter to restore it. Restored queues stay paused until
Ctrl+S; `/retry` recalls the current conversation's interrupted input for editing
before Enter sends it. See [recovery](../aster-tui/README.md#recovery) for details.
Bracketed paste inserts text without sending it. Live status
shows the current operation and elapsed time; context usage and cost appear
when the backend supplies them. Narrow terminals retain the composer and
conversation while hiding the sidebar. `NO_COLOR` uses terminal default colors.

| Key | Action |
| --- | --- |
| Enter | Send / queue the message, or select a picker item |
| Alt+Enter / Shift+Enter | Insert a newline (Shift+Enter requires terminal support) |
| `/` at the start of input | Inline commands: Up/Down select, Tab complete, Enter run |
| Ctrl+K | Search commands in a separate palette |
| Ctrl+R | Search and resume a conversation |
| Ctrl+F | Search this conversation and jump to a result |
| Ctrl+E / Ctrl+S | Manage queued messages / continue paused queue |
| Ctrl+T / Ctrl+G | Inspect tool output / file changes |
| Ctrl+O / click a tool | Expand or collapse tool cards |
| Alt+B | Toggle sidebar |
| PgUp / PgDn / mouse wheel | Scroll the conversation or inspector |
| Ctrl+End | Return to newest messages |
| Ctrl+P / Ctrl+N | Previous / next submitted input |
| Esc | Close suggestions / a panel; recall a message waiting for preparation; cancel a running task |
| Esc twice within 600 ms, while idle | Clear the draft; with empty input, select a conversation checkpoint |
| Ctrl+Z / Ctrl+U | Undo input edits, including double-Esc draft clearing |
| Ctrl+X / F1 | Keyboard help |
| Ctrl+C twice within 2 seconds | Save and exit; the first press only shows confirmation |
| Ctrl+D with empty input / Ctrl+Q | Save and exit |

Esc cancels a running task from the main conversation; an open panel or slash
suggestions close first, preserving the draft. The first Ctrl+C does not cancel
the task or clear input. Press it again within two seconds to exit; other keys,
paste, or mouse actions dismiss confirmation. If cancellation is still pending
after 20 seconds, the TUI reports that it is waiting and keeps the queue paused.
It stays open until the operation finishes or you explicitly choose to exit.

With an empty composer, double Esc or `/rewind` lets you select a previous user
message. The TUI opens a separate conversation ending before that message and
restores its text for editing; it sends nothing until you press Enter. The
original conversation is kept in `/sessions`, and workspace files remain as they
are. Rewind restores the saved conversation context and agent state. Only messages
with checkpoints recorded by this version can be selected. Resolve any queued
messages before rewinding. With a nonempty draft, double Esc clears it as a single
undoable edit. Esc used for cancellation or closing a panel does not count toward
double Esc; intervening input also resets the shortcut.

`/tree` opens the saved message tree, with shared history and alternative branches.
Type to search complete message text, use Up/Down to select, and press Enter to
navigate. Select a user message to recall its original text with the context from
before it, or an assistant reply to continue after it. Existing branches stay available when you
reopen the conversation. Navigation preserves your draft, sends no prompt, and
leaves workspace files unchanged. Finish active work and resolve the queue first;
older messages without saved restore points remain visible but unavailable.
See [session tree](../aster-tui/README.md#session-tree) for the input and recovery behavior.

Use `/help` for commands, `/cancel` to stop a turn, `/new` for a new conversation,
`/sessions` to select saved sessions, and `/resume <id-or-prefix>` to reopen one.
`/drafts` opens the saved draft menu; `/tree` opens the message branches.
`/tools` opens saved and current tool output; `/changes` shows Git and conversation diffs; `/logs`
shows backend diagnostics; `/status` shows the current session; `/exit`
closes the chat. A saved session
can also be selected when starting:

```bash
uv run --extra tui aster -w . --model my-model --continue
uv run --extra tui aster -w . --model my-model --resume SESSION_ID
```

`--continue` selects the most recent session in that workspace, or starts a
new conversation when there is no history. `/new` saves and closes the current
session, then prepares another in the background. Sessions use
the same `<workspace>/.aster/sessions` storage as Zed's ACP integration. Close
the session in its current host before opening it in another: each saved
session has one writer. Reopening restores conversation and agent context;
it does not resurrect a running command or a terminated process.
The backend replays saved Python calls, command output and file diffs alongside
messages. Terminal/file activity is saved starting with this version; older
sessions cannot recover events that were never recorded. `/changes` offers both
a read-only Git snapshot and the saved edits from the conversation.
The `.aster` data locations and the `aster`, `aster_cli`, and `aster_acp` Python
imports are unchanged. See [the Aster terminal README](../aster-tui/README.md) for
architecture and tests.

Native execution is the default. To use microsandbox, enable the checkout's
`tui` and `sandbox` extras and add `--sandbox` (or set `ASTER_SANDBOX=true`, with
`ASTER_SANDBOX` as a fallback):

```bash
uv run --extra tui --extra sandbox aster -w . --model my-model --sandbox
```

`--no-sandbox` explicitly overrides the environment. `--client-type responses`
or `--client-type completion` overrides the selected model's client type.
Both execution modes use the same ACP agent and session lifecycle as Zed;
see [sandbox execution](../aster-acp/README.md#sandbox-execution) for its scope.

## Interactive coding sessions

`aster_cli.sessions` owns durable coding-agent session identity, metadata, and
conversation replay shared by CLI hosts such as the native TUI and ACP. The
process running an agent owns the writable session handle; other hosts attach
through their transport or use read-only discovery. Generic event and SQLite
storage primitives remain in the core `aster` package.
