# Aster terminal workspace

Aster brings coding conversations, tool output, Git changes, and saved sessions
into one terminal workspace. Its Rust / Ratatui frontend connects to Aster's
Python ACP agent. The terminal owns the conversation view, composer, searchable
pickers, tool inspector, and change viewer. Python owns model configuration,
tools, execution, and saved sessions.

Launch a new conversation with `aster` or `aster tui`. Rust sources live in
`packages/aster-tui`; project configuration and sessions live in `.aster/`.

## Build and run

From this repository checkout, with Python 3.12–3.13, Rust 1.88+ and uv installed:

```powershell
uv sync --extra tui
cargo build --locked --release --manifest-path packages/aster-tui/Cargo.toml
uv run --extra tui aster
```

Use an existing model alias, or create one with `uv run --extra tui aster connect`.
When the Python launcher is on PATH, `aster` starts directly in the current
workspace. Model selection uses `--model`, then `ASTER_MODEL`, then
`tui.default_model` from layered `settings.yaml`. If no default is saved,
Aster can use a single configured chat model; several aliases require a choice.
For example, save this in your project's `.aster/settings.yaml`:

```yaml
tui:
  default_model: my-model
```

User defaults can be saved in `~/.config/aster/settings.yaml`. Existing model
aliases and project configuration continue to work.
`-w PATH`, `--client-type completion|responses`, `--sandbox`,
`--resume ID` and `--continue` are forwarded to the Rust client. Python is selected
from the environment running `aster`, so the ACP subprocess has the same installed
packages. New conversations prepare their model and session in the background
as the interface opens. You can type immediately. If preparation is still running
when you press Enter, the message waits in the visible queue and is sent when ready.
`--resume ID` reopens a chosen session. `--continue` reopens the most recent
session, or starts a new conversation when there is no history.

The Python wheel does not currently bundle the native executable. For a locally
installed package, build this crate and put `aster` (`aster.exe` on Windows) on
PATH, or set `ASTER_TUI_BIN` to its absolute path.
Launcher discovery checks this override, a bundled `aster_cli/bin` binary, the
checkout's release build, then PATH. It never downloads or compiles code during
startup. Rebuild after changing Rust sources.

Native execution is the default. To use the optional microsandbox runtime:

```powershell
uv run --extra tui --extra sandbox aster --model my-model --sandbox
```

`ASTER_SANDBOX=true` also enables sandbox execution. `--no-sandbox` explicitly
overrides the environment. See
[sandbox execution](../aster-acp/README.md#sandbox-execution) for runtime setup.

## Context and cost

The top-right header is present from startup. `context — · cost —` means that no
model response has supplied statistics yet. After a response, it shows the latest
reported input-token usage as a percentage of the model's known context window.
If the window is unknown, it shows the reported token count instead. A missing
usage report displays `context n/a`; it does not imply an empty context.

The ACP backend can read the exact model's published input limit from a configured
OpenAI-compatible gateway's `/models` response. This optional lookup runs when
opening a session or switching models, without sending a prompt. Explicit model
settings take priority, with provider-aware LiteLLM metadata as the fallback.

The amount is the estimated USD cost reported for model calls since this
conversation was opened in the current process. Loading an older conversation
does not reconstruct its earlier charges. A missing or invalid price displays
`cost n/a`, including when only some calls have prices, so an incomplete sum is
not presented as the full cost. An explicitly reported zero can display `$0.0000`.
Changing models clears the previous context measurement while keeping the current
cost total; the new model supplies its own context measurement on its next response.

Narrow terminals use shorter labels and keep the indicators visible. The
`examples/preview.rs` executable uses synthetic conversation and usage fixtures;
its `welcome` mode has no sample usage. Those images demonstrate rendering, not
measurements from a live model session.

## Controls

| Key | Action |
| --- | --- |
| Enter | Send a message; queue it while preparing or while a prompt is running |
| Alt+S | Send the current text as a correction to the running task |
| Alt+Enter / Shift+Enter | New line (Shift+Enter depends on terminal support) |
| Click / drag inside input | Position the cursor / select text; Shift+click extends the selection |
| Ctrl+C / Ctrl+X with selected input and no popup open | Copy / cut the selection to the system clipboard |
| Ctrl+V / Ctrl+Shift+V / Shift+Insert / Right click | Paste clipboard text or attach a copied image without sending |
| `@` at a word boundary / Alt+A | Find and attach a workspace file |
| Click an attachment name / +N | Preview that attachment / inspect hidden attachments |
| Enter / click the image in the attachment panel | Open the original image in your system image viewer |
| Click × beside an attachment / Alt+Backspace | Remove that attachment / the last attachment from the draft |
| Backspace with empty input | Remove the last attachment from the draft |
| Click Undo / Ctrl+Z immediately after removal | Restore the last removed attachment |
| Up / Down at the first / last input row | Previous / next submitted message; return to your draft |
| `/` at the start of input | Show inline command suggestions |
| Up / Down, Tab, Enter | Select, complete or run an inline command |
| Ctrl+K | Open the command palette without replacing the draft |
| Ctrl+R | Search and restore a conversation |
| Ctrl+F | Search messages, tool output and diffs in this conversation |
| Ctrl+E | Manage queued messages |
| Ctrl+S | Continue a paused message queue |
| Ctrl+T | Browse tool calls and saved output in this conversation |
| Ctrl+G | Inspect Git workspace changes; Tab switches to conversation edits |
| Ctrl+U in Git Changes | Review rollback of the selected file or one hunk |
| Alt+T | Inspect plan steps, waiting reasons and background jobs |
| Ctrl+O | Expand/collapse all message and tool details |
| Click a message / tool / more-lines hint | Expand or collapse that content |
| Alt+B | Toggle the conversation/activity sidebar |
| PgUp / PgDn, mouse wheel | Scroll conversation or inspector |
| Ctrl+End | Return to newest messages |
| Ctrl+P / Ctrl+N | Previous/next submitted input |
| Ctrl+C twice within 2 seconds | Save and exit from a popup or with no composer selection; the first press shows confirmation |
| Esc | Close suggestions / an overlay; recall a message waiting for preparation; cancel a running task |
| Esc twice within 600 ms, while idle | Clear a nonempty draft; with an empty composer, choose a message to rewind to |
| Ctrl+Z / Ctrl+U in the composer | Undo an input edit, including clearing a draft with double Esc |
| Ctrl+D with empty input, Ctrl+Q | Save and exit |
| Ctrl+X without selected input / F1 (no popup open) | Help |

Pickers support typing to filter, arrow selection, Enter to choose and Esc to
return to the draft. Slash commands stay in the composer, with a filtered list
above it. Tab completes the selected name and inserts a space for arguments;
Enter executes the selected command. Esc hides the list without changing the
draft. The Ctrl+K palette inserts commands with arguments only when the composer
is empty; an existing draft, cursor and undo history are preserved.
Type `/resume ` followed by a session ID or unique prefix. You can draft a
message during execution; Enter queues ordinary messages and shows what is next
above the composer. Messages run one at a time after a successful turn. Cancel or
errors pause the queue; Ctrl+S continues. Ctrl+E opens the queue: Enter takes an
item back for editing when the composer is empty, and Delete removes it. Taking
an item back also pauses the remaining queue. Drafts, queued messages and their
attachment snapshots are autosaved separately from submitted conversation history;
recovered inputs always start paused. See [Recovery](#recovery).
Finish or remove queued messages before switching to another conversation.
The same rule applies to rewinding a conversation.
When the next message starts automatically, the queue panel keeps the same
remaining message selected, including in filtered lists and with duplicate text.

Background preparation leaves the welcome screen and composer available. Esc
recalls the first waiting message into an empty composer and pauses any remaining
queue. If the composer already has text or attachments, both are kept and the paused queue
opens for editing. Preparation continues; a recalled message is never sent unless
you submit it again. Input history and conversation messages are recorded only
when a prompt is dispatched. Unused prepared sessions are cleaned up on normal
close. Preparation errors remain visible and preserve pending input.

Esc cancels the running task from the main conversation. With a panel or slash
suggestions open, Esc closes that view first and keeps the draft. Cancellation
keeps the queue paused until the operation actually finishes. After 20 seconds
without a terminal response, the TUI reports that it is still waiting and stays
open; it does not stop the backend automatically. Press Ctrl+C twice within two
seconds to leave, or use Ctrl+Q, Ctrl+D with an empty composer, or `/exit`.

With a popup open or no composer text selected, the first Ctrl+C shows an exit confirmation:
it does not cancel the task or clear your draft. Another key, a paste, or a mouse action dismisses confirmation;
waiting more than two seconds also resets it. Explicit exit requests use bounded
backend cleanup. Saved drafts and queued messages can be recovered after closing;
restoring them does not send a prompt automatically.

While idle, press Esc twice within 600 ms to clear the entire draft as one
undoable edit. With an empty composer, double Esc (or `/rewind`) opens a searchable
list of messages with saved conversation checkpoints, newest first. Enter creates
a separate conversation ending just before the selected message and places its
original text back in the composer. Edit it and press Enter when ready; selecting
a message does not send it. The original conversation remains available through
`/sessions`. This restores conversation context and agent state; workspace files
stay as they are. Checkpoints are recorded for new messages; older messages that
predate this feature cannot be rewound.

An Esc used to cancel work, recall a waiting message or dismiss a panel,
suggestions or search highlighting does not start a double-Esc sequence. Other
keys, paste and mouse input also reset the sequence. This prevents finishing a
cancellation or closing a panel from accidentally clearing a draft or rewinding.

Ctrl+F searches the current conversation, including collapsed tools. Enter jumps
to the selected result and expands its tool if needed; Ctrl+End returns to the
latest output and clears highlighting. The composer and cursor survive searches.
Picker lists support mouse selection and scrolling.
Clipboard shortcuts, right-click in the composer, and bracketed paste insert multiline
text without sending it. In a searchable panel, right-click its search field to paste
into that field. Releasing or dragging the right button does not paste again.
Windows reads native clipboard text and PNG/bitmap images. Linux image paste uses
`wl-paste` on Wayland or `xclip` on X11; text paste also supports `xsel`. Your
terminal's text Paste command works without these utilities. If reading fails,
the draft is kept and an error is shown.
Click the composer to position the cursor; drag to select text, including wrapped
Chinese and emoji. Shift+click extends the selection. With no popup open, Ctrl+C
copies the selection; Ctrl+X cuts it only after copying succeeds. If the draft changes before copying
finishes, its text is kept. Failed clipboard access also keeps the selection.
Copy, cut and paste requests run in order, including a paste requested while copying.
A pending paste is cancelled if you change its input, move the cursor, or switch conversations.
For transcript text, use your terminal's mouse-selection override (often Shift+drag).
Long user messages initially occupy at most three visual rows. Expanded tools show
the first two and last three rows of long output; click the omitted-lines hint to
show all rows, or use Ctrl+T for the full tool inspector. Search opens the entire
matching message or tool output before jumping to it. Conversation diffs remain
available in full; Git preview limits are described below.

Commands: `/new`, `/sessions`, `/resume ID`, `/drafts`, `/tree`, `/rewind`, `/search [text]`, `/queue`,
`/steer TEXT`, `/attach [PATH]`, `/attachments`, `/retry`, `/reconnect`, `/model [ALIAS]`,
`/mode [code|plan]`, `/permissions [ask|session]`, `/tasks`, `/stop JOB-ID`, `/tools`,
`/changes`, `/undo`, `/status`, `/logs`, `/help`, `/cancel`, `/exit` and commands
advertised by the agent.
The interface follows the layout observed in Grok Build 1.0.77: a grayscale
canvas, full-width conversation, compact bordered composer, model on its bottom
border and session title on its top border. The welcome menu centers available
actions; typing immediately focuses the conversation. Alt+B opens the sidebar when the terminal has at
least 104 columns and 20 rows. Smaller windows retain the conversation and composer.
The composer wraps long input and grows up to eight visual rows, including Chinese
and emoji, without inserting newlines into the message. Up/Down and Home/End move
through the visual rows; Shift extends the selection. Taller drafts scroll to keep
the cursor visible. At the first row, Up recalls a submitted message; at the last
row, Down advances through recalled messages and returns to your unsent draft,
including its cursor and undo history. Editing a recalled message makes it a new
draft. Ctrl+P / Ctrl+N recall history from any row.
Switching conversations while browsing input history preserves the hidden draft
and its cursor/undo history; Down past the latest recalled entry returns to it.
Submitted inputs are saved immediately in `.aster/tui-input-history.jsonl`
(up to 1,000 entries and 8 MiB).
Restarting in the same workspace loads this history without resuming a conversation.
On first use, recent user inputs from saved sessions are imported automatically.
Concurrent terminals merge their submissions; drafts and unsent queue entries are
not saved as submitted history. Shortcut hints sit below it and
running status sits above it. Inline commands support mouse selection and wheel
navigation; Ctrl+K, session and search pickers size themselves to their contents.
`NO_COLOR` uses terminal default colors with text markers and emphasis.
See [reference observations](docs/grok-reference.md) for the evidence and remaining differences.

## Steering during execution

Enter queues the draft for the next turn. Alt+S or `/steer TEXT` instead sends a
text correction to the current turn. The receipt distinguishes **received** from
**included in the model request**. The correction joins the next model request at
a safe execution boundary; a cell already executing can finish before it takes
effect. It does not undo that cell's effects. Use Esc to stop the foreground task.

Steering accepts text only. Attachments stay with messages queued for a later turn.
Rejected or interrupted corrections remain available for review, and typing a new
draft while waiting for acknowledgement preserves both inputs.

## Files and images

Type `@` after whitespace or at the beginning of a line, press Alt+A, or use
`/attach` to open the file picker. Search respects Git ignore rules and returns a
bounded list; narrow the query when necessary. `/attach src/parser.py` selects a
relative path, while an explicit absolute path can select a file outside the
workspace. Enter attaches the selection without sending it.

Ctrl+V attaches a copied image in the composer. Click the × beside an attachment
to remove it immediately, or use Alt+Backspace to remove the last attachment.
With an empty composer, Backspace also removes the last attachment. Alt+Backspace
does nothing when there are no attachments. Click a name to preview it, or click
`+N` to inspect hidden attachments. Click Undo or immediately press Ctrl+Z to
restore the last removed item at its original position, including its captured
bytes. This one-step undo lasts until the next text edit, successful attachment
addition, draft replacement or submission, or conversation switch. Inside
`/attachments`, Delete removes the selected item and Ctrl+Z restores it. These controls
affect the current draft; queued and submitted messages keep their snapshots.
Files and images are
captured when attached, so later source-file edits do not change the next prompt.
Both the prompt and saved conversation retain that snapshot. Sending waits for
pending file reads.

The attachment inspector displays PNGs as pixel images on Windows terminals that
report Sixel support and cell geometry, including Windows Terminal 1.24. Images
fit the available panel without enlargement, within a 1024-pixel limit per side.
Detection does not delay startup; decoding and encoding run in a background worker
and reuse a bounded cache. Other terminals retain the character thumbnail;
`NO_COLOR` retains the metadata-only view.

To inspect full-size detail, click the image or **Open original**, or press Enter with
an image selected. The system image viewer opens the captured original, where
you can zoom in. This also works for JPEG/GIF/WebP, `NO_COLOR`, and images that
cannot be shown as a terminal thumbnail. The original image snapshot is included
when sent; the terminal thumbnail does not reduce its quality.

A message accepts at most eight attachments: UTF-8 text up to 256 KiB per file,
or PNG/JPEG/GIF/WebP images up to 8 MiB each, within a 16 MiB aggregate prompt
limit. Directories, binary text, unsupported images and models without configured
image input produce an error while preserving the draft. Selecting a file adds
its context; it does not grant permission to execute code.

## Session tree

`/tree` opens a searchable tree of user messages and assistant replies. Shared
history appears once, with alternative paths branching from it; `●` marks the
current node. Type to search, use Up/Down to select, Enter to navigate, and Esc to
close. Search checks complete saved message text, including content beyond the
display preview, and keeps matching nodes' ancestors visible so branches remain
clear.

- Select a user message to restore the context before it and recall its original
  text for editing. Press Enter in the composer when ready to send it.
- Select an assistant reply to restore the context after that reply and continue
  from there.

Existing branches remain available in the tree after navigation and after
reopening the conversation. The model uses the selected path when you next send
a message. Navigation restores saved context without calling the model or
generating a summary, and leaves workspace files unchanged.

Your existing draft is kept. If it would be replaced by recalled text, that text
goes into the paused queue instead. Finish or stop active work and resolve queued
messages before navigating. Input and work state are checked again when the
selected branch finishes loading; a failed input save prevents the switch.
Nodes without a usable saved restore point, including unfinished progress,
background notices and some older messages, remain visible but cannot be
navigated to. A later background notice does not remove the restore point of an
already completed reply.

## Recovery

The composer, cursor, attachment snapshots, queue and interrupted input are saved
separately for each conversation under `<workspace>/.aster/tui-drafts/`. Starting
`aster` or `aster tui` opens a new conversation with an empty composer and queue.
Use `--continue`, `--resume ID` or `/sessions` to reopen a saved conversation and
restore its input.

`/drafts` opens a menu of drafts containing unsent input, newest first. This
includes drafts saved before a first message was sent. Select a draft and press
Enter to restore it.

Restored queues stay paused until Ctrl+S. `/retry` recalls only the current
conversation's interrupted input for editing before Enter sends it. If another
draft is present, the recovered message goes into the paused queue.

`/reconnect` restarts the ACP connection and restores a saved conversation when
available. Stop an active prompt before reconnecting. After an ambiguous delivery
failure, inspect the restored conversation before resending: reconnect never
assumes that an interrupted request did nothing. Autosave is bounded to 128 queued
messages and 32 MiB per saved draft, and reports write conflicts or storage errors.
It preserves newer saved input from another terminal instead of silently replacing it.
Messages whose recovery save fails remain in the composer or paused queue;
autosave retries transient storage errors. Exit also checks that the latest input
was saved, including edits made while the backend was closing. A failed save keeps
the interface open; resolve the storage error and retry exit. If the session has
already closed, `/reconnect` remains available.

## Session settings and approvals

`/model`, `/mode` and `/permissions` open session controls. Supply an argument to
select a configured model alias, `code` or `plan` mode, or `ask` or `session`
approval policy. Changes apply to the live session; they do not rewrite the model
registry. Finish or stop foreground and background work before changing settings.

With `ask`, execution pauses for a compact review. Literal shell commands and
file operations show their arguments first; Tab opens the complete Python cell.
Dynamic or incomplete previews show full code immediately. PgUp/PgDn scroll the
preview while the choices remain visible. Choose Allow once, Allow all code for
this session, or Reject. The initial selection is Allow once; Enter confirms the
selected choice, and Esc rejects the request. Approval covers the entire Python
cell, including any files or processes it accesses. Session consent permits all
code until the session closes or reconnects.
Approval dialogs close when their task ends or the backend disconnects.

Ordinary conversation can return a written answer without executing code or
requesting approval. A greeting combined with a coding task still follows the
normal tool and approval flow. Native Windows shell commands use PowerShell.

Plan mode disables all generated Python and tool execution. It can reason from
conversation context and supplied file/image snapshots, and cannot inspect new
workspace files through tools. Native and sandbox execution keep their existing
backend behavior; choosing Plan does not create a separate read-only filesystem.

## Reviewing and undoing file changes

Open Git Changes with Ctrl+G, select a file, then press Ctrl+U. Choose all of its
unstaged hunks or one hunk, inspect the proposed patch, and select Apply. The
confirmation starts on Cancel. Foreground and background work must be stopped.

Rollback accepts existing tracked files with unstaged-only UTF-8 modifications,
up to 4 MiB with a complete patch no larger than 256 KiB. It excludes staged,
new, deleted, renamed, conflicted, binary, symlink and custom-filter/encoding
changes. Unselected hunks and the Git index are preserved. If the selected file
or index changes after review, refresh and prepare the rollback again.

Original bytes are saved in `.aster/tui-rollbacks/` before replacement. `/undo`
finds the latest available backup, including after restarting, and asks for
confirmation before restoring it. Undo refuses to overwrite edits made after
rollback. Conversation `/rewind` continues to restore context independently of
these file operations.

## Task progress

Alt+T or `/tasks` displays the agent's actual plan steps, current step, waiting
reason and background jobs with bounded output previews. Ctrl+R refreshes the
panel. Select a running job and press Ctrl+X, or use `/stop JOB-ID`, to stop that
specific background job while keeping the foreground request and other jobs
running. Concurrent jobs sharing a channel retain separate IDs and can be stopped
individually. Finished or removed job IDs cannot stop a newer job with the same name.

Plan steps appear when the agent records them; elapsed time alone does not imply
a completed step or an overall completion percentage. Python cards use a leading
purpose comment when supplied and retain their code, output and measured time.

## Behavior and scope

- Sessions use `<workspace>/.aster/sessions`, shared with other ACP clients. Resume
  replays messages and restores agent context. A session can have only one writer.
  New sessions use the first message as their title unless a title was already set.
  Existing unnamed sessions are not migrated.
- Resume restores saved Python calls, terminal output and file diffs in their
  original order, including failed and interrupted tools. Terminal/file events
  are recorded starting with this version; older sessions can only restore the
  events they actually saved. Restoring does not execute the tools again.
- `/changes` reads the Git working tree in the background, with separate staged,
  unstaged and untracked entries, searchable filenames, and a selected patch.
  Ctrl+R refreshes the snapshot; Tab switches to the conversation's saved edits.
  Opening and refreshing the view only inspect Git; Ctrl+U starts the reviewed
  rollback flow described above. Ignored files are excluded; binary files and
  truncated previews are labelled. Previews are bounded to 500 entries, 256 KiB
  each and 8 MiB total.
- Completion and Responses clients can show final-answer text as provider tool
  arguments arrive. The validated final reply replaces its preview in the same
  message. Cancellation marks the preview incomplete; unvalidated preview text
  is not saved as an answer. Ordinary written final answers use the separate
  `return_result` tool, with `message` first. The coding strategy rejects definite
  literal final answers embedded in Python before executing the cell and asks the
  model to use that reply tool. This also applies to sandbox execution. A rejected
  cell may require one correction request; an ordinary reply needs no extra call.
  Replies actually computed inside Python still arrive together without a model
  rewrite. The routing check recognizes direct literals and simple bindings in a
  cell; arbitrary expressions, branches, and standalone progress messages keep
  normal execution behavior. Providers configured with streaming disabled, or
  returning all arguments in one chunk, continue to send whole replies.
  Internal reasoning and Python source are not shown as answer text.
- Replies render headings, nested lists, quotes, links with visible destinations,
  fenced code and tables with Unicode column widths. Fenced code hides delimiter
  and language-title rows, preserves indentation and blank lines, and highlights
  recognized languages including Rust, Python, JavaScript, JSON and Bash.
  Unknown languages stay readable as plain text; `NO_COLOR` disables syntax colors.
  Durations display seconds with two decimal places. Python and terminal tools
  use backend execution timing, so buffered notifications do not collapse durations
  to zero. Completed-only events without timing omit the duration.
  The display redraws when
  events arrive and updates elapsed time several times per second while working.
- Backend stderr is drained independently and retained in bounded `/logs` output.
  Unexpected backend exits also print the recent diagnostic tail after restoring
  the terminal. Protocol stdout remains JSON-RPC only. Terminal control sequences
  are filtered.
- Closing cancels an active prompt, waits for its terminal result, closes the
  session, then shuts down and reaps the owned backend. Windows Job Objects also
  clean up descendants, including nested jobs. Linux cleanup targets the backend's
  dedicated process group; processes that deliberately detach from that group are
  outside its scope. Hung operations and shutdown have time bounds. Native
  execution is the default; sandbox remains opt-in using the existing Python implementation.

## Development and verification

```powershell
uv sync --extra tui
cargo test --locked --manifest-path packages/aster-tui/Cargo.toml
cargo clippy --locked --manifest-path packages/aster-tui/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path packages/aster-tui/Cargo.toml -- --check
uv run pytest packages/aster-cli/tests/test_tui_command.py
```

Rust tests include controller transitions, CJK/paste/draft retention, TestBackend
rendering at several terminal sizes, noisy/broken subprocesses, bounded cleanup,
attachment snapshots, approvals, runtime control isolation, targeted background
stops, recovery and file rollback. `tests/session_controls.rs` exercises the public
controller and protocol boundary. Real Python ACP prompt/replay/cancel flows use
FakeLLMClient (no model calls).
They locate the repository `.venv`, or use `ASTER_TEST_PYTHON` when explicitly set.
`cargo run --manifest-path packages/aster-tui/Cargo.toml --example preview -- 120 36 work`
exports actual rendered cells as JSON for visual checks; modes include `welcome`,
`work`, `tools`, `commands`, `slash`, `changes`, `sessions`, `search`, `queue`,
`markdown`, `sidebar`, `compose`, `compact`, `python-error`, `steering`, `files`,
`attachments`, `image`, `approval`, `approval-expanded`, `approval-fallback`,
`tasks`, `progress`, `recovery` and `rollback`.
