# aster-acp

**Run the Aster coding agent inside your editor.** `aster-acp` is an
[Agent Client Protocol](https://agentclientprotocol.com) server, so any
ACP-speaking client — Zed today — can drive the same agent the terminal host
uses: CodeAct, repository tools, a persistent shell, installed skills, workspace
slash commands and durable sessions, with file edits and terminal commands
surfaced as structured activity.

It hosts `aster_cli.coding.CodingAgent` directly. Repository instructions
(`AGENTS.md`), coding tools, summarization, installed `aster.skills` entry points
and semantic file and terminal activity therefore have no separate ACP
implementations — fix something here and the terminal host gets it too.

This is new and we would like it exercised. If something breaks, please say so.

## Install

```bash
uv sync --extra acp    # from the repository root
```

> **Windows:** ACP runs natively, including stdio transport, durable sessions and PowerShell tools. No WSL environment or platform override is required. See the [main README](../../README.md#installation).

There is no default model. Set `ASTER_MODEL` or pass `--model`, or the command
exits with a usage error.

Startup uses LiteLLM's bundled pricing and model catalogue to avoid waiting for
a remote download before opening a session. Set
`LITELLM_LOCAL_MODEL_COST_MAP=False` before launch to request the remote catalogue.

On Windows, startup caches the installed LiteLLM and OpenAI Python code in a
single archive to reduce the number of individual file checks before the first
message. The cache is built once after SDK installation and reused across
reboots; Python and SDK wheel changes select a new cache. Its files live under
`~/.cache/aster/provider-code/` (or `XDG_CACHE_HOME/aster/provider-code/`).
SDK resources and native extensions use their installed paths. Editable SDKs
and project sources use ordinary imports. Set `ASTER_PROVIDER_CODE_CACHE=0` to
disable this cache when troubleshooting or editing installed SDK source files.
An unavailable or unwritable cache falls back to ordinary imports.

## Quick start: Zed

Zed launches ACP agents as "external agents". Add Aster to `settings.json`
(`cmd-,`):

```json
{
  "agent_servers": {
    "Aster": {
      "type": "custom",
      "command": "uv",
      "args": ["run", "--project", "/absolute/path/to/aster", "--package", "aster-acp", "--", "aster-acp"],
      "env": {
        "ASTER_MODEL": "nvidia_nim/nvidia/nemotron-3-super-120b-a12b",
        "NVIDIA_API_KEY": "nvapi-..."
      }
    }
  }
}
```

Replace `/absolute/path/to/aster` with this checkout's path.
Open a repository, then pick **Aster** from the `+` menu in the agent panel. Zed
runs the command with your worktree as its working directory, so repository
instructions, project skills and sessions resolve against the open project.

Credentials go in `env` here rather than in Zed's own settings: the agent is a
separate process and inherits only what Zed passes it. Use a secret-manager
wrapper as the `command` if you would rather not put a key in `settings.json`.

The equivalent command from the source checkout is:

```bash
uv run --project "$PWD" --package aster-acp -- aster-acp
```

### MCP servers do not carry over from Zed

**Remote MCP servers you authenticated inside Zed are not usable from an ACP
agent.** Zed holds those OAuth tokens itself and does not pass them down, so a
server showing a green indicator in Zed's own UI arrives at the agent either
with no tools at all or with nothing but its `authenticate` /
`__complete_authentication` stubs. Local stdio MCP servers are unaffected.

This is a known Zed limitation, tracked in
[zed-industries/zed#54410](https://github.com/zed-industries/zed/issues/54410)
(open, labelled `area:ai/mcp` + `area:ai/acp`). A maintainer has said the
plumbing largely exists and the work is queued, but as of this writing it is
unresolved.

Configure the MCP server directly for Aster instead — through Aster's own
`.mcp.json` — and it works normally, because the agent then owns the
connection and its credentials rather than borrowing Zed's.

## Launching the server yourself

```bash
aster-acp --model nvidia_nim/nvidia/nemotron-3-super-120b-a12b
```

This is a JSON-RPC server, not an interactive program: it speaks ACP on
stdin/stdout and exits when its input closes, so running it in a terminal
without a client does nothing. Launch it this way to wire up an ACP client
other than Zed, or to watch the diagnostics it writes to stderr while a client
drives it. `--model` accepts any LiteLLM model name or configured Aster alias.

### Startup timing

Set `ASTER_STARTUP_TIMING=1` in the agent's environment to diagnose slow loading.
Lines prefixed with `[aster-startup]` go to stderr (and Zed's agent log), leaving
the ACP protocol on stdout unchanged. They record import, configuration, model,
storage, MCP and agent preparation phases, including elapsed time and process
CPU time. Each phase logs its start, so an unfinished phase remains visible.
Only stage names, process IDs and timings are recorded.

These timings work on any fresh process. Comparing a first launch after boot
with later launches helps distinguish cold startup from recurring costs; a
fresh process alone does not reset the operating system's filesystem cache.
Unset the variable to disable the diagnostic output.

### Waiting for a reply

Each prompt shows a separate activity card while Aster prepares the execution
environment, waits for a model response, or works on the request. Titles follow
runtime events; they do not estimate a percentage or a remaining duration. The
card completes when the first answer is ready to send. If the turn ends earlier,
it shows completion, cancellation, or failure instead. These updates are optional
ACP `other` tool activity, separate from answer text and durable conversation
history. A client rejecting an activity update does not suppress the answer.

Set `ASTER_PROMPT_TIMING=1` to diagnose time from `session/prompt` to the answer.
`ASTER_STARTUP_TIMING=1` also enables these records for an end-to-end diagnosis.
Lines beginning with `[aster-prompt]` go only to stderr. Each accepted prompt has
an opaque `turn_id`, a process ID, an execution backend and elapsed wall time:

| Stage | What it measures |
| --- | --- |
| `received`, `accepted` | Entry to the adapter and acquisition of this session's foreground turn |
| `prefill_start`, `prefill_end` | Input inspection/setup, including lazy sandbox preparation |
| `model_start`, `model_end` | Each logical model call; `call` pairs its boundaries and `outcome` reports its exit |
| `first_model_response` | The first complete, valid response available to the runtime |
| `first_status` | First successful activity write to the ACP transport |
| `first_answer` | First successful nonempty answer write to the ACP transport |
| `dispatch_end` | Foreground execution returns, including its runtime cleanup |
| `complete` | Final updates have reached the transport, or delivery has failed; includes the outcome |

The model client currently collects a streaming response before returning it.
`first_model_response` therefore does **not** measure the provider's first token.
Likewise, transport completion cannot measure when Zed paints the text. Model
call time includes client preparation, middleware, queueing and retries; it is
not pure network latency. If cancellation or failure prevents an answer, there
is no `first_answer` record. The host's stop/error note is not counted as an
answer, and status updates are never counted as answer latency.

Only fixed stage/outcome names, opaque correlation IDs and timings are logged;
these records contain no prompts, responses, Python source, paths, model
configuration or reasoning. Both timing switches are off by default.

### Usage notifications

Each model response sends a `usage_update`. With reported token usage, `used` is
the input-token count and `size` is the known context-window size, or `0` when
the limit is unknown. Clients should show a token count when `size` is zero;
they must not interpret that value as a full context window.

Before creating or loading the live agent, and before adopting a replacement
model, ACP refreshes optional endpoint metadata. Explicit OpenAI-compatible
routes use one bounded `GET /models` and an exact wire-model ID match. The
two-second lookup does not perform inference or save configuration; failure
leaves configured and LiteLLM limits available. Explicit context settings win,
and automatic summarization uses the same resolved window as usage reporting.

If the response has no usage report, the notification has `used: 0`, `size: 0`
and `_meta.aster.usageAvailable: false`. Those zeros are protocol placeholders:
the terminal displays unavailable statistics rather than a measured zero.

`cost` contains the estimated USD total for calls since this session was opened
in the current server process. It is omitted once any response lacks a valid
price or usage, since the accumulated amount would then be incomplete. An
explicitly reported zero is valid. Loading a saved session does not reconstruct
historical costs, and new sessions start a separate accumulator.

## Sandbox execution

Pass `--sandbox` to run CodeAct's generated Python in a microsandbox microVM.
From the repository root, install both extras with `uv sync --extra acp --extra sandbox`.
On Windows, launch the ACP server with:

```powershell
uv run --project . --extra acp --extra sandbox aster-acp --model my-alias --sandbox
```

Add `--sandbox` to the command arguments in your ACP client, or set
`ASTER_SANDBOX=true` in the server's environment. `--no-sandbox` explicitly
overrides that environment setting; native execution is the default and works
on Windows with PowerShell tools.

The adapter creates a separate strategy for each ACP session and sets
`SandboxConfig.workspace` to that session's repository root. On Windows, the
guest sees this root at `/workspace`. Default guest permissions allow read-write
access to the workspace and read-only access to interpreter files, with internet
sockets blocked. Framework callers can customize image, resources, network and
extra file rules through
`CodingACPAdapter(llm_factory, sandbox=True, sandbox_config=SandboxConfig(...))`;
the workspace is always resolved from the ACP session.

ACP transport, model requests, workspace skill imports and brokered `self.*`
tools run on the host. The microVM isolates generated Python's direct
operations. Cancelling a running cell or closing its session tears down the VM.
The real stdio tests cover guest execution and cancellation using simulated
model responses.

Prepared dependency snapshots are shared across workspaces using the same image
and dependency versions. They live under `MSB_HOME` (default `~/.microsandbox`),
survive process restarts, and use a process lock to prevent duplicate preparation.
Snapshots from older Aster versions are reused when their environment matches.
Changing the image or dependencies creates a separate cached environment.

## Opening a repository runs code from it

**Creating a session imports Python from the workspace, before you send a
prompt.** This is deliberate — it is how workspace skills work — but it means
opening a folder is enough to execute code it contains. Treat opening a
repository with Aster as equivalent to running its build.

Three paths load workspace code at `session/new` and `session/load`:

- **Skill roots.** Every `.py` file under `.agents/skills`, `.cursor/skills`,
  `.claude/skills`, or `.claude/commands` is imported. Module-level code runs
  during import, before anything checks whether the file defines a skill, so the
  contents are irrelevant.
- **Workspace settings.** `<workspace>/.aster/settings.yaml` and the legacy
  `.aster/config.toml` may name *additional* skill roots. Those paths are not
  confined to the workspace: a relative path escaping it, an absolute path, or a
  symlink is accepted as written.
- **Libraries.** `<workspace>/.aster/libs/<package>/` is imported and its
  directory is prepended to `sys.path` for the life of the process. One ACP
  server serves several workspaces, so a package name there can shadow the same
  import for later sessions on other workspaces.

The agent runs as you, in a process holding your model credentials. There is no
consent prompt on these paths. Workspace imports happen on the host in both
execution modes.

**Open repositories you would run.** For anything else, use an OS-level sandbox,
or start a separate server per workspace with credentials scoped to that task.

## How it behaves

ACP uses standard input and output for JSON-RPC. Diagnostics are written to
standard error. Without `--sandbox`, generated Python shares the agent's process
environment, including model credentials. With `--sandbox`, generated Python
runs in the microVM under the configured guest permissions. Shell commands and
other brokered tools still execute on the host under the agent's permissions;
launch the server with credentials and host access scoped to the session.
Cancellation stops cooperative local work immediately. An in-flight provider
request may finish in the background when its client does not support
transport-level aborts. Slash commands run on the agent's event loop so they
have the same semantics as the native TUI and can safely start agent jobs. An
async command is cooperatively cancellable; a synchronous command that blocks
that loop cannot be preempted by the current in-process adapter. The planned
one-process-per-agent boundary is the safe kill mechanism for that case.

## Sessions and skills

Each ACP session has an independent live agent and allows one foreground prompt
at a time. Sessions are stored in `<workspace>/.aster/sessions`, where the TUI
and ACP adapter can share list and replay metadata. These files are inside the
workspace trust boundary: a repository can supply session records that appear
in `session/list` and are replayed as conversation history by `session/load`.
Open only repositories whose code and conversation history you trust. The
adapter also advertises session close; closing a live session preserves its
durable history.

### Session tree extension

`initialize` advertises Aster's message-tree extension through
`_meta.aster.sessionTree`. Clients using this capability can browse saved user
messages and assistant replies, then restore a chosen path while keeping the
other branches. The native terminal exposes it as `/tree`.

Call `_aster/session/tree` with `{"sessionId": "current-runtime-id"}`. The response
contains `nodes` and `currentNodeId` (a node ID or `null`). Each node has:

| Field | Meaning |
| --- | --- |
| `id` | Stable message-node ID |
| `parentId` | Parent message-node ID, or `null` at the root |
| `role` | `user` or `assistant` |
| `preview` | Text for the tree display; use the navigation result to recall the original prompt |
| `navigable` | Whether a usable restore point exists for this node |

When `_meta.aster.sessionTreeSearch` is advertised, the same request accepts a
`query` string of at most 4096 characters. Search matches complete saved message
text, including content beyond the display preview:

```json
{"sessionId": "current-runtime-id", "query": "a phrase from the message"}
```

The response still contains the entire `nodes` graph and its real
`currentNodeId`. It also echoes `query` unchanged and returns `matchedNodeIds`
for direct matches. Show those matches with their ancestors; the current node
need not be a match. An empty or whitespace-only query matches every node.
Discard results for an obsolete query or session before updating the selection.
Requests without `query` keep the original response shape.

To navigate, call `_aster/session/navigate` with the current node returned by the
tree request:

```json
{
  "sessionId": "current-runtime-id",
  "nodeId": "selected-node-id",
  "expectedCurrentNodeId": "current-node-id"
}
```

`expectedCurrentNodeId` is required, including when its value is `null`. A stale
value is rejected; refresh the tree before trying again. Navigation returns:

```json
{
  "sessionId": "target-runtime-id",
  "prompt": "Original user text to edit"
}
```

Load the returned `sessionId` through ordinary `session/load` to install the
target runtime. For a user node, the restored context ends before that message
and `prompt` contains its original text. For an assistant node, context ends
after the reply and `prompt` is `null`. Put recalled text in the editor for review;
do not dispatch it automatically or overwrite an existing draft. Navigation
does not call the model, generate a summary, or roll back workspace files.

The logical tree uses Aster's immutable snapshots and separately owned session
databases. Navigation copies the selected saved boundary into a fresh session,
including when selecting the current reply, so clients must load the returned
handle. A restore point belongs to the specific saved reply, so later background
messages do not replace it. Progress messages and older nodes without a verified
restore point can be listed with `navigable: false`. Hosts must finish active
work, resolve queued input and save drafts before switching. Recheck these
conditions after `session/load` finishes, before committing the new current
session; if the switch is rejected, keep the source and close the loaded target.
See the
[terminal guide](../aster-tui/README.md#session-tree)
for the user-facing behavior.

The current stdio adapter hosts those live agents in its own process. That is
an adapter-private implementation detail rather than part of the durable
session API: the live-session registry is isolated inside `aster-acp` so it can
later be replaced by handles to an agent daemon without changing stored
sessions, the shared coding agent, or the ACP protocol surface.

Python skill packages use the interpreter's normal import machinery. Multiple
sessions may use distinct skill package names, but two workspaces must not load
different checkouts under the same top-level Python package name in one ACP
server process. Launch a separate stdio server for those workspaces. A future
one-process-per-agent daemon will make that isolation an OS process boundary.

Installed `aster.skills` entry points are loaded into the shared skill registry
but remain opt-in. The agent can activate a relevant skill with
`self.skills.activate(["name"])`. Stdio MCP servers supplied by an ACP client
are registered and activated as `mcp.<name>` skills for that session.

Workspace and user skill roots are shared with the terminal host through
layered `settings.yaml`. New configuration should use:

```yaml
coding:
  additional_skills_dirs:
    - ../aster-skills
```

The existing `tui.additional_skills_dirs` key remains supported during the
migration, as does the older project-local `.aster/config.toml` key
`[tui].libs_dirs`. Packaged libraries declared through `aster.skills`, `SKILL.md`
skills, and standalone Python skills are discovered from each configured root.
Loaded `@slash_command` methods are advertised through ACP and matching
`/command arguments` prompts are dispatched through the shared typed command
router. Command discovery is refreshed when loaded skills change.

The current adapter accepts text and resource-link prompts plus stdio, HTTP,
and SSE MCP servers forwarded by an ACP client. ACP-transport MCP proxies,
additional workspace directories, images, and embedded resources are not
advertised yet. An unavailable, duplicate, or unsupported MCP server is skipped
with a session warning so it cannot prevent a new or restored Aster session from
opening.
