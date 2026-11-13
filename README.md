# Aster

**A terminal coding workspace for working with an AI agent in your repository.**

Aster brings conversations, code execution, tool output, Git changes, and saved
sessions into one keyboard-driven interface. Follow a task as it runs, inspect
its changes, and return to the same conversation later.

Aster combines a Rust / Ratatui interface, a Python agent framework, and an ACP
runtime shared by the terminal and editor integrations.

**[Run Aster](#run-aster-from-a-source-checkout)** · **[Terminal guide](packages/aster-tui/README.md)** · **[Framework docs](docs/README.md)** · **[Framework quick start](#quick-start)** · **[Notebook tutorials](notebook_tutorials/README.md)** · **[Examples](examples/README.md)**

## Run Aster from a source checkout

From the repository root, use Python 3.12–3.13,
[uv](https://docs.astral.sh/uv/getting-started/installation/), and Rust 1.88+:

```powershell
uv sync --extra tui
cargo build --locked --release --manifest-path packages/aster-tui/Cargo.toml
uv run --extra tui aster connect
uv run --extra tui aster
```

Skip Connect if you already have a configured model. Aster uses the saved default,
or the only configured chat model. With several aliases, choose a default in
your project's `.aster/settings.yaml`:

```yaml
tui:
  default_model: my-model
```

Add `-w PATH` to open another workspace. The build creates
`aster` (`aster.exe` on Windows) in `packages/aster-tui/target/release/`, which the
Python launcher finds automatically. Rebuild after changing Rust sources.

Once the Python launcher is on PATH, simply run `aster`. In a local Windows setup
where `%USERPROFILE%\.local\bin` is already on PATH, export the installed launcher:

```powershell
Copy-Item .venv\Scripts\aster.exe "$env:USERPROFILE\.local\bin\aster.exe"
aster
```

`aster` and `aster tui` start a new conversation. Use `--continue` to reopen the
most recent saved conversation or `--resume ID` to choose one. Inside the terminal,
`/sessions` selects a saved conversation and `/drafts` selects saved input to recover.
`/tree` shows saved message branches so you can continue from an earlier question
or reply while keeping the original path. See the [session tree](packages/aster-tui/README.md#session-tree).

Re-export the launcher if you move the checkout or recreate its environment.
`--model my-model` overrides `ASTER_MODEL` and the saved default. `aster tui`
explicitly opens the same interface; `aster connect`, `aster config`, and other
subcommands run the toolkit. `ASTER_TUI_BIN` selects a native binary by absolute path.
See the [terminal guide](packages/aster-tui/README.md) for controls and sandbox setup.

Python packages and imports use the `aster` namespace. Project configuration and
sessions live in `.aster/`; user configuration lives in `~/.config/aster/`.
Microsandbox is an optional execution dependency.

## Aster framework

Aster's Python framework supports agents whose state, tools, prompts, and typed
interfaces live in one Python class:

```python
from aster import Agent

# The agent is a Python object.
class SupportAgent(Agent):
    """You are a support agent."""

    # State lives on the object. Fields are typed.
    order_db: OrderDB

    # Ordinary method. Just Python.
    def is_refund_eligible(self, order: Order) -> bool:
        return order.delivered and order.days_since_delivery <= 30

    # Agentic method: the runtime hands this to an LLM.
    async def triage(self, message: str, order: Order) -> Ticket:
        """Create a typed support ticket."""
        ...
```

**What's happening here:**

- **Agents are Python objects.** Fields are state, methods are capabilities, docstrings are prompts, type annotations are contracts.
- **`...` bodies are LLM-driven.** A method with `...` becomes an agentic loop; a real body stays deterministic Python.
- **Generation commits one result.** Deterministic generators are supported, but generated streams need a separate stream contract and are rejected clearly.
- **Code as action.** The model acts by writing Python in a Jupyter-style REPL with access to `self`, imports, and helpers — Python methods and type annotations supply the callable interfaces, reducing the need to write separate tool-schema definitions.
- **Pythonic and agent-ready.** Typed I/O with auto-retry, live-object arguments passed by reference, and model-callable context and event APIs — designed around agent-oriented Python workflows.

This design supports familiar Python testing, tracing, refactoring, and version-control workflows — **just like the rest of your software**. Read [the paper](https://arxiv.org/abs/2607.20709) for the design principles and evaluation results.

Want to see how the pieces compose? Take the [**10-minute tour**](docs/tour.md), from one thinking method through tools, typed contracts, deterministic orchestration, and object composition.

## Installation

Install from this source checkout with [uv](https://docs.astral.sh/uv/getting-started/installation/).
Run these commands from the repository root; sibling distributions resolve from
the local workspace:

```bash
uv sync
uv sync --extra tui             # terminal launcher and ACP runtime
uv sync --extra viewer          # trace viewer
uv sync --extra cli --extra memory   # combine optional components
```

Supported platforms are Linux and native Windows. Host tools use PowerShell on
Windows and bash on Linux. Sandbox execution uses the optional microsandbox
runtime; enable it with `uv sync --extra tui --extra sandbox`. macOS is outside
the current support scope.

| Package | Core extra | What it adds |
|---|---|---|
| `aster` | — | Python agent framework |
| `aster-cli` | `cli` | `aster` launcher, model setup, configuration, traces and evaluations |
| `aster-cli[tui]` | `tui` | Shared Python backend for the Rust terminal — [build guide](packages/aster-tui/README.md) |
| `aster-acp` | `acp` | Agent Client Protocol server for editor integrations — [setup](packages/aster-acp/README.md) |
| `aster-memory` | `memory` | Long-term memory subsystem (`MemoryManager`) |
| `aster-bench` | `bench` | Benchmark agents and the Harbor runner |

`uv sync --group dev` installs the evaluation pipeline and development tools too.
To use the core framework from another Python project, add this checkout by its
absolute path: `uv add --editable /absolute/path/to/aster`. See
[Releasing](RELEASING.md) for building distributions and configuring a release channel.

## Quick Start

### ⚠️ Before Starting: safety note
Aster is **research software**, and agents can be configured to execute LLM-generated code. We welcome contributions and fixes, but expect rough edges. LLM-generated code may take dangerous or unwanted actions, including sending private data to uncontrolled locations, deleting files, or modifying its environments.  Ensure you run Aster agents in a sandboxed environment isolated from your primary filesystem, such as [NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell).


Aster validates generated code (AST checks) and applies module deny-lists before execution. **These are defense-in-depth guardrails, not a containment boundary.** They exist to keep generated code from freezing the event loop and to catch common mistakes early — not to stop code that is actively trying to escape. A static checker over Python cannot provide that guarantee: `open()` gives arbitrary file access, `importlib` can load modules straight from a path, and reflection reaches the rest. **The containment boundary is OS-level isolation** — always run agents that execute generated code inside a sandbox such as a container, VM, or [NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell). Do not rely on the in-process validators alone.


### 1. Connect a model

Install the CLI and run the guided setup:

```bash
uv sync --extra cli
uv run aster connect
```

Choose your provider or endpoint, API key, exact model, and an alias such as
`my-model`. Connect helps your agents run smoothly: it suggests and saves a
reply/output token limit, records the context window when known (you can supply
it), and lists your chosen media formats. It also tests the configured route,
tools, reasoning, and caching. The checks send real requests and may incur
provider charges. The wizard asks before checks and saving. Keep keys in
environment variables or Aster's secrets file, not in model configuration or
committed code.

Use the saved alias in Python:

```python
from aster.unifiedllm import get_llm_client

llm = get_llm_client("my-model")
```

See [Connect setup](docs/model-connect.md)
for providers, model settings, editing, and diagnostic stages.

You can also construct clients directly using
[LiteLLM-supported model names](https://docs.litellm.ai/), with your endpoint
and credentials. See [Model configuration](docs/model-configuration.md).

### 2. Your first agent

***Agents are Python objects***. Methods with `...` bodies are **generation methods** — implemented at runtime by an LLM-driven strategy. The signature defines the contract; the docstring is the prompt.

```python
import asyncio

from aster import Agent


class FeedbackAgent(Agent, llm=llm):
    """You are an agent specializing in analyzing customer feedback."""

    async def analyze_feedback(self, text: str) -> str:
        """Analyze customer feedback for sentiment and key topics in one sentence."""
        ...


async def main():
    try:
        agent = FeedbackAgent()
        result = await agent.analyze_feedback("Great product, but shipping was slow")
        print(result)
    finally:
        await llm.aclose()


asyncio.run(main())
```

Save the model-selection and agent snippets together as `first_agent.py`, then
run `uv run python first_agent.py` in your project. After your final call,
`await llm.aclose()` releases client resources.

The [checked-in first example](examples/quickstart/01_first_generation_method.py)
is a separate source-checkout example: it chooses its own client from environment
credentials and does not automatically use your `my-model` alias.

Rename `analyze_feedback` to `analyze_feedback_briefly` and the output changes — your method name, parameters, and docstring *are* the prompt.

Prefer a guided notebook path? Start with the [**notebook tutorials**](notebook_tutorials/README.md), which walk through the same ideas in Colab-friendly steps, with more notebooks planned.

Ready to run something specific? Use the [**examples catalog**](examples/README.md) to find quickstarts for structured output, tools, strategies, tracing, context blocks, MCP, and more.

### 3. See what your agent is doing

Every LLM call, code execution, and method invocation is traced by default — orchestrators, generation methods, and helpers, with parent-child spans preserved. If you installed the CLI and viewer dependencies, start the trace viewer and open the run in your browser:

```bash
uv sync --extra cli --extra viewer    # includes viewer runtime dependencies
uv run aster start-dev        # trace viewer on http://localhost:5001
```

If the viewer isn't running, tracing is silently disabled — no configuration needed either way.

### 4. Give your agent media inputs

Pass `Image`, `Audio`, `Video`, or PDF `File` objects as method arguments.
CodeAct can also use `show()` to see media created while it runs.

Aster assumes models are text-only until configured. Set `media_inputs`, such
as `[image, pdf]`, in the model entry. Connect lets you choose formats manually
or test them with generated samples. If you enable a format the provider does
not accept, requests may fail.

```python
from aster import Agent, PredictStrategy, strategy
from aster.media import Image
from aster.unifiedllm import get_llm_client

# This alias must allow image input and use an endpoint that accepts it.
vision = get_llm_client("vision")

class PhotoReader(Agent, llm=vision):
    @strategy(PredictStrategy())
    async def describe(self, image: Image) -> str:
        """Describe the image briefly. Do not invent details."""
        ...

# In your async application, with permission to send the photo:
# result = await PhotoReader().describe(Image.from_file("photo.png"))
# await vision.aclose()
```

Load media with `from_file`, `from_bytes(..., media_type=...)`, or `from_url`.
URL support depends on the provider. `File` supports PDFs. Formats left out of
`media_inputs` are replaced with text notices when sending a request.

See the [media API guide](docs/media.md)
for a complete example, CodeAct `show()`, audio/video/PDF inputs, and limits.

## Learn more

- **[Documentation](docs/README.md)** — human-oriented reading paths, core concepts, architecture, and safety guidance.
- **[Framework tour](docs/tour.md)** — a concise conceptual showcase of Aster's core ideas and Python-first design.
- **[Decision models](docs/concepts/decisions.md)** — route classification, yes/no, and scoring methods to a calibrated decision model such as `typesafe/jev-1.13` with `DecideStrategy`; agents that only make decisions need no chat LLM. Runnable example: [`16_decisions.py`](examples/quickstart/16_decisions.py).
- **[Notebook tutorials](notebook_tutorials/README.md)** — the primary hands-on path for your first agent, strategy selection, CodeAct's live-object workflow, and composing subagents. More notebooks are planned.
- **[Examples catalog](examples/README.md)** — runnable quickstarts, advanced mechanics, and complete benchmark systems, indexed by capability and setup requirements.
- **[Paper](https://arxiv.org/abs/2607.20709)** — design principles, harness details, capability tests, and SWE-bench Verified / Terminal-Bench 2.0 results.
- **[Blog post](https://developer.nvidia.com/blog/six-agent-harness-capabilities-for-higher-model-performance/)** — Six Agent Harness Capabilities for Higher Model Performance.
- **[AGENTS.md](AGENTS.md)** — conventions used inside this repo (helpful when reading the source).

## Contributing

From the repository root, sync the development environment with `uv`. To run
Aster, use the [Aster quickstart](#run-aster-from-a-source-checkout).

```bash
uv sync --group dev
```

This installs the core framework, workspace packages, development tools, the `aster` CLI, and the trace viewer runtime in the repo's `.venv`. Run CLI commands through `uv`:

```bash
uv run aster --help
uv run aster start-dev       # trace viewer on http://localhost:5001
```

Enable pre-commit hooks and run the test/lint suite:

```bash
uv run pre-commit install
uv run pytest                # run tests
uv run ruff check            # lint
uv run pyright               # type check
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## License

Apache 2.0. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
