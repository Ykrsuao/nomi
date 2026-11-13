# Notebook Tutorials

Start here if you want to learn Aster by running code, inspecting prompts, and
seeing how small object-oriented agents come together step by step.

The notebooks cover the fundamentals: agents as Python objects, strategy
selection, CodeAct's live-object workflow, and composing subagents. Run them in
order with the checkout's Python environment (`uv sync --group dev`). Select that
environment as your notebook kernel. The setup cells validate the local Aster
installation; they do not install a package from an external index.

| # | Tutorial | Covers |
|---:|---|---|
| 1 | [Your First Object-Oriented Agent](./01_your_first_agent.ipynb) | Agents as Python objects, generation methods, methods as tools, typed returns, state, and the first trace-viewer hook. |
| 2 | [Choosing a Strategy](./02_choosing_a_strategy.ipynb) | How to choose between single-shot structured prediction and CodeAct's iterative Python execution. |
| 3 | [CodeAct REPL and Pass by Reference](./03_codeact_tools_and_live_objects.ipynb) | How CodeAct uses a persistent REPL, passes large live objects by reference, and points toward Recursive Language Models (RLMs) through small helper surfaces. |
| 4 | [Composing Subagents](./04_composing_subagents.ipynb) | How to compose subagents with plain Python — sequential, parallel with `asyncio.gather`, and LLM-driven spawning from generated CodeAct code. |

Looking for compact copy-paste examples? Use `examples/quickstart/`.
For deeper authoring reference material, see the `skills/aster-*` documents.
