# Coding-agent skills for authoring Aster

Portable `SKILL.md` bundles for coding agents (Claude Code, Cursor, Codex, or any Agent Skills host) that help developers **author application agents with** Aster, capture traces, and debug runs with the trace viewer and trace explorer.

These are instructions *for coding agents about the framework* — not `aster.Skill` runtime skills (though the file format is compatible with `TextSkill`, see Validate below).

## Skills

| Skill | Use for |
|---|---|
| [`aster-agent-authoring`](aster-agent-authoring/SKILL.md) | Core authoring: Agent subclasses, generation methods (`...`), docstring prompts, structured output, strategies (CodeAct/Predict), visibility, orchestrators, subagents, prompt debugging |
| [`aster-connect`](aster-connect/SKILL.md) | Complete headless Connect workflow for agents: permission, discovery, model/media checks, JSON reports and saving aliases |
| [`aster-model-configuration`](aster-model-configuration/SKILL.md) | Registry semantics, manual configuration, reply budgets, reasoning levels, caching and reasoning replay; refers to aster-connect for headless operation |
| [`aster-media`](aster-media/SKILL.md) | Typed image/audio/video/PDF inputs, CodeAct show(), route-local media permissions, MIME/URL/size limits, and text-only replay |
| [`aster-codeact-advanced`](aster-codeact-advanced/SKILL.md) | Advanced strategy tuning: prefill (custom/disable/pre-ellipsis), loop guards, truncation tuning, code restrictions, execution internals, PredictConfig |
| [`aster-agentdoc`](aster-agentdoc/SKILL.md) | Making types render beautiful docs for the LLM: `doc()`, `spec()`, `hidden`, `Annotated` descriptions, `pformat`/`pprint` tuning |
| [`aster-context-and-state`](aster-context-and-state/SKILL.md) | Context blocks, event history and `EventQuery`, history summarization, persistence and memory |
| [`aster-tools-and-skills`](aster-tools-and-skills/SKILL.md) | Methods as tools, built-in tools (`ShellTools`/`TodoManager`), MCP integration, agent skills (`Skill`/`TextSkill`), multimodal media |
| [`aster-channels`](aster-channels/SKILL.md) | Reactive input: Channel/QueueManager, race() dispatch loops, spawn() background jobs, monitor/cron/tail producers |
| [`aster-self-extending`](aster-self-extending/SKILL.md) | Agent-authored code: persistent skill libraries (self.libs), in-cell helpers and standalone @strategy sub-calls, @slash_command |
| [`aster-middleware-hooks`](aster-middleware-hooks/SKILL.md) | Intercepting execution: middleware (`intercept()` guardrails/transforms/blocking), event observers (`on()`), InstrumentationHooks protocol |
| [`aster-capturing-traces`](aster-capturing-traces/SKILL.md) | Capturing traces: auto-tracing, `enable_tracing` + exporters (jsonl/otlp/langfuse/journal), `@no_trace`, span model, env vars |
| [`aster-trace-viewer`](aster-trace-viewer/SKILL.md) | Running and using the trace viewer (`aster start-dev`): UI, import/export, REST API |
| [`aster-trace-explorer`](aster-trace-explorer/SKILL.md) | Programmatic trace analysis: `trace-explorer` CLI, `TraceExplorer` library, thin client, experiment-level debugging |

All content was verified against `src/aster` at the time of writing; where the skills contradict older docs (e.g. `SkillManager`, `from agentdoc import ...`, `enable_tracing(trace_dir=...)`, "private methods aren't traced"), the skills reflect the code.

## Install

Copy or symlink skill directories into the location your coding agent reads:

```bash
# Claude Code — all skills, user-global
mkdir -p ~/.claude/skills
cp -R skills/aster-* ~/.claude/skills/

# Or the core authoring/setup/media skills into a project
mkdir -p .claude/skills
cp -R skills/aster-agent-authoring skills/aster-model-configuration skills/aster-connect skills/aster-media .claude/skills/
```

When installing an authoring, model-configuration or tools skill, also install
`aster-connect` so its headless workflow reference is available.

Other hosts read from different directories (e.g. `.codex/skills/`, `.agents/skills/`).

Sibling-skill links require those siblings to be installed together. Links to
`../../docs/` resolve in the source checkout, not in a copied skill directory.
Keep the matching source checkout or obtain the docs for your installed release
tag/commit from this repository; do not assume development `main` describes an
older wheel. Give the coding agent that checkout's absolute path when needed.

## Validate

The bundles are loadable as framework `TextSkill`s, which validates their frontmatter:

```bash
uv run python - <<'PY'
from pathlib import Path
from aster import TextSkill
for path in sorted(Path("skills").glob("*/SKILL.md")):
    s = TextSkill(path=path.parent)
    print(f"{s.id}: {s.description[:88]}")
PY
```
