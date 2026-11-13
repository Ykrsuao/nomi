# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline regressions for Aster output and read-only historical result support."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
_LEGACY_TEAM = "nemo"


@pytest.fixture(autouse=True)
def example_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))


def _events(run: Path, team: str, *, role: str = "team_leader") -> Path:
    path = run / "agent_logs" / team / role / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"event": "agent_turn", "step": 0, "turn": 1, "actions": ["UP"], "rationale": "move"},
        {
            "event": "env_step",
            "step": 1,
            "unix_time_s": 2,
            "levels_completed": 0,
            "diff_pixels": 1,
            "action_name": "UP",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _span() -> SimpleNamespace:
    return SimpleNamespace(
        name="aresponses",
        start_time=3_000_000_000,
        end_time=4_000_000_000,
        attributes={
            "llm.input_messages.0.message.role": "user",
            "llm.input_messages.0.message.content": "choose an action",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": "move up",
            "llm.token_count.prompt": 5,
            "llm.token_count.completion": 2,
        },
    )


def _load_script(name: str, relative: str, monkeypatch: pytest.MonkeyPatch):
    spec = importlib.util.spec_from_file_location(name, EXAMPLE_DIR / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("team", [None, "aster", _LEGACY_TEAM])
def test_recorder_writes_only_aster_layout_and_metadata(tmp_path: Path, team: str | None) -> None:
    from recorder import RunRecorder

    kwargs = {} if team is None else {"team": team}
    recorder = RunRecorder(tmp_path, game_id="ls20", variant="memory", **kwargs)
    recorder.solver_start(
        max_steps=5,
        available_actions=["UP"],
        config={"solver": f"{_LEGACY_TEAM}_single_agent"},
    )
    recorder.result(
        levels_completed=0,
        total_steps=1,
        termination_reason="done",
        extras={"solver": f"{_LEGACY_TEAM}_single_agent"},
    )

    event = json.loads(recorder.events_path.read_text())
    result = json.loads((tmp_path / "result.json").read_text())
    assert recorder.events_path == tmp_path / "agent_logs/aster/team_leader/events.jsonl"
    assert (tmp_path / "team_aster/shared").is_dir()
    assert event["config"]["solver"] == result["solver"] == "aster_single_agent"
    assert not (tmp_path / f"team_{_LEGACY_TEAM}").exists()
    assert not (tmp_path / "agent_logs" / _LEGACY_TEAM).exists()


def test_readers_prefer_each_current_artifact_without_writing(tmp_path: Path) -> None:
    from _run_layout import read_log, read_workspace

    legacy_events = _events(tmp_path, _LEGACY_TEAM)
    legacy_store = tmp_path / f"team_{_LEGACY_TEAM}/shared/memory.sqlite"
    legacy_store.parent.mkdir(parents=True)
    legacy_store.write_bytes(b"historical")

    assert read_log(tmp_path, "events.jsonl") == legacy_events
    assert read_workspace(tmp_path, "memory.sqlite") == legacy_store
    assert not (tmp_path / "team_aster").exists()
    assert not (tmp_path / "agent_logs/aster").exists()

    # Regenerated messages do not hide the older event stream or workspace.
    current_messages = tmp_path / "agent_logs/aster/team_leader/messages"
    current_messages.mkdir(parents=True)
    (tmp_path / "team_aster/shared").mkdir(parents=True)
    assert read_log(tmp_path, "messages") == current_messages
    assert read_log(tmp_path, "events.jsonl") == legacy_events
    assert read_workspace(tmp_path, "memory.sqlite") == legacy_store

    current_events = _events(tmp_path, "aster")
    current_store = tmp_path / "team_aster/shared/memory.sqlite"
    current_store.write_bytes(b"current")
    assert read_log(tmp_path, "events.jsonl") == current_events
    assert read_workspace(tmp_path, "memory.sqlite") == current_store
    assert legacy_store.read_bytes() == b"historical"


def test_missing_read_paths_use_aster_without_creating_files(tmp_path: Path) -> None:
    from _run_layout import read_log, read_workspace

    assert (
        read_log(tmp_path, "events.jsonl") == tmp_path / "agent_logs/aster/team_leader/events.jsonl"
    )
    assert read_workspace(tmp_path) == tmp_path / "team_aster/shared"
    assert list(tmp_path.iterdir()) == []


def test_fleet_reader_does_not_double_count_mixed_layouts(tmp_path: Path) -> None:
    from _run_layout import fleet_event_paths

    old_run = tmp_path / "first/memory/20260101"
    current_run = tmp_path / "second/memory/20260102"
    old = _events(old_run, _LEGACY_TEAM)
    _events(current_run, _LEGACY_TEAM)
    current = _events(current_run, "aster")
    _events(tmp_path / "third/mdfiles/20260103", "aster")
    assert fleet_event_paths(tmp_path) == [old, current]


@pytest.mark.parametrize("group_filter", [None, ["aster_solver"], [f"{_LEGACY_TEAM}_solver"]])
def test_comparison_reads_history_and_normalizes_generated_metadata(
    tmp_path: Path, group_filter: list[str] | None
) -> None:
    from compare import load_runs

    original = {}
    for team in (_LEGACY_TEAM, "aster", "unrelated"):
        path = tmp_path / f"{team}_solver" / f"run_{team}" / "result.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"solver": f"{team}_single_agent", "game_id": "ls20"}))
        original[path] = path.read_bytes()
    runs = load_runs(tmp_path, group_filter)
    assert len(runs) == 2
    assert all(run["solver"] == "aster_single_agent" for run in runs)
    assert all(run["_group"] == "aster_solver" for run in runs)
    assert all(path.read_bytes() == content for path, content in original.items())


@pytest.mark.parametrize("team", [_LEGACY_TEAM, "aster"])
def test_backfill_reads_both_layouts_and_writes_current_messages(tmp_path: Path, team: str) -> None:
    from backfill_messages import backfill

    events = _events(tmp_path, team)
    original = events.read_bytes()
    assert backfill(tmp_path) == 1
    assert backfill(tmp_path) == 1  # idempotent
    target = tmp_path / "agent_logs/aster/team_leader/messages"
    assert "move" in (target / "step_001_round_00_assistant.md").read_text()
    assert len(list(target.glob("*.md"))) == 2
    assert events.read_bytes() == original
    assert not (tmp_path / "agent_logs" / _LEGACY_TEAM / "team_leader/messages").exists()


@pytest.mark.parametrize("team", [_LEGACY_TEAM, "aster"])
def test_trace_dump_reads_history_and_writes_current_messages(tmp_path: Path, team: str) -> None:
    from dump_llm_messages import write_viewer_messages

    events = _events(tmp_path, team)
    original = events.read_bytes()
    span = _span()
    trace = {
        "name": span.name,
        "startTimeUnixNano": str(span.start_time),
        "attributes": [
            {"key": key, "value": {"stringValue": value}} for key, value in span.attributes.items()
        ],
    }
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "test.jsonl").write_text(
        json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": [trace]}]}]}) + "\n"
    )
    assert write_viewer_messages(tmp_path) == 2
    assert write_viewer_messages(tmp_path) == 2
    target = tmp_path / "agent_logs/aster/team_leader/messages"
    assert "move up" in (target / "step_001_round_00_assistant.md").read_text()
    assert events.read_bytes() == original
    assert not (tmp_path / "agent_logs" / _LEGACY_TEAM / "team_leader/messages").exists()


@pytest.mark.parametrize("team", [None, _LEGACY_TEAM])
def test_exporters_use_aster_paths_and_keep_step_mapping(tmp_path: Path, team: str | None) -> None:
    from opentelemetry.sdk.trace.export import SpanExportResult
    from viewer_event_exporter import ViewerEventExporter
    from viewer_trace_exporter import ViewerMessageExporter

    events = _events(tmp_path, "aster")
    kwargs = {} if team is None else {"team": team}
    message_exporter = ViewerMessageExporter(tmp_path, **kwargs)
    event_exporter = ViewerEventExporter(tmp_path, **kwargs)
    assert message_exporter.export([_span()]) == SpanExportResult.SUCCESS
    assert event_exporter.export([_span()]) == SpanExportResult.SUCCESS
    assert event_exporter.events_path == events
    llm_event = json.loads(events.read_text().splitlines()[-1])
    assert llm_event["event"] == "llm_call"
    assert llm_event["step"] == 1
    assert (tmp_path / "agent_logs/aster/team_leader/messages/step_001_round_00_user.md").exists()
    assert not (tmp_path / "agent_logs" / _LEGACY_TEAM).exists()


def test_message_exporter_maps_historical_events_without_changing_them(tmp_path: Path) -> None:
    from viewer_trace_exporter import ViewerMessageExporter

    events = _events(tmp_path, _LEGACY_TEAM)
    original = events.read_bytes()
    exporter = ViewerMessageExporter(tmp_path)
    exporter.export([_span()])
    assert (tmp_path / "agent_logs/aster/team_leader/messages/step_001_round_00_user.md").exists()
    assert events.read_bytes() == original


@pytest.mark.parametrize("team", [_LEGACY_TEAM, "aster"])
def test_viewer_reads_both_layouts_and_prefers_current_role(tmp_path: Path, team: str) -> None:
    pytest.importorskip("rich", reason="viewer requires the optional arc extra")
    from viewer import GameTracker

    events = _events(tmp_path, team, role="observer")
    tracker = GameTracker("ls20", tmp_path, "observer")
    tracker.poll()
    assert tracker.path == events
    assert tracker.stats.step == 1
    current = _events(tmp_path, "aster", role="observer")
    new_tracker = GameTracker("ls20", tmp_path, "observer")
    assert new_tracker._resolve_path() == current


@pytest.mark.parametrize("team", [_LEGACY_TEAM, "aster"])
@pytest.mark.parametrize("nested", [False, True])
def test_memory_analysis_finds_both_store_layouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, team: str, nested: bool
) -> None:
    module = _load_script("_arc_memory_analysis", "analysis/memory/analyze_memory.py", monkeypatch)
    run = tmp_path / "memory/20260101" if nested else tmp_path
    store = run / f"team_{team}/shared/memory.sqlite"
    store.parent.mkdir(parents=True)
    store.write_bytes(b"fixture")
    assert module._find_store(tmp_path) == store


@pytest.mark.parametrize("team", [_LEGACY_TEAM, "aster"])
def test_knowledge_analysis_reads_historical_helpers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], team: str
) -> None:
    from analyze_knowledge import analyze_run

    helper = tmp_path / f"team_{team}/shared/helpers/model.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("def predict(): return 1\n")
    analyze_run(tmp_path)
    assert "model.py" in capsys.readouterr().out


@pytest.mark.parametrize("group", [None, f"{_LEGACY_TEAM}_solver", "custom"])
def test_runner_normalizes_only_the_old_default_group(
    monkeypatch: pytest.MonkeyPatch, group: str | None
) -> None:
    from run_solver import parse_args

    argv = ["run_solver.py", "--game", "ls20", "--variant", "memory"]
    if group is not None:
        argv += ["--group", group]
    monkeypatch.setattr(sys, "argv", argv)
    assert parse_args().group == ("custom" if group == "custom" else "aster_solver")


@pytest.mark.parametrize("team", [_LEGACY_TEAM, "aster"])
def test_audit_readers_and_signals_preserve_historical_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, team: str
) -> None:
    monkeypatch.setenv("RT_RUN_ROOT", str(tmp_path))
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR / "analysis/red_team"))
    rt = _load_script("rt_common", "analysis/red_team/rt_common.py", monkeypatch)
    first = _load_script(
        "_arc_first_violation", "analysis/red_team/first_violation.py", monkeypatch
    )
    digest = _load_script("_arc_digest_escape", "analysis/red_team/digest_escape.py", monkeypatch)
    messages = tmp_path / "agent_logs" / team / "team_leader/messages"
    messages.mkdir(parents=True)
    assert rt.messages_dir(tmp_path) == messages
    assert rt.BENIGN.search(f"team_{team}/shared/helpers/model.py")
    assert rt.RULE3_FOREIGN["results_dir"].search(f"{team}_solver/20260101")
    assert first.OFFLIMITS.search(f"{team}_solver/20260101")
    assert digest.outcome("read", f"{team}_solver/20260101") == "DATA_RETURNED"
