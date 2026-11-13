# SPDX-License-Identifier: Apache-2.0
"""New evaluation outputs and archived results stay out of trace operations."""

from datetime import datetime

import pytest
from aster_cli.commands import traces
from click.testing import CliRunner

from aster.trace_explorer import TraceExplorer
from eval_pipeline.eval_parser import EvalFileParser
from eval_pipeline.experiment_writer import ExperimentWriter


def test_writer_uses_aster_filename_and_roundtrips(tmp_path):
    writer = ExperimentWriter(tmp_path, "sample", datetime(2026, 11, 12, 12, 0))
    path = writer.start(models=["synthetic"])
    writer.finalize()

    assert path.name == "sample_20261112_120000.aster-eval.jsonl"
    metadata, results, completion, _annotations = EvalFileParser().parse_file(path)
    assert metadata is not None
    assert metadata.metadata.suite_name == "sample"
    assert results == []
    assert completion is not None
    assert completion.status == "completed"


def test_trace_cleanup_excludes_current_and_archived_eval_files(tmp_path):
    eval_files = []
    for suffix in (".aster-eval.jsonl", ".noo-eval.jsonl", ".006eval.json"):
        path = tmp_path / f"result{suffix}"
        path.write_text("{}\n", encoding="utf-8")
        eval_files.append(path)
    trace = tmp_path / "run.jsonl"
    trace.write_text("{}\n", encoding="utf-8")
    (tmp_path / "run.annotations.jsonl").write_text("{}\n", encoding="utf-8")

    assert traces._find_trace_files(tmp_path) == [trace]
    assert traces._find_eval_files(tmp_path) == sorted(eval_files)

    result = CliRunner().invoke(traces.command, ["delete", str(tmp_path), "--all", "--dry-run"])
    assert result.exit_code == 0
    assert "Found 1 trace files" in result.output
    assert all(path.name not in result.output for path in eval_files)
    assert all(path.exists() for path in [trace, *eval_files])


@pytest.mark.parametrize("suffix", [".aster-eval.jsonl", ".noo-eval.jsonl", ".006eval.json"])
async def test_trace_explorer_rejects_evaluation_artifacts(tmp_path, suffix):
    path = tmp_path / f"result{suffix}"
    with pytest.raises(ValueError, match="Cannot load eval file as trace"):
        await TraceExplorer.from_file(path)
