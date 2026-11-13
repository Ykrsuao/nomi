# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Input recall can be bootstrapped without resuming or locking old conversations."""

from aster_acp.server import CodingACPAdapter
from aster_cli.sessions import SessionStore

from aster.interactive import AgentMessage
from aster.unifiedllm import FakeLLMClient


async def test_list_opt_in_recovers_only_user_inputs_without_opening_agent(tmp_path):
    store = SessionStore(tmp_path / ".aster" / "sessions")
    first = store.create(working_directory=str(tmp_path))
    first.record_user_message("first input")
    first.events.add(AgentMessage(content="not an input"))
    second = store.create(working_directory=str(tmp_path))
    second.record_user_message("  中文\nsecond input  \n")
    first.record_user_message("latest input")

    # Both writer handles remain open while the picker reads their history.
    def factory():
        raise AssertionError("Reading input history must not create an agent")

    adapter = CodingACPAdapter(factory)
    try:
        ordinary = await adapter.list_sessions(str(tmp_path))
        assert ordinary.field_meta is None
        page = await adapter.list_sessions(str(tmp_path), aster={"includeInputHistory": True})
        assert (page.field_meta or {}).get("aster", {}).get("inputHistory") == [
            "first input",
            "  中文\nsecond input  \n",
            "latest input",
        ]
        assert adapter._client is None
    finally:
        await adapter.close()
        first.close()
        second.close()


async def test_empty_workspace_history_does_not_create_a_session_directory(tmp_path):
    adapter = CodingACPAdapter(FakeLLMClient)
    try:
        page = await adapter.list_sessions(str(tmp_path), aster={"includeInputHistory": True})
        assert (page.field_meta or {}).get("aster", {}).get("inputHistory") == []
        assert not (tmp_path / ".aster").exists()
    finally:
        await adapter.close()


def test_history_bootstrap_keeps_latest_inputs_with_count_and_encoded_byte_bounds(tmp_path):
    import json

    from aster_acp._input_history import recent_inputs

    store = SessionStore(tmp_path / ".aster" / "sessions")
    with store.create() as session:
        for text in ["old input", "x" * 100, "中文", "latest"]:
            session.record_user_message(text)
        assert recent_inputs(store, limit=2) == ["中文", "latest"]
        budget = sum(
            len(json.dumps(text, ensure_ascii=False).encode()) + 1 for text in ["中文", "latest"]
        )
        assert recent_inputs(store, max_bytes=budget) == ["中文", "latest"]
