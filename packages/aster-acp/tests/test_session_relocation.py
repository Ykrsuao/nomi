# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Moved workspaces resume through their current location without rewriting history."""

import json
import re
import shutil
from pathlib import Path
from unittest.mock import create_autospec

import pytest
from acp.interfaces import Client
from aster_acp.server import CodingACPAdapter
from aster_cli.sessions import SessionStore

from aster import Context
from aster.skill_registry import SkillRegistry
from aster.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


@pytest.mark.parametrize("source_retained", [False, True])
@pytest.mark.parametrize("has_saved_environment", [False, True])
async def test_listed_session_resumes_current_workspace_after_relocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_retained: bool,
    has_saved_environment: bool,
) -> None:
    monkeypatch.setenv("ASTER_USER_DIR", str(tmp_path / "isolated-config"))
    monkeypatch.delenv("ASTER_SETTINGS", raising=False)
    monkeypatch.setattr(SkillRegistry, "discovered", lambda self: [])
    monkeypatch.setattr("aster_acp.server.load_coding_skills_dirs", lambda root: [])
    source = tmp_path / "original"
    source.mkdir()
    destination = tmp_path / "relocated-{1+1}"
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    with SessionStore(unrelated / ".aster/sessions").create(session_id="other"):
        pass

    adapter = CodingACPAdapter(
        lambda: FakeLLMClient(
            [
                LLMResponse(
                    finish_reason="tool_calls",
                    parts=(
                        ToolCall(
                            id="direct-reply",
                            name="return_result",
                            arguments=json.dumps(
                                {"result": {"message": "Ready", "explanation": "Replied"}}
                            ),
                        ),
                    ),
                )
            ],
            strict_exhaustion=True,
        )
    )
    adapter.on_connect(create_autospec(Client, instance=True, spec_set=True))
    try:
        created = await adapter.new_session(str(source))
        session = (await adapter._sessions.get(created.session_id)).value
        current_shell_instructions = session.agent.context["host_shell"]
        session.agent.context["user_note"] = "Keep this saved context."
        if has_saved_environment:
            # Existing snapshots contain static paths and may describe another host.
            session.agent.context["workspace_paths"] = Context(
                json.dumps({"workspace_root": str(source.resolve())}), prefix=True
            )
            session.agent.context["host_shell"] = Context("Outdated host shell instructions.")
        else:
            del session.agent.context["workspace_paths"]
            del session.agent.context["host_shell"]
        session.handle.record_user_message("Retain the original message")
        session.handle.checkpoint(session.agent)
        session.handle.storage.save_current_snapshot(session.agent)
        await adapter.close_session(created.session_id)

        if source_retained:
            shutil.copytree(source, destination)
        else:
            assert source.resolve().parent == destination.parent.resolve() == tmp_path.resolve()
            source.rename(destination)
        store = SessionStore(destination / ".aster/sessions")
        database = store.path_for(created.session_id)
        before = database.read_bytes()
        modified_at = database.stat().st_mtime_ns

        page = await adapter.list_sessions(str(destination))

        assert len(page.sessions) == 1
        listed = page.sessions[0]
        assert listed.session_id == created.session_id
        assert listed.cwd == str(destination.resolve())
        assert database.read_bytes() == before
        assert database.stat().st_mtime_ns == modified_at
        assert store.get(created.session_id).working_directory == str(source)

        await adapter.load_session(cwd=listed.cwd, session_id=listed.session_id)
        restored = (await adapter._sessions.get(created.session_id)).value
        assert restored.agent.cwd == destination.resolve()
        assert restored.agent.shell.cwd == destination.resolve()
        assert restored.handle.turns()[0].content == "Retain the original message"
        assert restored.handle.storage.checkpoint_ids() == ["user:1"]
        assert restored.agent.context["user_note"] == "Keep this saved context."

        await restored.agent.handle({"user_messages": ["hi"]})
        prompt_parts = []
        for message in restored.agent.llm.last_messages:
            content = message.get("content", "")
            if isinstance(content, str):
                prompt_parts.append(content)
            else:
                prompt_parts.extend(part["text"] for part in content if part.get("type") == "text")
        prompt = "\n".join(prompt_parts)
        rendered_paths = re.search(
            r"<workspace_paths(?: [^>]*)?>(.*?)</workspace_paths>", prompt, re.S
        )
        assert rendered_paths is not None
        expected_paths = {"workspace_root": str(destination.resolve())}
        assert Path(json.loads(rendered_paths.group(1))["workspace_root"]) == destination.resolve()
        assert json.loads(restored.agent.context["workspace_paths"]) == expected_paths
        assert "relocated-{1+1}" in prompt
        assert current_shell_instructions in prompt
        assert restored.agent.context["host_shell"] == current_shell_instructions
    finally:
        await adapter.close()
