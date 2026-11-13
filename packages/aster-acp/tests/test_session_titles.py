# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Saved conversation pickers retain a meaningful title across ACP sessions."""

import json

import pytest
from acp import text_block
from acp.schema import SessionInfoUpdate
from aster_acp.server import CodingACPAdapter
from aster_cli.sessions import SessionStore

from aster.llm_types import LLMResponse, ToolCall
from aster.unifiedllm import FakeLLMClient


@pytest.mark.parametrize("existing_title", [None, "My custom title"])
async def test_first_prompt_titles_saved_session_without_overwriting_a_title(
    tmp_path, monkeypatch, existing_title
):
    monkeypatch.setenv("ASTER_USER_DIR", str(tmp_path / "user-config"))
    updates = []

    class Client:
        async def session_update(self, session_id, update, **kwargs):
            updates.append(update)

    def factory():
        return FakeLLMClient(
            scripted_responses=[
                LLMResponse(
                    raw_response=None,
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=f"call_{index}",
                            name="execute_python",
                            arguments=json.dumps(
                                {"code": "return_result(Done(explanation='done'))"}
                            ),
                        )
                    ],
                    finish_reason="tool_calls",
                    reasoning=None,
                )
                for index in range(2)
            ],
            strict_exhaustion=True,
        )

    adapter = CodingACPAdapter(factory)
    adapter.on_connect(Client())
    try:
        created = await adapter.new_session(str(tmp_path))
        if existing_title:
            runtime = await adapter._get_runtime(created.session_id)
            runtime.value.handle.set_title(existing_title, user_set=True)
        await adapter.prompt(created.session_id, [text_block("检查中文路径\n  和会话恢复")])
        await adapter.prompt(created.session_id, [text_block("A later request")])
        await adapter.close_session(created.session_id)
        page = await adapter.list_sessions(cwd=str(tmp_path))
        expected = existing_title or "检查中文路径 和会话恢复"
        assert page.sessions[0].title == expected
        assert SessionStore(tmp_path / ".aster/sessions").get(created.session_id).title == expected
        titles = [update.title for update in updates if isinstance(update, SessionInfoUpdate)]
        assert titles == ([] if existing_title else [expected])
    finally:
        await adapter.close()
