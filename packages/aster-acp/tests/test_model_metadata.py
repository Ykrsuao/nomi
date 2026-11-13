# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Endpoint model limits reach live ACP sessions without an inference request."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from acp import text_block
from acp.schema import UsageUpdate
from aster_acp.runtime_controls import RuntimeControls
from aster_acp.server import CodingACPAdapter
from aster_cli.coding.agent import CodingAgent

from aster.llm_types import LLMUsage
from aster.unifiedllm import (
    CompletionClient,
    FakeLLMClient,
    LLMResponse,
    ResponsesClient,
    ToolCall,
    registry,
)

_MODEL = "openai/auto/deepseek-v4.1-flash"
_BASE = "https://metadata-gateway.example.invalid/v1"
_WINDOW = 1_048_576


@pytest.fixture(autouse=True)
def isolated_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASTER_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("ASTER_PROJECT_DIR", str(tmp_path / "project"))
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "true")
    monkeypatch.delenv("ASTER_SETTINGS", raising=False)
    monkeypatch.setattr(registry, "_loaded", True)
    monkeypatch.setattr(registry, "MODELS", {})
    monkeypatch.setattr("aster_acp.server.load_coding_skills_dirs", lambda root: [])


class _Client:
    """Only session updates are exercised by these metadata-only exchanges."""

    def __init__(self) -> None:
        self.updates: list[Any] = []

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        self.updates.append(update)


@pytest.mark.parametrize("client_class", [CompletionClient, ResponsesClient])
async def test_new_session_reads_exact_gateway_context_without_model_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    client_class: type[CompletionClient] | type[ResponsesClient],
) -> None:
    requests: list[httpx.Request] = []

    async def send(
        client: httpx.AsyncClient, request: httpx.Request, **kwargs: Any
    ) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == _BASE + "/models"
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "workbuddy/deepseek-v4.1-flash", "context_window": 1_000_000},
                    {
                        "id": "auto/deepseek-v4.1-flash",
                        "max_input_tokens": _WINDOW,
                        "context_window": _WINDOW,
                        "context_length": _WINDOW,
                        "max_output_tokens": 393_216,
                    },
                ]
            },
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    llm = client_class(model=_MODEL, api_base=_BASE, api_key="metadata-test-key")
    llm._registry_config = {}
    adapter = CodingACPAdapter(lambda: llm)
    adapter.on_connect(_Client())  # type: ignore[arg-type]
    try:
        created = await adapter.new_session(str(tmp_path))
        session = (await adapter._sessions.get(created.session_id)).value
        assert session.agent.llm.context_window == _WINDOW
        assert session.agent.llm.get_context_limits().context_window == _WINDOW
        assert len(requests) == 1
        assert "context_window" not in llm.config
    finally:
        await adapter.close()


def _listing(
    monkeypatch: pytest.MonkeyPatch, windows: list[int], *, status: int = 200
) -> list[httpx.Request]:
    requests: list[httpx.Request] = []

    async def send(
        client: httpx.AsyncClient, request: httpx.Request, **kwargs: Any
    ) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == _BASE + "/models"
        requests.append(request)
        window = windows[min(len(requests) - 1, len(windows) - 1)]
        return httpx.Response(
            status,
            json={"data": [{"id": "auto/deepseek-v4.1-flash", "context_window": window}]},
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    return requests


def _model() -> CompletionClient:
    llm = CompletionClient(model=_MODEL, api_base=_BASE, api_key="metadata-test-key")
    llm._registry_config = {}
    return llm


async def test_loaded_session_refreshes_endpoint_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = _listing(monkeypatch, [_WINDOW, _WINDOW // 2])
    creator = CodingACPAdapter(_model)
    creator.on_connect(_Client())  # type: ignore[arg-type]
    try:
        created = await creator.new_session(str(tmp_path))
    finally:
        await creator.close()

    loader = CodingACPAdapter(_model)
    loader.on_connect(_Client())  # type: ignore[arg-type]
    try:
        await loader.load_session(str(tmp_path), created.session_id)
        session = (await loader._sessions.get(created.session_id)).value
        assert session.agent.llm.context_window == _WINDOW // 2
        assert len(requests) == 2
    finally:
        await loader.close()


async def test_model_switch_adopts_metadata_before_applying_automatic_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = _listing(monkeypatch, [_WINDOW])
    old = FakeLLMClient()
    old._context_window = 8_192
    agent = CodingAgent(llm=old, cwd=tmp_path, skills_dirs=[])
    candidate = _model()
    published: list[Any] = []
    controls = RuntimeControls(
        agent,
        _Client(),  # type: ignore[arg-type]
        "metadata-switch",
        publish=published.append,
        model_factory=lambda name: candidate,
        model_names=lambda: ["new-model"],
    )
    try:
        await controls.configure(model="new-model")
        assert agent.llm is candidate
        assert candidate.context_window == _WINDOW
        assert len(requests) == 1
        automatic = [s for s in agent._summarizers if s._automatic_context_budget]
        assert automatic
        assert all(s.config.max_tokens > 500_000 for s in automatic)
        assert published[-1].field_meta["aster"]["runtime"]["modelName"] == _MODEL
    finally:
        await agent.close()


@pytest.mark.parametrize("metadata_status", [200, 503])
async def test_first_reply_reports_endpoint_context_and_keeps_missing_price_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metadata_status: int
) -> None:
    requests = _listing(monkeypatch, [_WINDOW], status=metadata_status)
    llm = _model()
    simulated = FakeLLMClient(
        [
            LLMResponse(
                finish_reason="tool_calls",
                usage=LLMUsage(input_tokens=262_144, output_tokens=10),
                parts=(
                    ToolCall(
                        id="metadata-return",
                        name="return_result",
                        arguments=json.dumps(
                            {"result": {"message": "Metadata verified", "explanation": "done"}}
                        ),
                    ),
                ),
            )
        ],
        strict_exhaustion=True,
    )
    monkeypatch.setattr(llm, "acall", simulated.acall)
    receiver = _Client()
    adapter = CodingACPAdapter(lambda: llm)
    adapter.on_connect(receiver)  # type: ignore[arg-type]
    try:
        created = await adapter.new_session(str(tmp_path))
        assert simulated.call_count == 0
        assert len(requests) == 1
        await adapter.prompt(created.session_id, [text_block("Check model metadata")])
        session = (await adapter._sessions.get(created.session_id)).value
        await session.bridge.flush()
        usage = [update for update in receiver.updates if isinstance(update, UsageUpdate)]
        assert len(usage) == 1
        assert usage[0].used == 262_144
        assert usage[0].size == (_WINDOW if metadata_status == 200 else 0)
        assert usage[0].cost is None
        assert simulated.call_count == 1
        assert len(requests) == 1
    finally:
        await adapter.close()
