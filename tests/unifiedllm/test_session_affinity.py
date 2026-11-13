# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""prompt_cache_key doubles as the provider-side session-affinity hint."""

from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import Choices, Message, ModelResponse

from aster.unifiedllm import CompletionClient, ResponsesClient, RetryConfig
from aster.unifiedllm.cache_policy import SESSION_AFFINITY_HEADER, add_session_affinity_header


def test_header_mirrors_prompt_cache_key():
    """The key becomes the header when no headers were given."""
    params = {"prompt_cache_key": "agent-1-CodeActV2"}
    add_session_affinity_header(params)
    assert params["extra_headers"] == {SESSION_AFFINITY_HEADER: "agent-1-CodeActV2"}


@pytest.mark.parametrize("params", [{}, {"prompt_cache_key": ""}, {"prompt_cache_key": None}])
def test_no_key_adds_nothing(params):
    """Without a usable key the params are left untouched."""
    before = dict(params)
    add_session_affinity_header(params)
    assert params == before


def test_existing_headers_are_kept_and_an_explicit_value_wins():
    """Other headers survive and a caller's own affinity value is kept."""
    params = {
        "prompt_cache_key": "k",
        "extra_headers": {"x-other": "1", SESSION_AFFINITY_HEADER: "pinned-elsewhere"},
    }
    add_session_affinity_header(params)
    assert params["extra_headers"] == {"x-other": "1", SESSION_AFFINITY_HEADER: "pinned-elsewhere"}


def test_caller_header_wins_regardless_of_case():
    """HTTP header names are case-insensitive, so X-Session-Affinity also counts as present."""
    params = {"prompt_cache_key": "k", "extra_headers": {"X-Session-Affinity": "pinned"}}
    add_session_affinity_header(params)
    assert params["extra_headers"] == {"X-Session-Affinity": "pinned"}


def test_non_mapping_extra_headers_is_rejected():
    """A non-mapping extra_headers is an error, not silently replaced."""
    with pytest.raises(ValueError, match="extra_headers"):
        add_session_affinity_header({"prompt_cache_key": "k", "extra_headers": "nope"})


def _chat_response() -> ModelResponse:
    """Minimal successful litellm reply."""
    return ModelResponse(
        model="m",
        choices=[Choices(finish_reason="stop", message=Message(content="ok", role="assistant"))],
    )


def _responses_response() -> ResponsesAPIResponse:
    """Minimal successful litellm reply."""
    return ResponsesAPIResponse(
        id="resp_1",
        created_at=0,
        model="m",
        object="response",
        status="completed",
        output=[
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "ok", "annotations": []}],
            }
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("key", [None, "session-7-CodeActV2"])
async def test_completion_client_sends_the_header_only_with_a_key(monkeypatch, key):
    """CompletionClient forwards the header iff prompt_cache_key is set."""
    sent = {}

    async def fake(**kwargs):
        sent.update(kwargs)
        return _chat_response()

    monkeypatch.setattr("litellm.acompletion", fake)
    async with CompletionClient("openai/nvidia/moonshotai/kimi-k3", api_key="test") as client:
        extra = {"prompt_cache_key": key} if key else {}
        await client.acall([{"role": "user", "content": "hi"}], **extra)
    if key:
        assert sent["extra_headers"][SESSION_AFFINITY_HEADER] == key
        assert sent["prompt_cache_key"] == key
    else:
        assert "extra_headers" not in sent


@pytest.mark.asyncio
async def test_responses_client_sends_the_header(monkeypatch):
    """ResponsesClient merges the header into caller-provided extra_headers."""
    sent = {}

    async def fake(**kwargs):
        sent.update(kwargs)
        return _responses_response()

    monkeypatch.setattr("litellm.aresponses", fake)
    async with ResponsesClient("openai/nvidia/moonshotai/kimi-k3", api_key="test") as client:
        await client.acall(
            [{"role": "user", "content": "hi"}],
            prompt_cache_key="s-9",
            extra_headers={"x-other": "1"},
        )
    assert sent["extra_headers"] == {"x-other": "1", SESSION_AFFINITY_HEADER: "s-9"}
    assert sent["prompt_cache_key"] == "s-9"


@pytest.mark.parametrize("key", [None, "session-7-CodeActV2"])
def test_completion_client_sync_call_sends_the_header_only_with_a_key(monkeypatch, key):
    """The synchronous CompletionClient.call path forwards the header too."""
    sent = {}

    def fake(**kwargs):
        sent.update(kwargs)
        return _chat_response()

    monkeypatch.setattr("litellm.completion", fake)
    client = CompletionClient("openai/nvidia/moonshotai/kimi-k3", api_key="test")
    try:
        extra = {"prompt_cache_key": key} if key else {}
        client.call([{"role": "user", "content": "hi"}], **extra)
    finally:
        client.close()
    if key:
        assert sent["extra_headers"][SESSION_AFFINITY_HEADER] == key
        assert sent["prompt_cache_key"] == key
    else:
        assert "extra_headers" not in sent


def test_responses_client_sync_call_sends_the_header(monkeypatch):
    """The synchronous ResponsesClient.call path merges the header into extra_headers."""
    sent = {}

    def fake(**kwargs):
        sent.update(kwargs)
        return _responses_response()

    monkeypatch.setattr("litellm.responses", fake)
    client = ResponsesClient("openai/nvidia/moonshotai/kimi-k3", api_key="test")
    try:
        client.call(
            [{"role": "user", "content": "hi"}],
            prompt_cache_key="s-9",
            extra_headers={"x-other": "1"},
        )
    finally:
        client.close()
    assert sent["extra_headers"] == {"x-other": "1", SESSION_AFFINITY_HEADER: "s-9"}
    assert sent["prompt_cache_key"] == "s-9"


@pytest.mark.parametrize(
    ("configured_model", "overrides", "body_key"),
    [
        ("openai/model", {"model": "anthropic/claude-sonnet-4-5"}, False),
        ("anthropic/claude-sonnet-4-5", {"model": "openai/anthropic/gateway-model"}, True),
        ("openai/model", {"custom_llm_provider": "anthropic"}, False),
        ("test-model", {}, True),
    ],
)
@pytest.mark.parametrize("is_async", [False, True])
async def test_completion_affinity_follows_the_call_route(
    monkeypatch, configured_model, overrides, body_key, is_async
):
    sent = {}

    async def fake_async(**kwargs):
        sent.update(kwargs)
        return _chat_response()

    def fake_sync(**kwargs):
        sent.update(kwargs)
        return _chat_response()

    monkeypatch.setattr("litellm.acompletion", fake_async)
    monkeypatch.setattr("litellm.completion", fake_sync)
    client = CompletionClient(configured_model, api_key="test", cache_breakpoint=None)
    try:
        if is_async:
            await client.acall(
                [{"role": "user", "content": "hi"}],
                prompt_cache_key="conversation-key",
                **overrides,
            )
        else:
            client.call(
                [{"role": "user", "content": "hi"}],
                prompt_cache_key="conversation-key",
                **overrides,
            )
    finally:
        await client.aclose()
    assert sent["extra_headers"][SESSION_AFFINITY_HEADER] == "conversation-key"
    assert ("prompt_cache_key" in sent) is body_key
    if body_key:
        assert sent["prompt_cache_key"] == "conversation-key"


@pytest.mark.parametrize("is_async", [False, True])
async def test_unknown_completion_provider_is_rejected_by_dispatch(monkeypatch, is_async):
    from aster.unifiedllm._litellm import litellm

    operation = "acompletion" if is_async else "completion"
    request = (AsyncMock if is_async else Mock)(wraps=getattr(litellm, operation))
    monkeypatch.setattr(litellm, operation, request)
    network = Mock(side_effect=AssertionError("Unknown providers must fail before HTTP dispatch"))
    monkeypatch.setattr(httpx.Client, "send", network)
    monkeypatch.setattr(httpx.AsyncClient, "send", network)
    client = CompletionClient(
        "test-model",
        api_key="test",
        retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
    )
    try:
        with pytest.raises(litellm.BadRequestError, match="LLM Provider NOT provided"):
            if is_async:
                await client.acall(
                    [{"role": "user", "content": "hi"}], prompt_cache_key="conversation-key"
                )
            else:
                client.call(
                    [{"role": "user", "content": "hi"}], prompt_cache_key="conversation-key"
                )
    finally:
        await client.aclose()
    request.assert_called_once()
    if is_async:
        request.assert_awaited_once()
    sent = request.call_args.kwargs
    assert sent["extra_headers"][SESSION_AFFINITY_HEADER] == "conversation-key"
    assert sent["prompt_cache_key"] == "conversation-key"
    network.assert_not_called()
