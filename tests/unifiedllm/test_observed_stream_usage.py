# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observed chat streaming keeps accounting through the real SDK/LiteLLM wrapper."""

import json
from contextlib import nullcontext

import httpx
import litellm
import pytest
from litellm.caching.llm_caching_handler import LLMClientCache

from aster.unifiedllm import CompletionClient, RetryConfig
from aster.unifiedllm.streaming import observe_stream
from aster.unifiedllm.unifiedllm import _ClientHttp

MESSAGES = [{"role": "user", "content": "Explain how streamed token usage survives aggregation."}]
USAGE = {
    "prompt_tokens": 321,
    "completion_tokens": 67,
    "total_tokens": 388,
    "prompt_tokens_details": {"cached_tokens": 15},
    "completion_tokens_details": {"reasoning_tokens": 9},
}


def _sse(include_usage, *, model="gpt-4o-mini"):
    base = {
        "id": "chatcmpl-accounting",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": model,
    }
    chunks = [
        {**base, "choices": [{"index": 0, "delta": {"content": "streamed answer"}}]},
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    if include_usage:
        chunks.append({**base, "choices": [], "usage": USAGE})
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"


@pytest.mark.parametrize(
    ("observed", "config", "expected_options", "usage_available"),
    [
        (True, {}, {"include_usage": True}, True),
        (True, {"stream_options": {}}, {"include_usage": True}, True),
        (True, {"stream_options": {"include_usage": True}}, {"include_usage": True}, True),
        (
            True,
            {"stream_options": {"include_obfuscation": False}},
            {"include_obfuscation": False, "include_usage": True},
            True,
        ),
        (True, {"stream_options": {"include_usage": False}}, {"include_usage": False}, False),
        (True, {"stream": False}, None, True),
        (False, {}, None, True),
        (False, {"stream": True, "stream_options": {}}, {}, False),
    ],
)
async def test_observed_stream_keeps_provider_usage_and_explicit_configuration(
    monkeypatch, observed, config, expected_options, usage_available
):
    requests = []
    original_config = json.loads(json.dumps(config))

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        if body.get("stream"):
            return httpx.Response(
                200,
                text=_sse(body.get("stream_options", {}).get("include_usage", False)),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-accounting",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "streamed answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": USAGE,
            },
        )

    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        _ClientHttp, "_httpx_hardening", staticmethod(lambda: {"transport": transport})
    )
    client = CompletionClient(
        "openai/gpt-4o-mini",
        api_base="https://endpoint.example/v1",
        api_key="synthetic",
        retry_config=RetryConfig(max_retries=0),
        **config,
    )
    seen = []
    try:
        with (
            observe_stream(lambda kind, value: seen.append((kind, value)))
            if observed
            else nullcontext()
        ):
            response = await client.acall(MESSAGES)
    finally:
        await client.aclose()

    assert len(requests) == 1
    assert requests[0].get("stream_options") == expected_options
    assert bool(requests[0].get("stream")) == config.get("stream", observed)
    assert config == original_config
    assert response.content == "streamed answer"
    assert response.usage is not None
    assert response.usage.input_tokens_reported is usage_available
    # This route has registered pricing. Exercise cost availability through
    # the actual SDK/LiteLLM wrapper, including observed-stream aggregation.
    assert response.usage.cost_known is usage_available
    if usage_available:
        assert response.usage.cost_usd > 0
    else:
        assert response.usage.cost_usd == 0
    if usage_available:
        assert response.usage.input_tokens == 321
        assert response.usage.output_tokens == 67
        assert response.usage.cached_input_tokens == 15
        assert response.usage.reasoning_tokens == 9
    else:
        # Explicit usage opt-out remains respected; the aggregation still has
        # the actual request messages for the local prompt-token fallback.
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0
    assert any(kind == "chat" for kind, _ in seen) == (observed and config.get("stream", True))


def test_litellm_filters_stream_options_for_native_anthropic_route():
    from litellm.utils import get_optional_params

    options = get_optional_params(
        model="claude-sonnet-4-5",
        custom_llm_provider="anthropic",
        stream=True,
        stream_options={"include_usage": True},
    )
    # Native providers consume this in LiteLLM's wrapper; it must not become
    # an unsupported field in the provider's HTTP request.
    assert options["stream"] is True
    assert "stream_options" not in options


@pytest.mark.parametrize("is_async", [False, True])
async def test_stream_with_unknown_price_does_not_borrow_response_model_price(
    monkeypatch, is_async
):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        # A compatible gateway may return a familiar model spelling even when
        # its requested route has different, unknown billing.
        return httpx.Response(200, text=_sse(True), headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        _ClientHttp, "_httpx_hardening", staticmethod(lambda: {"transport": transport})
    )
    client = CompletionClient(
        "openai/aster-unknown-price-test-model",
        api_base="https://endpoint.example/v1",
        api_key="synthetic",
        retry_config=RetryConfig(max_retries=0),
        stream=True,
        stream_options={"include_usage": True},
    )
    try:
        with observe_stream(lambda *_: None):
            response = await client.acall(MESSAGES) if is_async else client.call(MESSAGES)
    finally:
        await client.aclose()

    assert len(requests) == 1
    assert requests[0]["model"] == "aster-unknown-price-test-model"
    assert response.content == "streamed answer"
    assert response.usage is not None
    assert response.usage.input_tokens == 321
    assert response.usage.cost_known is False


@pytest.mark.parametrize(
    ("provider_usage", "reported_input", "reported_output", "expected_cost"),
    [
        ({}, None, None, None),
        ({"prompt_tokens": 321}, 321, None, None),
        ({"completion_tokens": 67}, None, 67, None),
        ({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, 0, 0, 0.0),
        ({"prompt_tokens": 321, "completion_tokens": 0}, 321, 0, 321 * 0.00000015),
        ({"prompt_tokens": 0, "completion_tokens": 67}, 0, 67, 67 * 0.0000006),
    ],
)
@pytest.mark.parametrize("is_async", [False, True])
async def test_stream_cost_never_prices_builder_fallback_counts(
    monkeypatch, provider_usage, reported_input, reported_output, expected_cost, is_async
):
    monkeypatch.setitem(globals(), "USAGE", provider_usage)

    def respond(request):
        return httpx.Response(200, text=_sse(True), headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        _ClientHttp, "_httpx_hardening", staticmethod(lambda: {"transport": transport})
    )
    client = CompletionClient(
        "openai/gpt-4o-mini",
        api_base="https://endpoint.example/v1",
        api_key="synthetic",
        retry_config=RetryConfig(max_retries=0),
        stream=True,
        stream_options={"include_usage": True},
    )
    try:
        with observe_stream(lambda *_: None):
            response = await client.acall(MESSAGES) if is_async else client.call(MESSAGES)
    finally:
        await client.aclose()

    assert response.content == "streamed answer"
    assert response.usage is not None
    assert response.usage.input_tokens_reported is (reported_input is not None)
    if reported_input is None:
        assert response.usage.input_tokens > 0  # Core consumers retain the local fallback.
    else:
        assert response.usage.input_tokens == reported_input
    if reported_output is None:
        assert response.usage.output_tokens > 0
    else:
        assert response.usage.output_tokens == reported_output
    assert response.usage.cost_known is (expected_cost is not None)
    assert response.usage.cost_usd == pytest.approx(expected_cost or 0)


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("model", ["gpt-4o-mini", "aster-unknown-price-test-model"])
@pytest.mark.parametrize("cost", [0.0, 0.25])
@pytest.mark.parametrize("complete_usage", [False, True])
async def test_stream_preserves_provider_cost_even_with_unknown_route_or_partial_usage(
    monkeypatch, is_async, model, cost, complete_usage
):
    provider_usage = {**USAGE, "cost": cost} if complete_usage else {"cost": cost}
    monkeypatch.setitem(globals(), "USAGE", provider_usage)

    def respond(request):
        return httpx.Response(200, text=_sse(True), headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        _ClientHttp, "_httpx_hardening", staticmethod(lambda: {"transport": transport})
    )
    client = CompletionClient(
        f"openai/{model}",
        api_base="https://endpoint.example/v1",
        api_key="synthetic",
        retry_config=RetryConfig(max_retries=0),
        stream=True,
        stream_options={"include_usage": True},
    )
    try:
        response = await client.acall(MESSAGES) if is_async else client.call(MESSAGES)
    finally:
        await client.aclose()

    assert response.content == "streamed answer"
    assert response.usage is not None
    assert response.usage.input_tokens_reported is complete_usage
    assert response.usage.cost_known is True
    assert response.usage.cost_usd == cost


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("custom_prices", [False, True])
async def test_stream_prices_requested_route_instead_of_known_response_model(
    monkeypatch, is_async, custom_prices
):
    monkeypatch.setitem(globals(), "USAGE", {"prompt_tokens": 321, "completion_tokens": 67})
    # LiteLLM registers per-call custom prices in its process-wide cost map.
    monkeypatch.setattr(litellm, "model_cost", dict(litellm.model_cost))

    def respond(request):
        return httpx.Response(
            200, text=_sse(True, model="gpt-4o"), headers={"content-type": "text/event-stream"}
        )

    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        _ClientHttp, "_httpx_hardening", staticmethod(lambda: {"transport": transport})
    )
    price_config = (
        {"input_cost_per_token": 0.000003, "output_cost_per_token": 0.000009}
        if custom_prices
        else {}
    )
    model = "aster-custom-price-test-model" if custom_prices else "gpt-4o-mini"
    client = CompletionClient(
        f"openai/{model}",
        api_base="https://endpoint.example/v1",
        api_key="synthetic",
        retry_config=RetryConfig(max_retries=0),
        stream=True,
        stream_options={"include_usage": True},
        **price_config,
    )
    try:
        response = await client.acall(MESSAGES) if is_async else client.call(MESSAGES)
    finally:
        await client.aclose()

    assert response.usage is not None
    assert response.usage.cost_known is True
    expected = (
        321 * 0.000003 + 67 * 0.000009 if custom_prices else 321 * 0.00000015 + 67 * 0.0000006
    )
    assert response.usage.cost_usd == pytest.approx(expected)
