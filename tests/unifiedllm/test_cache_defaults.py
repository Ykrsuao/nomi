# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default registry clients must honor the default renderer's cache boundary."""

import copy
import json

import httpx
import pytest

from aster.context_blocks.formatter import OpenAIProviderFormatter
from aster.context_blocks.models import BlockMetadata, ResolvedBlock, Role
from aster.context_blocks.renderer import render_context
from aster.context_blocks.renderers.cached import CachedBlockFormatter
from aster.unifiedllm import CacheBoundary, ResponsesClient, RetryConfig, registry


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("style", ["responses", "anthropic", "chat"])
@pytest.mark.parametrize("mode", ["default", "no-boundary", "disabled"])
async def test_registry_default_cache_on_wire(monkeypatch, style, mode, asynchronous):
    """Exercise renderer -> registry client -> SDK -> HTTP, with no cache opt-in."""
    entry = {
        "model_name": "anthropic/claude-sonnet-4-5" if style == "anthropic" else "openai/gpt-4o",
        "client_type": "responses" if style == "responses" else "completion",
        "api_base": "https://models.example"
        if style == "anthropic"
        else "https://models.example/v1",
        "max_tokens": 100,
    }
    monkeypatch.setattr(registry, "MODELS", {"cache-test": entry})
    monkeypatch.setattr(registry, "ensure_loaded", lambda: None)
    bodies = []

    def respond(request):
        assert request.url.host == "models.example"
        bodies.append(json.loads(request.content))
        if style == "responses":
            assert request.url.path == "/v1/responses"
            data = {
                "id": "r",
                "object": "response",
                "created_at": 1,
                "model": "gpt-4o",
                "status": "completed",
                "output": [],
                "usage": {"input_tokens": 10, "output_tokens": 0, "total_tokens": 10},
            }
        elif style == "anthropic":
            assert request.url.path == "/v1/messages"
            data = {
                "id": "m",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "content": [{"type": "text", "text": "OK"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 1},
            }
        else:
            assert request.url.path == "/v1/chat/completions"
            data = {
                "id": "c",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            }
        return httpx.Response(200, json=data, request=request)

    async def send_async(self, request, **kwargs):
        return respond(request)

    monkeypatch.setattr(httpx.Client, "send", lambda self, request, **kwargs: respond(request))
    monkeypatch.setattr(httpx.AsyncClient, "send", send_async)
    messages = render_context(
        [
            ResolvedBlock(
                key="instructions",
                content="Stable instructions",
                role=Role.SYSTEM,
                metadata=BlockMetadata(static=True),
            ),
            ResolvedBlock(
                key="live",
                content="Volatile suffix",
                role=Role.SYSTEM,
                metadata=BlockMetadata(static=False, user_block=True),
            ),
        ],
        block_formatter=CachedBlockFormatter(),
        provider_formatter=OpenAIProviderFormatter(),
    ).output
    assert sum(isinstance(m, CacheBoundary) for m in messages) == 1
    if mode == "no-boundary":
        messages = [m for m in messages if not isinstance(m, CacheBoundary)]
    before = copy.deepcopy(messages)
    overrides = {"cache_breakpoint": None} if mode == "disabled" else {}
    async with registry.get_llm_client(
        "cache-test",
        api_key="test",
        retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
        **overrides,
    ) as client:
        if asynchronous:
            await client.acall(messages)
        else:
            client.call(messages)
    assert messages == before
    assert "cache_breakpoint" not in entry
    assert len(bodies) == 1
    body = bodies[0]
    encoded = json.dumps(body)
    assert "aster_cache_boundary" not in encoded
    explicit = style == "responses" and mode == "default"
    assert ("prompt_cache_options" in body) is explicit
    assert encoded.count('"prompt_cache_breakpoint"') == int(explicit)
    if explicit:
        assert body["prompt_cache_options"]["mode"] == "explicit"
        marked = body["input"][0]["content"][-1]
        assert "Stable instructions" in marked["text"]
        assert marked["prompt_cache_breakpoint"] == {"mode": "explicit"}
    # Completion's existing no-boundary behavior marks only leading instructions.
    assert ('"cache_control"' in encoded) is (style == "anthropic" and mode != "disabled")
    assert "Volatile suffix" in encoded


@pytest.mark.parametrize(
    "messages",
    [
        [CacheBoundary(), {"role": "user", "content": "live"}],
        [
            {"role": "assistant", "content": "answer"},
            CacheBoundary(),
            {"role": "user", "content": "live"},
        ],
    ],
)
def test_auto_without_eligible_prefix_stays_implicit(messages, caplog):
    with ResponsesClient("openai/gpt-4o", api_key="test") as client:
        wire, instructions, explicit = client._prepare_cache_boundary(messages, responses=True)
    assert wire == [m for m in messages if not isinstance(m, CacheBoundary)]
    assert instructions is None
    assert not explicit
    assert not caplog.text


@pytest.fixture
def responses_requests(monkeypatch):
    from aster.unifiedllm._litellm import litellm

    monkeypatch.setattr(litellm, "api_base", None)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    requests = []

    def send(self, request, **kwargs):
        requests.append(request)
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "r",
                "object": "response",
                "created_at": 1,
                "model": json.loads(request.content)["model"],
                "status": "completed",
                "output": [],
                "usage": {"input_tokens": 10, "output_tokens": 0, "total_tokens": 10},
            },
        )

    monkeypatch.setattr(httpx.Client, "send", send)
    return requests


@pytest.mark.parametrize(
    ("model", "api_base", "mode", "keeps_marker"),
    [
        ("openai/gpt-4o", "https://models.example/v1", "explicit", True),
        ("openai/gpt-4o", "https://api.openai.com/v1", "explicit", False),
        ("openai/gpt-4o", "https://api.openai.com./v1", "explicit", False),
        ("openai/gpt-4o", "https://eu.api.openai.com/v1", "explicit", False),
        ("openai/gpt-4o", None, "explicit", False),
        ("openai/gpt-4o", "https://models.example/v1", "implicit", False),
        ("openai/gpt-4o", "https://models.example/v1", "disabled", False),
        ("openai/gpt-4o", "https://models.example/v1", "no-boundary", False),
        ("azure/gpt-4o", "https://models.example/openai/v1", "explicit", False),
        ("openai/gpt-5.6", "https://api.openai.com/v1", "explicit", True),
    ],
)
def test_responses_cache_opt_in_is_scoped_to_custom_openai_endpoints(
    responses_requests, model, api_base, mode, keeps_marker
):
    # Supply an existing marker too: disabling our mapping must not accidentally
    # opt the request into the custom-endpoint compatibility path.
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Stable prefix",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ],
        },
        {"role": "user", "content": "Volatile suffix"},
    ]
    if mode != "no-boundary":
        messages.insert(1, CacheBoundary())
    before = copy.deepcopy(messages)
    options = {"prompt_cache_options": {"mode": "implicit"}} if mode == "implicit" else {}
    mapping = "auto" if mode == "no-boundary" else "openai" if mode == "explicit" else None
    with ResponsesClient(
        model,
        api_key="test",
        api_base=api_base,
        drop_params=True,
        cache_breakpoint=mapping,
        retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
        **options,
    ) as client:
        client.call(messages)
    assert messages == before
    assert len(responses_requests) == 1
    body = json.loads(responses_requests[0].content)
    assert json.dumps(body).count('"prompt_cache_breakpoint"') == int(keeps_marker)
    assert "prompt_cache_breakpoint" not in repr(body["input"][-1])
    if mode in {"explicit", "implicit"}:
        assert body["prompt_cache_options"] == {"mode": mode}
    else:
        assert "prompt_cache_options" not in body


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        (
            {"prompt_cache_options": {"root_option": 1, "mode": "implicit"}},
            {"root_option": 1, "mode": "explicit"},
        ),
        (
            {"extra_body": {"prompt_cache_options": {"extra_option": 2, "mode": "implicit"}}},
            {"extra_option": 2, "mode": "explicit"},
        ),
        (
            {
                "prompt_cache_options": {"root_option": 1, "shared": "root"},
                "extra_body": {
                    "prompt_cache_options": {"extra_option": 2, "shared": "extra"},
                    "custom": 123,
                },
            },
            {"root_option": 1, "extra_option": 2, "shared": "extra", "mode": "explicit"},
        ),
    ],
)
def test_explicit_cache_preserves_caller_options_on_wire(responses_requests, options, expected):
    before = copy.deepcopy(options)
    with ResponsesClient(
        "openai/gpt-4o",
        api_key="test",
        api_base="https://models.example/v1",
        drop_params=True,
        retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
    ) as client:
        client.call(
            [
                {"role": "user", "content": "Stable prefix"},
                CacheBoundary(),
                {"role": "user", "content": "Volatile suffix"},
            ],
            **options,
        )
    assert options == before
    assert len(responses_requests) == 1
    body = json.loads(responses_requests[0].content)
    assert body["prompt_cache_options"] == expected
    assert json.dumps(body).count('"prompt_cache_breakpoint"') == 1
    assert body["input"][0]["content"][-1]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert "prompt_cache_breakpoint" not in repr(body["input"][-1])
    if "custom" in options.get("extra_body", {}):
        assert body["custom"] == 123


@pytest.mark.parametrize(
    ("api_base_env", "base_url_env", "global_base", "api_base", "expected_host"),
    [
        ("models.example", None, None, None, "models.example"),
        (None, "models.example", None, None, "models.example"),
        (None, None, "models.example", None, "models.example"),
        ("models.example", "api.openai.com", None, None, "api.openai.com"),
        ("api.openai.com", "models.example", None, None, "models.example"),
        (None, "models.example", "api.openai.com", None, "api.openai.com"),
        (None, None, "models.example", "api.openai.com", "api.openai.com"),
        (None, "api.openai.com", "api.openai.com", "models.example", "models.example"),
    ],
)
def test_explicit_cache_follows_the_effective_openai_endpoint(
    monkeypatch,
    responses_requests,
    api_base_env,
    base_url_env,
    global_base,
    api_base,
    expected_host,
):
    from aster.unifiedllm._litellm import litellm

    if api_base_env:
        monkeypatch.setenv("OPENAI_API_BASE", f"https://{api_base_env}/v1")
    if base_url_env:
        monkeypatch.setenv("OPENAI_BASE_URL", f"https://{base_url_env}/v1")
    if global_base:
        monkeypatch.setattr(litellm, "api_base", f"https://{global_base}/v1")
    with ResponsesClient(
        "openai/gpt-4o",
        api_key="test",
        api_base=f"https://{api_base}/v1" if api_base else None,
        drop_params=True,
        retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
    ) as client:
        client.call(
            [
                {"role": "user", "content": "Stable prefix"},
                CacheBoundary(),
                {"role": "user", "content": "Volatile suffix"},
            ],
        )
    assert len(responses_requests) == 1
    assert responses_requests[0].url.host == expected_host
    body = json.loads(responses_requests[0].content)
    assert body["prompt_cache_options"] == {"mode": "explicit"}
    assert json.dumps(body).count('"prompt_cache_breakpoint"') == int(
        expected_host == "models.example"
    )
