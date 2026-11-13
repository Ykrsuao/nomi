# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Endpoint metadata follows the exact request route without making model calls."""

import asyncio
import logging
import os

import httpx
import pytest

from aster.unifiedllm import model_metadata

_BASE = "https://metadata.example.invalid/deployment/v1"
_MODEL = "openai/auto/model"
_ID = "auto/model"


@pytest.fixture(autouse=True)
def isolated_auth(monkeypatch):
    monkeypatch.setattr(os, "environ", {"LITELLM_LOCAL_MODEL_COST_MAP": "true"})
    from aster.unifiedllm._litellm import litellm

    for name in ("api_key", "openai_key", "organization", "headers", "secret_manager_client"):
        monkeypatch.setattr(litellm, name, None)


async def _fetch(records, *, model=_MODEL, config=None, client_type="completion"):
    requests = []

    def metadata(request):
        requests.append(request)
        return httpx.Response(200, json={"data": records})

    async with httpx.AsyncClient(transport=httpx.MockTransport(metadata)) as client:
        info = await model_metadata.fetch_endpoint_model_info(
            model,
            {"api_base": _BASE} if config is None else config,
            client,
            client_type=client_type,
        )
    return info, requests


@pytest.mark.parametrize("client_type", ["completion", "responses"])
async def test_gateway_auto_route_uses_its_own_advertised_input_window(client_type):
    from aster.unifiedllm.model_metadata import fetch_endpoint_model_info

    requests = []

    def metadata(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "workbuddy/deepseek-v4.1-flash", "context_window": 1_000_000},
                    {
                        "id": "auto/deepseek-v4.1-flash",
                        "context_window": 1_048_576,
                        "max_input_tokens": 1_048_576,
                        "max_output_tokens": 393_216,
                    },
                    {"id": "deepseek-v4.1-flash", "context_window": 128_000},
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(metadata)) as client:
        info = await fetch_endpoint_model_info(
            "openai/auto/deepseek-v4.1-flash",
            {"api_base": "https://metadata.example.invalid/deployment/v1/"},
            client,
            client_type=client_type,
        )
        assert not client.is_closed

    assert info == {"max_input_tokens": 1_048_576, "max_output_tokens": 393_216}
    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert str(requests[0].url) == "https://metadata.example.invalid/deployment/v1/models"
    assert "authorization" not in requests[0].headers


async def test_input_window_obeys_all_input_limits_and_ignores_prices_and_reply_cap():
    info, _ = await _fetch(
        [
            {
                "id": _ID,
                "context_window": 32_768,
                "context_length": 16_384.0,
                "max_input_tokens": 8192,
                "max_output_tokens": 2048,
                "max_tokens": 1024,
                "input_cost_per_token": 0,
                "pricing": {"prompt": "0", "completion": "0"},
            }
        ]
    )
    assert info == {"max_input_tokens": 8192, "max_output_tokens": 2048}


@pytest.mark.parametrize(
    "records",
    [
        [{"id": "other/model", "context_window": 8192}],
        [{"id": "model", "context_window": 8192}],
        [{"id": _MODEL, "context_window": 8192}],
        [{"id": _ID, "max_output_tokens": 8192, "max_tokens": 8192}],
        [{"id": _ID, "context_window": 8192}, {"id": _ID, "context_window": 16384}],
    ],
)
async def test_missing_input_limits_or_ambiguous_ids_do_not_supply_context(records):
    info, requests = await _fetch(records)
    assert info is None
    assert len(requests) == 1


@pytest.mark.parametrize("value", [True, False, 0, -1, "8192", None, [], {}, 3.5])
async def test_invalid_window_is_not_used(value):
    info, _ = await _fetch([{"id": _ID, "context_window": value}])
    assert info is None


@pytest.mark.parametrize(
    "model,provider,wire_id",
    [
        ("auto/model", "openai", "auto/model"),
        ("openai/auto/model", "openai", "auto/model"),
        ("openai/openai/model", "openai", "openai/model"),
        ("auto/model", "custom_openai", "auto/model"),
        ("openai/model", "custom_openai", "openai/model"),
    ],
)
async def test_explicit_openai_provider_preserves_the_wire_model(model, provider, wire_id):
    info, requests = await _fetch(
        [{"id": wire_id, "context_window": 8192}],
        model=model,
        config={"base_url": _BASE, "custom_llm_provider": provider},
    )
    assert info == {"max_input_tokens": 8192}
    assert len(requests) == 1


async def test_normalized_endpoint_aliases_and_explicit_key_share_one_request():
    info, requests = await _fetch(
        [{"id": _ID, "context_window": 8192}],
        config={
            "api_base": _BASE + "/",
            "base_url": "HTTPS://METADATA.EXAMPLE.INVALID/deployment/v1",
            "api_key": "synthetic-metadata-key",
        },
    )
    assert info == {"max_input_tokens": 8192}
    assert len(requests) == 1
    assert requests[0].headers["authorization"] == "Bearer synthetic-metadata-key"
    assert str(requests[0].url) == _BASE + "/models"


@pytest.mark.parametrize("client_type", ["completion", "responses"])
@pytest.mark.parametrize("source", ["OPENAI_API_KEY", "api_key", "openai_key"])
async def test_ambient_key_never_supplies_anonymous_metadata(monkeypatch, client_type, source):
    from aster.unifiedllm._litellm import litellm

    if source == "OPENAI_API_KEY":
        monkeypatch.setenv(source, "synthetic-ambient-key")
    else:
        monkeypatch.setattr(litellm, source, "synthetic-ambient-key")
    requests = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: requests.append(request) or httpx.Response(200)
        )
    ) as client:
        info = await model_metadata.fetch_endpoint_model_info(
            _MODEL, {"api_base": _BASE}, client, client_type=client_type
        )
    assert info is None
    assert requests == []


@pytest.mark.parametrize(
    "source,value",
    [
        ("OPENAI_ORG_ID", "synthetic-org"),
        ("OPENAI_PROJECT_ID", "synthetic-project"),
        ("OPENAI_ORGANIZATION", "synthetic-org"),
        ("OPENAI_CUSTOM_HEADERS", "Authorization: Bearer synthetic-other-key"),
        ("organization", "synthetic-org"),
        ("headers", {"x-tenant": "synthetic-tenant"}),
    ],
)
async def test_ambient_completion_scope_is_not_queried_with_an_explicit_key(
    monkeypatch, source, value
):
    from aster.unifiedllm._litellm import litellm

    if source.startswith("OPENAI_"):
        monkeypatch.setenv(source, value)
    else:
        monkeypatch.setattr(litellm, source, value)
    info, requests = await _fetch(
        [{"id": _ID, "context_window": 8192}],
        config={"api_base": _BASE, "api_key": "synthetic-explicit-key"},
    )
    assert info is None
    assert requests == []


@pytest.mark.parametrize("client_type", ["completion", "responses"])
async def test_environment_style_key_keeps_the_direct_sdk_literal_semantics(
    monkeypatch, client_type
):
    monkeypatch.setenv("METADATA_TEST_KEY", "synthetic-resolved-key")
    requests = []

    def metadata(request):
        requests.append(request)
        return httpx.Response(200, json={"data": [{"id": _ID, "context_window": 8192}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(metadata)) as client:
        info = await model_metadata.fetch_endpoint_model_info(
            _MODEL,
            {"api_base": _BASE, "api_key": "os.environ/METADATA_TEST_KEY"},
            client,
            client_type=client_type,
        )
    assert info == {"max_input_tokens": 8192}
    assert len(requests) == 1
    assert requests[0].headers["authorization"] == "Bearer os.environ/METADATA_TEST_KEY"


@pytest.mark.parametrize("client_type", ["completion", "responses"])
async def test_explicit_key_keeps_precedence_over_ambient_keys(monkeypatch, client_type):
    from aster.unifiedllm._litellm import litellm

    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-ambient-key")
    monkeypatch.setattr(litellm, "api_key", "synthetic-global-key")
    monkeypatch.setattr(litellm, "openai_key", "synthetic-provider-key")
    info, requests = await _fetch(
        [{"id": _ID, "context_window": 8192}],
        config={"api_base": _BASE, "api_key": "synthetic-explicit-key"},
        client_type=client_type,
    )
    assert info == {"max_input_tokens": 8192}
    assert requests[0].headers["authorization"] == "Bearer synthetic-explicit-key"


async def test_responses_route_ignores_completion_only_ambient_scope(monkeypatch):
    from aster.unifiedllm._litellm import litellm

    for name in ("OPENAI_ORG_ID", "OPENAI_PROJECT_ID", "OPENAI_ORGANIZATION"):
        monkeypatch.setenv(name, "synthetic-scope")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "Authorization: Bearer synthetic-other-key")
    monkeypatch.setattr(litellm, "organization", "synthetic-org")
    monkeypatch.setattr(litellm, "headers", {"x-tenant": "synthetic-tenant"})
    info, requests = await _fetch(
        [{"id": _ID, "context_window": 8192}],
        config={"api_base": _BASE, "api_key": "synthetic-explicit-key"},
        client_type="responses",
    )
    assert info == {"max_input_tokens": 8192}
    assert requests[0].headers["authorization"] == "Bearer synthetic-explicit-key"
    assert "openai-organization" not in requests[0].headers
    assert "openai-project" not in requests[0].headers
    assert "x-tenant" not in requests[0].headers


@pytest.mark.parametrize("client_type", ["completion", "responses"])
@pytest.mark.parametrize("key", [None, "synthetic-explicit-key"])
async def test_secret_manager_is_never_used_to_resolve_optional_metadata_auth(
    monkeypatch, client_type, key
):
    from aster.unifiedllm._litellm import litellm

    class NoSecretReads:
        def __getattr__(self, name):
            raise AssertionError("Optional metadata must not access the secret manager")

    monkeypatch.setattr(litellm, "secret_manager_client", NoSecretReads())
    info, requests = await _fetch(
        [{"id": _ID, "context_window": 8192}],
        config={"api_base": _BASE, "api_key": key},
        client_type=client_type,
    )
    if client_type == "responses" and key:
        assert info == {"max_input_tokens": 8192}
        assert requests[0].headers["authorization"] == "Bearer synthetic-explicit-key"
    else:
        assert info is None
        assert requests == []


@pytest.mark.parametrize(
    "model,config",
    [
        (_MODEL, {}),
        (_MODEL, {"api_base": _BASE, "base_url": "https://different.example.invalid/v1"}),
        (_MODEL, {"api_base": "https://user:password@metadata.example.invalid/v1"}),
        (_MODEL, {"api_base": _BASE + "?api_key=synthetic-key"}),
        (_MODEL, {"api_base": _BASE + "#fragment"}),
        (_MODEL, {"api_base": "ftp://metadata.example.invalid/v1"}),
        (_MODEL, {"api_base": "https://metadata.example.invalid:bad/v1"}),
        (_MODEL, {"api_base": "https://metadata.example.invalid/deploy\nment/v1"}),
        (_MODEL, {"api_base": _BASE, "client": object()}),
        (_MODEL, {"api_base": _BASE, "api_key": object()}),
        (_MODEL, {"api_base": _BASE, "api_key": False}),
        (_MODEL, {"api_base": _BASE, "custom_llm_provider": "anthropic"}),
        ("anthropic/model", {"api_base": _BASE}),
        ("auto/model", {"api_base": _BASE}),
        ("custom_openai/model", {"api_base": _BASE, "custom_llm_provider": "custom_openai"}),
        ("openai/", {"api_base": _BASE}),
    ],
)
async def test_unsupported_or_ambiguous_routes_make_no_metadata_request(model, config):
    info, requests = await _fetch([{"id": _ID, "context_window": 8192}], model=model, config=config)
    assert info is None
    assert requests == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("headers", {"Authorization": "Bearer separate-scope"}),
        ("extra_headers", {"x-tenant": "separate-scope"}),
        ("default_headers", {"x-api-key": "separate-scope"}),
        ("organization", "separate-scope"),
        ("project", "separate-scope"),
        ("api_version", "separate-scope"),
    ],
)
async def test_unimplemented_header_scopes_do_not_use_anonymous_metadata(field, value):
    info, requests = await _fetch(
        [{"id": _ID, "context_window": 8192}], config={"api_base": _BASE, field: value}
    )
    assert info is None
    assert requests == []


@pytest.mark.parametrize("status", [302, 401, 404, 503])
async def test_http_failure_never_retries_redirects_or_guesses_another_path(status):
    requests = []

    def metadata(request):
        requests.append(request)
        return httpx.Response(status, headers={"Location": "https://other.example.invalid/models"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(metadata), follow_redirects=True
    ) as client:
        info = await model_metadata.fetch_endpoint_model_info(
            _MODEL, {"api_base": _BASE}, client, client_type="completion"
        )
    assert info is None
    assert len(requests) == 1


async def test_total_timeout_cancels_the_metadata_request(monkeypatch):
    monkeypatch.setattr(model_metadata, "_TIMEOUT_SECONDS", 0.02)
    cancelled = asyncio.Event()

    async def metadata(request: httpx.Request) -> httpx.Response:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        raise AssertionError("The closed metadata gate must not complete")

    async with httpx.AsyncClient(transport=httpx.MockTransport(metadata)) as client:
        info = await model_metadata.fetch_endpoint_model_info(
            _MODEL, {"api_base": _BASE}, client, client_type="completion"
        )
    assert info is None
    assert cancelled.is_set()


async def test_caller_cancellation_propagates_and_cleans_up_request():
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def metadata(request: httpx.Request) -> httpx.Response:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        raise AssertionError("The closed metadata gate must not complete")

    async with httpx.AsyncClient(transport=httpx.MockTransport(metadata)) as client:
        task = asyncio.create_task(
            model_metadata.fetch_endpoint_model_info(
                _MODEL, {"api_base": _BASE}, client, client_type="completion"
            )
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert cancelled.is_set()


async def test_oversized_stream_stops_reading_and_closes_response(monkeypatch):
    monkeypatch.setattr(model_metadata, "_MAX_BYTES", 32)

    class Oversized(httpx.AsyncByteStream):
        count = 0
        closed = False

        async def __aiter__(self):
            for _ in range(20):
                self.count += 1
                yield b"a" * 20

        async def aclose(self):
            self.closed = True

    stream = Oversized()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=stream))
    ) as client:
        info = await model_metadata.fetch_endpoint_model_info(
            _MODEL, {"api_base": _BASE}, client, client_type="completion"
        )
    assert info is None
    assert stream.count == 2
    assert stream.closed


async def test_record_limit_is_enforced_before_selecting_a_model(monkeypatch):
    monkeypatch.setattr(model_metadata, "_MAX_MODELS", 1)
    info, _ = await _fetch([{"id": _ID, "context_window": 8192}, {"id": "other"}])
    assert info is None


async def test_failed_request_diagnostics_do_not_echo_credentials_or_server_text(caplog):
    private = "synthetic-sensitive-server-text"

    def metadata(request):
        raise httpx.ConnectError(private, request=request)

    caplog.set_level(logging.DEBUG, logger=model_metadata.__name__)
    async with httpx.AsyncClient(transport=httpx.MockTransport(metadata)) as client:
        info = await model_metadata.fetch_endpoint_model_info(
            _MODEL, {"api_base": _BASE, "api_key": private}, client, client_type="completion"
        )
    assert info is None
    messages = [
        record.getMessage() for record in caplog.records if record.name == model_metadata.__name__
    ]
    assert messages == ["Endpoint model metadata unavailable"]
    assert private not in "\n".join(messages)
