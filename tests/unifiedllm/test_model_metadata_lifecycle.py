# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fetched context limits belong to one client route and one completed refresh."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from aster.unifiedllm import (
    AdmissionControl,
    AdmissionControlConfig,
    CompletionClient,
    RetryConfig,
    registry,
)
from aster.unifiedllm.unifiedllm import _ClientHttp

_MODEL = "openai/auto/metadata-lifecycle-test"
_BASE = "https://metadata-lifecycle.example.invalid/v1"
_WINDOW = 1_048_576


class _MetadataServer:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.unexpected: list[str] = []
        self.status = 200
        self.handler: Callable[[httpx.Request], Awaitable[int]] | None = None

    async def send(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method != "GET" or request.url.path != "/v1/models":
            self.unexpected.append(f"{request.method} {request.url.path}")
            raise AssertionError("Only the fake metadata endpoint may be called")
        window = await self.handler(request) if self.handler is not None else _WINDOW
        return httpx.Response(
            self.status,
            json={"data": [{"id": _MODEL.removeprefix("openai/"), "max_input_tokens": window}]},
            request=request,
        )


@pytest.fixture
def metadata_server(monkeypatch: pytest.MonkeyPatch) -> _MetadataServer:
    server = _MetadataServer()

    async def send(
        client: httpx.AsyncClient, request: httpx.Request, **kwargs: Any
    ) -> httpx.Response:
        return await server.send(request)

    def forbid_sync_request(
        client: httpx.Client, request: httpx.Request, **kwargs: Any
    ) -> httpx.Response:
        server.unexpected.append("synchronous HTTP")
        raise AssertionError("No synchronous HTTP requests are allowed")

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    monkeypatch.setattr(httpx.Client, "send", forbid_sync_request)
    return server


@pytest.fixture(autouse=True)
def isolated_auth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "environ", {"LITELLM_LOCAL_MODEL_COST_MAP": "true"})
    monkeypatch.setenv("ASTER_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("ASTER_PROJECT_DIR", str(tmp_path / "project"))
    monkeypatch.setattr(registry, "_loaded", True)
    monkeypatch.setattr(registry, "MODELS", {})
    from aster.unifiedllm._litellm import litellm

    for name in ("api_key", "openai_key", "organization", "headers", "secret_manager_client"):
        monkeypatch.setattr(litellm, name, None)


@pytest.fixture
async def client_factory(
    metadata_server: _MetadataServer,
) -> AsyncIterator[Callable[..., CompletionClient]]:
    clients: list[CompletionClient] = []

    def make(**overrides: Any) -> CompletionClient:
        config: dict[str, Any] = {
            "model": _MODEL,
            "api_base": _BASE,
            "api_key": "metadata-test-key",
        }
        config.update(overrides)
        client = CompletionClient(**config)
        clients.append(client)
        return client

    try:
        yield make
    finally:
        for client in clients:
            await client.aclose()
        assert metadata_server.unexpected == []


@pytest.mark.parametrize("changed", ["model", "api_base", "api_key"])
async def test_changed_route_does_not_reuse_fetched_context(
    client_factory: Callable[..., CompletionClient], changed: str
) -> None:
    client = client_factory()
    await client.refresh_model_info()
    assert client.context_window == _WINDOW

    if changed == "model":
        client.model = "openai/auto/another-lifecycle-model"
    else:
        client.config[changed] = (
            "https://another-metadata.example.invalid/v1"
            if changed == "api_base"
            else "another-metadata-test-key"
        )

    assert client.context_window is None
    assert client.get_context_limits().context_window is None


async def test_per_call_endpoint_override_keeps_metadata_scoped_to_original_route(
    client_factory: Callable[..., CompletionClient], metadata_server: _MetadataServer
) -> None:
    client = client_factory()
    await client.refresh_model_info()

    assert client.get_context_limits().context_window == _WINDOW
    assert (
        client.get_context_limits(
            {"api_base": "https://another-metadata.example.invalid/v1"}
        ).context_window
        is None
    )
    assert client.get_context_limits().context_window == _WINDOW
    assert len(metadata_server.requests) == 1


async def test_explicit_window_remains_authoritative_after_metadata_refresh(
    client_factory: Callable[..., CompletionClient],
) -> None:
    client = client_factory(context_window=8192)
    await client.refresh_model_info()

    assert client.context_window == 8192
    assert client.get_context_limits().context_window == 8192
    assert "context_window" not in client.config


async def test_clients_keep_independent_metadata_and_public_results_are_detached(
    client_factory: Callable[..., CompletionClient], metadata_server: _MetadataServer
) -> None:
    async def per_key(request: httpx.Request) -> int:
        return 32_768 if request.headers["Authorization"] == "Bearer tenant-b" else _WINDOW

    metadata_server.handler = per_key
    first = client_factory()
    second = client_factory(api_key="tenant-b")
    await first.refresh_model_info()
    await second.refresh_model_info()

    public_info = first.get_model_info()
    public_info["max_input_tokens"] = 1
    assert first.context_window == _WINDOW
    assert second.context_window == 32_768
    assert len(metadata_server.requests) == 2


async def test_older_refresh_cannot_replace_newer_response(
    client_factory: Callable[..., CompletionClient], metadata_server: _MetadataServer
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def out_of_order(request: httpx.Request) -> int:
        if len(metadata_server.requests) == 1:
            entered.set()
            await release.wait()
            return 32_768
        return _WINDOW

    metadata_server.handler = out_of_order
    client = client_factory()
    older = asyncio.create_task(client.refresh_model_info())
    try:
        async with asyncio.timeout(5):
            await entered.wait()
            await client.refresh_model_info()
            assert client.context_window == _WINDOW
            release.set()
            await older
        assert client.context_window == _WINDOW
    finally:
        release.set()
        await asyncio.gather(older, return_exceptions=True)


async def test_route_change_while_refreshing_discards_the_old_result(
    client_factory: Callable[..., CompletionClient], metadata_server: _MetadataServer
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def held(request: httpx.Request) -> int:
        entered.set()
        await release.wait()
        return _WINDOW

    metadata_server.handler = held
    client = client_factory()
    refresh = asyncio.create_task(client.refresh_model_info())
    try:
        async with asyncio.timeout(5):
            await entered.wait()
            client.config["api_key"] = "replacement-metadata-test-key"
            release.set()
            await refresh
        assert client.context_window is None
    finally:
        release.set()
        await asyncio.gather(refresh, return_exceptions=True)


async def test_cancelled_refresh_propagates_and_a_later_refresh_can_succeed(
    client_factory: Callable[..., CompletionClient], metadata_server: _MetadataServer
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def held(request: httpx.Request) -> int:
        entered.set()
        await release.wait()
        return _WINDOW

    metadata_server.handler = held
    client = client_factory()
    refresh = asyncio.create_task(client.refresh_model_info())
    try:
        async with asyncio.timeout(5):
            await entered.wait()
            refresh.cancel()
            with pytest.raises(asyncio.CancelledError):
                await refresh
        assert client.context_window is None
        metadata_server.handler = None
        await client.refresh_model_info()
        assert client.context_window == _WINDOW
    finally:
        release.set()
        await asyncio.gather(refresh, return_exceptions=True)


async def test_failed_refresh_does_not_keep_unverified_old_limits(
    client_factory: Callable[..., CompletionClient], metadata_server: _MetadataServer
) -> None:
    client = client_factory()
    await client.refresh_model_info()
    assert client.context_window == _WINDOW

    metadata_server.status = 503
    await client.refresh_model_info()
    assert client.context_window is None


async def test_admission_wrapper_refreshes_the_underlying_client(
    client_factory: Callable[..., CompletionClient], metadata_server: _MetadataServer
) -> None:
    client = client_factory()
    wrapped = AdmissionControl(
        client,
        AdmissionControlConfig(max_in_flight=2, concurrency_group="model-metadata-lifecycle"),
    )
    await wrapped.refresh_model_info()

    assert wrapped.context_window == client.context_window == _WINDOW
    assert wrapped.get_context_limits().context_window == _WINDOW
    assert len(metadata_server.requests) == 1


async def test_client_without_explicit_endpoint_makes_no_metadata_request(
    client_factory: Callable[..., CompletionClient], metadata_server: _MetadataServer
) -> None:
    client = client_factory(api_base=None)
    await client.refresh_model_info()

    assert client.context_window is None
    assert metadata_server.requests == []


@pytest.mark.parametrize("remove_after_init", [False, True])
@pytest.mark.parametrize(
    "source,header,scope",
    [
        ("OPENAI_API_KEY", "authorization", "Bearer synthetic-ambient-key"),
        ("OPENAI_ORG_ID", "openai-organization", "synthetic-org"),
        ("OPENAI_PROJECT_ID", "openai-project", "synthetic-project"),
        ("OPENAI_CUSTOM_HEADERS", "authorization", "Bearer synthetic-header-key"),
    ],
)
async def test_metadata_does_not_cross_the_actual_sdk_ambient_scope(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    header: str,
    scope: str,
    remove_after_init: bool,
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET" and request.url.path == "/v1/models":
            window = 16_384 if request.headers.get(header) == scope else 8192
            return httpx.Response(
                200,
                json={"data": [{"id": _MODEL.removeprefix("openai/"), "context_window": window}]},
            )
        assert request.method == "POST" and request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-metadata-auth",
                "object": "chat.completion",
                "created": 1,
                "model": _MODEL.removeprefix("openai/"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "synthetic answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        _ClientHttp, "_httpx_hardening", staticmethod(lambda: {"transport": transport})
    )
    # Also bind SDK-created fallback clients to the mock; no request may reach the network.
    monkeypatch.setattr(httpx.AsyncClient, "_transport_for_url", lambda self, url: transport)
    monkeypatch.setattr(httpx.Client, "_transport_for_url", lambda self, url: transport)
    value = f"Authorization: {scope}" if source == "OPENAI_CUSTOM_HEADERS" else scope
    if source == "OPENAI_API_KEY":
        value = scope.removeprefix("Bearer ")
    monkeypatch.setenv(source, value)
    client = CompletionClient(
        _MODEL,
        api_base=_BASE,
        api_key=None if source == "OPENAI_API_KEY" else "synthetic-explicit-key",
        retry_config=RetryConfig(max_retries=0),
    )
    try:
        if remove_after_init:
            monkeypatch.delenv(source)
        await client.refresh_model_info()
        selected_window = client.context_window
        response = await client.acall([{"role": "user", "content": "Synthetic request."}])
    finally:
        await client.aclose()

    assert response.content == "synthetic answer"
    inference = [request for request in requests if request.method == "POST"]
    assert len(inference) == 1
    assert inference[0].headers.get(header) == scope
    print(
        {
            "source": source,
            "removed_after_init": remove_after_init,
            "requests": [
                {
                    "method": request.method,
                    "path": request.url.path,
                    "same_scope_as_inference": request.headers.get(header) == scope,
                }
                for request in requests
            ],
            "selected_window": selected_window,
        }
    )
    assert selected_window is None
    assert all(request.method == "POST" for request in requests)


_AMBIENT_SOURCES = [
    "OPENAI_API_KEY",
    "OPENAI_ORG_ID",
    "OPENAI_PROJECT_ID",
    "OPENAI_ORGANIZATION",
    "OPENAI_CUSTOM_HEADERS",
    "api_key",
    "openai_key",
    "organization",
    "headers",
    "secret_manager_client",
]


def _set_ambient_scope(monkeypatch: pytest.MonkeyPatch, source: str) -> None:
    from aster.unifiedllm._litellm import litellm

    if source == "secret_manager_client":
        monkeypatch.setattr(litellm, source, object())
    elif source.startswith("OPENAI_"):
        value = "x-tenant: synthetic-tenant" if source == "OPENAI_CUSTOM_HEADERS" else "synthetic"
        monkeypatch.setenv(source, value)
    else:
        value = {"x-tenant": "synthetic-tenant"} if source == "headers" else "synthetic"
        monkeypatch.setattr(litellm, source, value)


@pytest.mark.parametrize("source", _AMBIENT_SOURCES)
async def test_new_ambient_scope_invalidates_cached_metadata(
    client_factory: Callable[..., CompletionClient], monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    config = {"api_key": None} if source in {"OPENAI_API_KEY", "api_key", "openai_key"} else {}
    client = client_factory(**config)
    await client.refresh_model_info()
    assert client.context_window == _WINDOW

    _set_ambient_scope(monkeypatch, source)

    assert client.context_window is None
    assert client.get_context_limits().context_window is None


@pytest.mark.parametrize("source", _AMBIENT_SOURCES)
async def test_ambient_scope_change_while_refreshing_discards_the_result(
    client_factory: Callable[..., CompletionClient],
    metadata_server: _MetadataServer,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def held(request: httpx.Request) -> int:
        entered.set()
        await release.wait()
        return _WINDOW

    metadata_server.handler = held
    config = {"api_key": None} if source in {"OPENAI_API_KEY", "api_key", "openai_key"} else {}
    client = client_factory(**config)
    refresh = asyncio.create_task(client.refresh_model_info())
    try:
        async with asyncio.timeout(5):
            await entered.wait()
            _set_ambient_scope(monkeypatch, source)
            release.set()
            await refresh
        assert client.context_window is None
    finally:
        release.set()
        await asyncio.gather(refresh, return_exceptions=True)


async def test_unreadable_sdk_scope_invalidates_metadata_without_failing_the_session(
    client_factory: Callable[..., CompletionClient],
    metadata_server: _MetadataServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnreadableScope:
        @property
        def api_key(self) -> str:
            raise RuntimeError("Synthetic unsupported SDK scope")

    client = client_factory()
    await client.refresh_model_info()
    assert client.context_window == _WINDOW
    assert client._http is not None

    with monkeypatch.context() as patch:
        patch.setattr(client._http, "_openai_clients", [UnreadableScope()])
        assert client.context_window is None
        await client.refresh_model_info()
        assert client.context_window is None
        assert len(metadata_server.requests) == 1


async def test_changed_endpoint_cannot_query_metadata_for_a_stale_sdk_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed_base = "https://changed-metadata.example.invalid/v1"
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET" and request.url.path == "/v1/models":
            window = 32_768 if str(request.url).startswith(changed_base) else 8192
            return httpx.Response(
                200,
                json={"data": [{"id": _MODEL.removeprefix("openai/"), "context_window": window}]},
            )
        assert request.method == "POST" and request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-metadata-route",
                "object": "chat.completion",
                "created": 1,
                "model": _MODEL.removeprefix("openai/"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "synthetic answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        _ClientHttp, "_httpx_hardening", staticmethod(lambda: {"transport": transport})
    )
    monkeypatch.setattr(httpx.AsyncClient, "_transport_for_url", lambda self, url: transport)
    monkeypatch.setattr(httpx.Client, "_transport_for_url", lambda self, url: transport)
    client = CompletionClient(
        _MODEL,
        api_base=_BASE,
        api_key="synthetic-explicit-key",
        retry_config=RetryConfig(max_retries=0),
    )
    try:
        client.config["api_base"] = changed_base
        await client.refresh_model_info()
        selected_window = client.context_window
        response = await client.acall([{"role": "user", "content": "Synthetic request."}])
    finally:
        await client.aclose()

    assert response.content == "synthetic answer"
    inference = [request for request in requests if request.method == "POST"]
    assert len(inference) == 1
    assert str(inference[0].url) == _BASE + "/chat/completions"
    print(
        {
            "requests": [
                {"method": request.method, "url": str(request.url)} for request in requests
            ],
            "selected_window": selected_window,
            "inference_endpoint_window": 8192,
        }
    )
    assert selected_window is None
    assert all(request.method == "POST" for request in requests)
