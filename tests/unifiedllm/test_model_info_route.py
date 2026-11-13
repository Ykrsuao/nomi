# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Model information must describe the provider and endpoint that receive the request."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest

from aster.unifiedllm import CompletionClient, ResponsesClient, UnifiedLLM, registry

MODEL = "minimax/minimax-m2.1"
BASE = "https://metadata-route.example.invalid/v1"
PROVIDER = "novita"


@pytest.fixture
def catalogue(monkeypatch: pytest.MonkeyPatch) -> dict[tuple[str, str | None, str | None], Any]:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "true")
    monkeypatch.setattr(registry, "MODELS", {})
    monkeypatch.setattr(registry, "_loaded", True)

    def forbid_http(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Catalogue lookup must not send model or metadata requests")

    monkeypatch.setattr(httpx.Client, "send", forbid_http)
    monkeypatch.setattr(httpx.AsyncClient, "send", forbid_http)
    from aster.unifiedllm._litellm import litellm

    entries: dict[tuple[str, str | None, str | None], Any] = {
        (MODEL, None, None): {"max_input_tokens": 1_000_000},
        (MODEL, PROVIDER, BASE): {
            "max_input_tokens": 204_800,
            "max_output_tokens": 128_000,
            "input_cost_per_token": 0.0000003,
            "output_cost_per_token": 0.0000012,
        },
    }

    def lookup(
        model: str, custom_llm_provider: str | None = None, api_base: str | None = None
    ) -> Any:
        value = entries.get((model, custom_llm_provider, api_base))
        if isinstance(value, Exception):
            raise value
        if value is None:
            raise LookupError("No catalogue entry for this route")
        return value

    monkeypatch.setattr(litellm, "get_model_info", lookup)
    return entries


@pytest.fixture
async def client_factory(
    catalogue: dict[tuple[str, str | None, str | None], Any],
) -> AsyncIterator[Callable[..., UnifiedLLM]]:
    clients: list[UnifiedLLM] = []

    def make(
        *, client_type: str = "completion", alias: bool = False, **settings: Any
    ) -> UnifiedLLM:
        config = {
            "model": MODEL,
            "custom_llm_provider": PROVIDER,
            "api_base": BASE,
            "api_key": "synthetic-metadata-key",
            **settings,
        }
        if alias:
            registry.MODELS["metadata-route"] = {
                "model_name": config.pop("model"),
                "media_inputs": [],
                "client_type": client_type,
                **config,
            }
            client = registry.get_llm_client("metadata-route", api_key="synthetic-metadata-key")
        else:
            client_class = ResponsesClient if client_type == "responses" else CompletionClient
            client = client_class(**config)
        clients.append(client)
        return client

    try:
        yield make
    finally:
        for client in clients:
            await client.aclose()


@pytest.mark.parametrize("client_type", ["completion", "responses"])
@pytest.mark.parametrize("alias", [False, True])
async def test_provider_context_applies_to_direct_and_registry_clients(
    client_factory: Callable[..., UnifiedLLM], client_type: str, alias: bool
) -> None:
    client = client_factory(client_type=client_type, alias=alias)

    assert client.context_window == 204_800
    assert client.get_model_info()["output_cost_per_token"] == 0.0000012


@pytest.mark.parametrize(
    "endpoints",
    [
        {"api_base": BASE},
        {"api_base": None, "base_url": BASE},
        {"api_base": BASE, "base_url": "https://unused.example.invalid/v1"},
    ],
)
async def test_metadata_uses_the_effective_endpoint_spelling(
    client_factory: Callable[..., UnifiedLLM], endpoints: dict[str, str | None]
) -> None:
    assert client_factory(**endpoints).context_window == 204_800


async def test_changing_provider_queries_its_own_catalogue_entry(
    client_factory: Callable[..., UnifiedLLM],
    catalogue: dict[tuple[str, str | None, str | None], Any],
) -> None:
    catalogue[(MODEL, "deepinfra", BASE)] = {"max_input_tokens": 65_536}
    client = client_factory()
    assert client.context_window == 204_800

    client.config["custom_llm_provider"] = "deepinfra"

    assert client.context_window == 65_536


async def test_responses_metadata_uses_base_url_that_overrides_api_base(
    client_factory: Callable[..., UnifiedLLM],
    catalogue: dict[tuple[str, str | None, str | None], Any],
) -> None:
    ignored_base = "https://ignored-responses-route.example.invalid/v1"
    catalogue[(MODEL, PROVIDER, ignored_base)] = {"max_input_tokens": 8192}
    client = client_factory(client_type="responses", api_base=ignored_base, base_url=BASE)

    assert client.get_model_info()["max_input_tokens"] == 204_800


async def test_missing_metadata_keeps_fallback_and_sanitizes_diagnostics(
    client_factory: Callable[..., UnifiedLLM],
    catalogue: dict[tuple[str, str | None, str | None], Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    catalogue[(MODEL, PROVIDER, BASE)] = ValueError(
        f"Failed route {BASE}?api_key=synthetic-secret-value"
    )
    client = client_factory()

    with caplog.at_level(logging.DEBUG, logger="aster.unifiedllm.unifiedllm"):
        assert client.context_window is None

    assert "Model metadata lookup unavailable (ValueError)" in caplog.text
    assert BASE not in caplog.text
    assert "synthetic-secret-value" not in caplog.text
