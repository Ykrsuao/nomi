# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, optional input-window discovery for a configured chat endpoint."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

import httpx

from .route_identity import normalize_endpoint

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 2.0
_MAX_BYTES = 5 * 1024 * 1024
_MAX_MODELS = 5000


def _endpoint(config: Mapping[str, Any]) -> str | None:
    endpoints = set()
    for key in ("api_base", "base_url"):
        value = config.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if not isinstance(value, str) or any(ord(char) < 32 for char in value.strip()):
            return None
        endpoint = normalize_endpoint(value)
        if endpoint is None:
            return None
        parts = urlsplit(endpoint)
        if (
            parts.scheme not in {"http", "https"}
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.query
            or parts.fragment
        ):
            return None
        # Access also validates a present port before any request is sent.
        _ = parts.port
        endpoints.add(endpoint)
    return endpoints.pop() if len(endpoints) == 1 else None


def _wire_model(model: str, config: Mapping[str, Any]) -> str | None:
    if (
        not isinstance(model, str)
        or not model
        or model != model.strip()
        or len(model) > 500
        or any(ord(char) < 32 for char in model)
    ):
        return None
    provider = config.get("custom_llm_provider")
    if provider == "custom_openai":
        # With this explicit override LiteLLM preserves a leading openai/ in
        # the wire model. Leave provider-prefixed custom_openai IDs to the SDK.
        if model.startswith("custom_openai/"):
            return None
        return model
    if provider == "openai" or (provider in (None, "") and model.startswith("openai/")):
        return model.removeprefix("openai/") or None
    return None


def _token_limit(value: Any) -> int | None:
    if type(value) is int:
        return value if value > 0 else None
    if isinstance(value, float) and math.isfinite(value) and value > 0 and value.is_integer():
        return int(value)
    return None


def is_metadata_scope_safe(
    config: Mapping[str, Any],
    *,
    client_type: str,
    sdk_clients: Sequence[Any] = (),
) -> bool:
    """Prove that a plain metadata GET uses the inference authentication scope.

    Ambient credentials are only checked for presence, never resolved or
    forwarded. Completion SDKs retain scope captured at construction, so the
    owner also supplies those clients when checking a refresh or cached result.
    Their key values are compared in memory, never stored or logged here.
    """
    if client_type not in {"completion", "responses"} or config.get("client") is not None:
        return False
    if any(
        config.get(name) not in (None, "", {})
        for name in (
            "headers",
            "extra_headers",
            "default_headers",
            "organization",
            "project",
            "api_version",
        )
    ):
        return False
    key = config.get("api_key")
    if key is not None and not isinstance(key, str):
        return False

    # Inspect already-loaded globals without initializing a provider SDK (and
    # its model catalogue/network setup) for an optional metadata check.
    litellm = sys.modules.get("litellm")

    secret_manager = getattr(litellm, "secret_manager_client", None)
    if not key and (
        os.environ.get("OPENAI_API_KEY")
        or getattr(litellm, "api_key", None)
        or getattr(litellm, "openai_key", None)
        or secret_manager is not None
    ):
        return False
    if client_type == "responses":
        return True

    # The chat SDK and LiteLLM each add scope beyond the configured API key.
    # A configured secret manager may also supply an implicit organization.
    if (
        any(
            os.environ.get(name)
            for name in (
                "OPENAI_ORG_ID",
                "OPENAI_PROJECT_ID",
                "OPENAI_ORGANIZATION",
                "OPENAI_CUSTOM_HEADERS",
            )
        )
        or getattr(litellm, "organization", None)
        or getattr(litellm, "headers", None)
        or secret_manager is not None
    ):
        return False
    try:
        endpoint = _endpoint(config)
        return endpoint is not None and all(
            sdk.api_key == (key or "")
            and normalize_endpoint(str(sdk.base_url)) == endpoint
            and not sdk.organization
            and not sdk.project
            and not sdk._custom_headers
            for sdk in sdk_clients
        )
    except Exception:
        # SDK internals can change. Unknown scope must disable only metadata,
        # never session creation or a synchronous context-limit getter.
        return False


async def fetch_endpoint_model_info(
    model: str,
    config: Mapping[str, Any],
    client: httpx.AsyncClient,
    *,
    client_type: str,
) -> dict[str, int] | None:
    """Read the exact model's published input window using one optional GET.

    Only explicit OpenAI-compatible routes with an explicit endpoint qualify.
    A caller-owned SDK client or additional request header scope can change the
    real route and is left untouched.
    The HTTP client stays open; its TLS, certificate and transport settings are
    reused. A configured API key is used only on this exact origin, with no
    redirects, endpoint guesses, ambient credential resolution, model calls or
    retries. If inference needs implicit authentication or additional scope,
    this optional request is skipped.

    The complete exchange is bounded by two seconds, 5 MiB and 5,000 records.
    Unknown, malformed or ambiguous metadata returns None. Cancellation still
    propagates. Prices, capabilities and reply caps are never inferred here.
    """
    try:
        if not is_metadata_scope_safe(config, client_type=client_type):
            return None
        endpoint = _endpoint(config)
        wire_model = _wire_model(model, config)
        if endpoint is None or wire_model is None:
            return None
        key = config.get("api_key")
        headers = {"Authorization": f"Bearer {key}"} if isinstance(key, str) and key else {}
        async with asyncio.timeout(_TIMEOUT_SECONDS):
            async with client.stream(
                "GET",
                endpoint + "/models",
                headers=headers,
                timeout=_TIMEOUT_SECONDS,
                follow_redirects=False,
            ) as response:
                if response.status_code != 200:
                    return None
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(body) + len(chunk) > _MAX_BYTES:
                        return None
                    body.extend(chunk)
                payload = json.loads(body)
                records = payload.get("data") if isinstance(payload, dict) else None
                if not isinstance(records, list) or len(records) > _MAX_MODELS:
                    return None
                matches = [
                    item
                    for item in records
                    if isinstance(item, dict) and item.get("id") == wire_model
                ]
                if len(matches) != 1:
                    return None
                selected = matches[0]
                inputs = [
                    limit
                    for field in ("max_input_tokens", "context_window", "context_length")
                    if (limit := _token_limit(selected.get(field))) is not None
                ]
                if not inputs:
                    return None
                info = {"max_input_tokens": min(inputs)}
                output = _token_limit(selected.get("max_output_tokens"))
                if output is not None:
                    info["max_output_tokens"] = output
                return info
    except Exception:
        # Provider bodies and exception messages may include credentials or
        # server-supplied text. Optional metadata never logs either of them.
        logger.debug("Endpoint model metadata unavailable")
        return None
