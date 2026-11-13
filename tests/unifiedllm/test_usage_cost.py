# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Missing billing metadata must remain distinct from an explicitly free call."""

from types import SimpleNamespace

import pytest

from aster.llm_types import LLMUsage
from aster.unifiedllm.unifiedllm import _extract_usage


@pytest.mark.parametrize("cost", [0, 0.0, 0.25, "0", "0.25"])
@pytest.mark.parametrize("key", ["cost_usd", "cost"])
def test_provider_explicit_cost_is_known_including_zero(key, cost):
    usage = LLMUsage.from_provider({"input_tokens": 40, key: cost})

    assert usage is not None
    assert usage.input_tokens == 40
    assert usage.cost_usd == float(cost)
    assert usage.cost_known is True


@pytest.mark.parametrize(
    "value",
    [None, True, False, "unknown", "NaN", "inf", "-1", float("nan"), float("inf"), -1.0, [], {}],
)
def test_provider_missing_or_invalid_cost_preserves_tokens_without_claiming_free(value):
    usage = LLMUsage.from_provider({"input_tokens": 40, "cost_usd": value})

    assert usage is not None
    assert usage.input_tokens == 40
    assert usage.cost_usd == 0.0
    assert usage.cost_known is False


@pytest.mark.parametrize("cost", [0.0, 0.25])
def test_explicit_normalized_cost_is_known(cost):
    usage = LLMUsage(input_tokens=40, cost_usd=cost)

    assert usage.cost_known is True
    assert LLMUsage.model_validate_json(usage.model_dump_json()) == usage


def test_unpriced_usage_stays_unknown_through_serialization_and_provider_normalization():
    usage = LLMUsage(input_tokens=40)

    assert usage.cost_usd == 0.0
    assert usage.cost_known is False
    assert LLMUsage.model_validate_json(usage.model_dump_json()).cost_known is False
    normalized = LLMUsage.from_provider(usage.model_dump())
    assert normalized is not None
    assert normalized.cost_known is False


def test_explicit_unknown_cost_flag_is_preserved():
    usage = LLMUsage.from_provider({"input_tokens": 40, "cost_usd": 0.25, "cost_known": False})

    assert usage is not None
    assert usage.cost_usd == 0.25
    assert usage.cost_known is False


@pytest.mark.parametrize("cost", [0, 0.0, 0.25])
def test_litellm_response_cost_supplies_missing_provider_price(cost):
    raw = SimpleNamespace(
        usage={"input_tokens": 40, "output_tokens": 10},
        _hidden_params={"response_cost": cost},
    )
    usage = _extract_usage(raw)

    assert usage is not None
    assert usage.cost_usd == cost
    assert usage.cost_known is True


@pytest.mark.parametrize("cost", [True, "unknown", float("nan"), float("inf"), -1.0])
def test_invalid_litellm_cost_does_not_turn_missing_price_into_known_zero(cost):
    raw = SimpleNamespace(
        usage={"input_tokens": 40, "output_tokens": 10},
        _hidden_params={"response_cost": cost},
    )
    usage = _extract_usage(raw)

    assert usage is not None
    assert usage.input_tokens == 40
    assert usage.cost_usd == 0.0
    assert usage.cost_known is False


def test_missing_provider_usage_is_not_replaced_by_cost_metadata():
    raw = SimpleNamespace(usage=None, _hidden_params={"response_cost": 0.25})

    assert _extract_usage(raw) is None


@pytest.mark.parametrize("cost", [0.0, 0.25])
@pytest.mark.parametrize("key", ["cost", "cost_usd"])
def test_provider_price_takes_precedence_over_litellm_bill(key, cost):
    raw = SimpleNamespace(
        usage={"input_tokens": 40, key: cost},
        _hidden_params={"response_cost": 1.5},
    )

    usage = _extract_usage(raw)

    assert usage is not None
    assert usage.cost_usd == cost
    assert usage.cost_known is True


@pytest.mark.parametrize("upstream_cost", [None, 0.0, 0.25])
@pytest.mark.parametrize("unavailable", [False, True])
def test_litellm_stamped_cost_retains_upstream_provenance(upstream_cost, unavailable):
    raw = SimpleNamespace(
        usage={"input_tokens": 40, "cost": 1.5},
        _hidden_params={
            "response_cost": 1.5,
            "upstream_usage_cost": upstream_cost,
            "aster_cost_unavailable": unavailable,
        },
    )

    usage = _extract_usage(raw)

    assert usage is not None
    assert usage.input_tokens == 40
    expected = upstream_cost if upstream_cost is not None else (None if unavailable else 1.5)
    assert usage.cost_known is (expected is not None)
    assert usage.cost_usd == (expected or 0.0)
    assert raw.usage["cost"] == 1.5


def test_litellm_stamp_does_not_hide_explicit_cost_usd():
    raw = SimpleNamespace(
        usage={"input_tokens": 40, "cost": 1.5, "cost_usd": 0.25},
        _hidden_params={"response_cost": 1.5, "upstream_usage_cost": None},
    )

    usage = _extract_usage(raw)

    assert usage is not None
    assert usage.cost_usd == 0.25
    assert usage.cost_known is True


@pytest.mark.parametrize(
    ("provider_usage", "reported"),
    [({}, False), ({"output_tokens": 10}, False), ({"input_tokens": 0}, True)],
)
def test_provider_input_presence_survives_normalization_and_serialization(provider_usage, reported):
    usage = LLMUsage.from_provider(provider_usage)

    assert usage is not None
    assert usage.input_tokens_reported is reported
    assert LLMUsage.model_validate_json(usage.model_dump_json()).input_tokens_reported is reported
