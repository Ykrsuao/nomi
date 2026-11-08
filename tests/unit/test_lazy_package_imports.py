# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``import nooa`` does not load the strategies or LiteLLM; the names still resolve."""

import subprocess
import sys

import pytest

_HEAVY = ("litellm", "nooa.strategies", "nooa.unifiedllm")


@pytest.mark.parametrize("statement", ["import nooa", "import nooa.runtime"])
def test_package_import_does_not_load_the_llm_stack(statement):
    code = f"{statement}\nimport sys\nprint([m for m in {_HEAVY!r} if m in sys.modules])\n"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "[]"


@pytest.mark.parametrize(
    "statement",
    [
        "from nooa.unifiedllm import FakeLLMClient, LLMResponse, Tool",
        "from nooa.strategies import CodeActStrategy, CodeActV2",
        "from nooa.runtime.sandbox.guest import return_function; return_function(int)",
        "from nooa.interactive import Done, InteractiveAgent",
    ],
)
def test_execution_imports_do_not_load_model_transport(statement):
    # Each microVM starts a fresh interpreter. Importing the execution types
    # must not initialize the provider SDK before any generated code can run.
    code = f"{statement}\nimport sys\nassert 'litellm' not in sys.modules\n"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("client", ["CompletionClient", "ResponsesClient"])
def test_real_clients_initialize_provider_settings_on_first_use(client):
    code = f"""
import asyncio
import sys
from nooa.unifiedllm import {client}
assert 'litellm' not in sys.modules
async def check():
    async with {client}('openai/gpt-4o-mini', api_key='test-key'):
        import litellm
        assert litellm.modify_params is True
        assert litellm.disable_aiohttp_transport is True
        from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
        config = OpenAIGPTConfig()
        messages = [{{'role': 'system', 'content': 'Hi', 'cache_control': {{'type': 'ephemeral'}}}}]
        result, _ = config.remove_cache_control_flag_from_messages_and_tools(
            model='openai/aws/anthropic/bedrock-claude-sonnet-4-5-v1', messages=messages,
        )
        assert result[0]['cache_control'] == {{'type': 'ephemeral'}}
asyncio.run(check())
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_lazy_names_resolve_to_the_defining_objects():
    import nooa
    import nooa.llm_config
    import nooa.strategies
    import nooa.unifiedllm
    from nooa import CodeActStrategy, LLMResponse, llm_config_chain, set_default_strategy

    assert CodeActStrategy is nooa.strategies.CodeActStrategy
    assert set_default_strategy is nooa.strategies.set_default_strategy
    assert LLMResponse is nooa.unifiedllm.LLMResponse
    assert llm_config_chain is nooa.llm_config.llm_config_chain
    for name in nooa._LAZY:
        assert name in nooa.__all__
        assert getattr(nooa, name) is not None


def test_unknown_attribute_still_raises():
    import nooa

    with pytest.raises(AttributeError):
        nooa.does_not_exist  # noqa: B018
