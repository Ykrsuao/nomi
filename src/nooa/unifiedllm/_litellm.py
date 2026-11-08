# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Initialize the provider SDK only when a transport operation needs it.

Execution workers import the public LLM types and strategies but send their
model requests through the host. Keeping this module out of those imports
avoids loading the provider stack in every new guest interpreter. Python's
module import lock also keeps the one-time SDK configuration atomic.
"""

import logging

import litellm

from nooa.unifiedllm.unifiedllm import _is_anthropic_model

logger = logging.getLogger(__name__)

# Bedrock/Anthropic reject messages with tool_call blocks without tools=.
litellm.modify_params = True

# NOOA owns its httpx clients; LiteLLM's default aiohttp sessions otherwise
# remain open at shutdown and emit ResourceWarnings.
litellm.disable_aiohttp_transport = True


def _apply_cache_control_preserve_patch():
    """Preserve cache_control for Anthropic behind OpenAI-compatible routes."""
    try:
        from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

        original_remove = OpenAIGPTConfig.remove_cache_control_flag_from_messages_and_tools

        def patched_remove(self, model, messages, tools=None):
            if _is_anthropic_model(model):
                return messages, tools
            return original_remove(self, model, messages, tools)

        OpenAIGPTConfig.remove_cache_control_flag_from_messages_and_tools = patched_remove
        logger.debug("Applied cache_control preserve patch for Anthropic models")
    except (ImportError, AttributeError) as exc:
        logger.warning("Could not apply cache_control preserve patch: %s", exc)


_apply_cache_control_preserve_patch()
