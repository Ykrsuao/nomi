# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Initialize the provider SDK only when a transport operation needs it.

Execution workers import the public LLM types and strategies but send their
model requests through the host. Keeping this module out of those imports
avoids loading the provider stack in every new guest interpreter. Python's
module import lock also keeps the one-time SDK configuration atomic.
"""

import logging
from collections.abc import Mapping
from urllib.parse import urlsplit

import litellm

from aster.unifiedllm.unifiedllm import _is_anthropic_model

logger = logging.getLogger(__name__)

# Bedrock/Anthropic reject messages with tool_call blocks without tools=.
litellm.modify_params = True

# ASTER owns its httpx clients; LiteLLM's default aiohttp sessions otherwise
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


def _apply_responses_explicit_cache_preserve_patch():
    """Honor an explicit cache declaration on a custom OpenAI deployment."""
    try:
        from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
        from litellm.types.router import GenericLiteLLMParams

        original_reads = OpenAIResponsesAPIConfig._reads_prompt_cache_breakpoints

        def patched_reads(
            self: OpenAIResponsesAPIConfig,
            model: str,
            litellm_params: GenericLiteLLMParams,
            prompt_cache_options: object,
        ) -> bool:
            if (
                self.custom_llm_provider == "openai"
                and isinstance(prompt_cache_options, Mapping)
                and prompt_cache_options.get("mode") == "explicit"
            ):
                if isinstance(litellm_params, Mapping):
                    api_base = litellm_params.get("api_base") or litellm_params.get("base_url")
                else:
                    api_base = getattr(litellm_params, "api_base", None) or getattr(
                        litellm_params, "base_url", None
                    )
                if api_base is None or isinstance(api_base, str):
                    # Use the SDK's wire endpoint resolution, including globals
                    # and OPENAI_BASE_URL / OPENAI_API_BASE environment defaults.
                    endpoint = urlsplit(self.get_complete_url(api_base, {}))
                    host = (endpoint.hostname or "").rstrip(".")
                    # A custom deployment can use an alias whose catalogue entry
                    # describes a different cache dialect. Its explicit opt-in wins.
                    if (
                        endpoint.scheme in {"http", "https"}
                        and host
                        and host != "api.openai.com"
                        and not host.endswith(".api.openai.com")
                    ):
                        return True
            return original_reads(self, model, litellm_params, prompt_cache_options)

        OpenAIResponsesAPIConfig._reads_prompt_cache_breakpoints = patched_reads
    except (ImportError, AttributeError) as exc:
        logger.warning("Could not apply Responses explicit cache preserve patch: %s", exc)


_apply_responses_explicit_cache_preserve_patch()
