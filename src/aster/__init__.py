# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
aster - Code-Generating Agent Orchestration System

A minimal viable runtime for agent orchestration with event sourcing,
serialized execution, and complete transparency.
"""

import logging as _logging
import typing as _typing

from aster._version import __version__

# ---------------------------------------------------------------------------
# Library logging: add NullHandler so applications that don't configure
# logging never see "No handlers could be found for logger 'aster'".
# This is the only handler a library should ever add (see Python docs).
# ---------------------------------------------------------------------------
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

# Export core types
# Export agent and decorators

from aster._logging import enable_logging  # noqa: E402
from aster._visible import visible  # noqa: E402
from aster.agent import Agent  # noqa: E402
from aster.agentdoc import hidden, spec  # noqa: E402
from aster.context_blocks import Context, ContextWindowStats, DynamicContext  # noqa: E402
from aster.decisions import (  # noqa: E402
    BooleanDecision,
    ChoiceDecision,
    Criteria,
    Decision,
    DecisionClient,
    DecisionModel,
    DecisionModelRequiredError,
    DecisionRequest,
    DecisionResponse,
    Instructions,
    ScoreDecision,
    Threshold,
    UnifiedDecisionModel,
)
from aster.decorators import strategy  # noqa: E402

# Export errors
from aster.errors import (  # noqa: E402
    AsterError,
    AsterRuntimeError,
    DynamicMethodAdditionError,
    GenerationError,
    RestrictedCodeError,
    SerializationError,
    SnapshotNotFoundError,
    StorageNotConfiguredError,
    ValidationError,
)
from aster.library_manager import LibraryManager  # noqa: E402
from aster.media import Audio, File, Image, Media, Video  # noqa: E402
from aster.metaclass import AgentMeta, no_trace  # noqa: E402

# Export prompt inspection utilities
from aster.prompts import PromptData, build_prompt_data, print_prompt  # noqa: E402
from aster.runtime.channels import Channel, QueueManager, QueueOutput  # noqa: E402

# Export runtime API classes
from aster.runtime.context import ContextApi  # noqa: E402
from aster.runtime.context_manager import ContextManager  # noqa: E402

# Export event filtering
from aster.runtime.event_query import EventQuery  # noqa: E402
from aster.runtime.events import EventsApi  # noqa: E402
from aster.skill import Skill, TextSkill, get_slash_commands, slash_command  # noqa: E402
from aster.skill_registry import skill_from_module  # noqa: E402

# Export storage
from aster.storage import StorageManager  # noqa: E402
from aster.strategy_validation import (  # noqa: E402
    InvariantError,
    MethodPostcondition,
    MethodPrecondition,
)
from aster.token_counter import char_approximate_token_counter  # noqa: E402

# Names loaded on first use. The strategies import CodeAct and the LLM
# client types; llm_config_chain touches the llm_config / paths machinery.
# LiteLLM itself loads only for provider operations. ``import aster`` stays cheap, and
# ``from aster import CodeActStrategy`` still works.
# NOTE: ReflexionStrategy is experimental; it resolves to the
# warning-emitting factory in aster.experimental (below).
if _typing.TYPE_CHECKING:  # the same names, for type checkers and IDEs
    from aster.llm_config import llm_config_chain
    from aster.llm_types import LLMResponse
    from aster.strategies import (
        CodeActStrategy,
        CodeActV2,
        DecideStrategy,
        GenerationStrategy,
        InspectInputsPrefill,
        PredictStrategy,
        TextOnlyResponseAction,
        TextOnlyResponseContext,
        TextOnlyResponseHandler,
        get_default_strategy,
        retry_text_only_response,
        return_text_as_result,
        set_default_strategy,
    )

_LAZY = {
    "CodeActStrategy": "aster.strategies",
    "CodeActV2": "aster.strategies",
    "DecideStrategy": "aster.strategies",
    "GenerationStrategy": "aster.strategies",
    "InspectInputsPrefill": "aster.strategies",
    "PredictStrategy": "aster.strategies",
    "TextOnlyResponseAction": "aster.strategies",
    "TextOnlyResponseContext": "aster.strategies",
    "TextOnlyResponseHandler": "aster.strategies",
    "get_default_strategy": "aster.strategies",
    "retry_text_only_response": "aster.strategies",
    "return_text_as_result": "aster.strategies",
    "set_default_strategy": "aster.strategies",
    "LLMResponse": "aster.llm_types",  # the class aster.unifiedllm re-exports
    "llm_config_chain": "aster.llm_config",
}


def __getattr__(name):
    module_name = _LAZY.get(name)
    if module_name is not None:
        import importlib

        value = getattr(importlib.import_module(module_name), name)
        globals()[name] = value
        return value
    # Experimental strategies: route through the warning factories so that
    # `from aster import ReflexionStrategy; ReflexionStrategy()`
    # emits the same FutureWarning as importing from aster.experimental.
    if name == "ReflexionStrategy":
        from aster import experimental

        return getattr(experimental, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "llm_config_chain",  # Lazy (see _LAZY)
    # Types
    "ContextWindowStats",  # Re-exported from context_blocks
    "Context",  # Re-exported from context_blocks
    "DynamicContext",  # Re-exported from context_blocks (deprecated, use Context)
    "EventQuery",  # Event filtering configuration
    "ContextApi",  # LLM-facing context API wrapper (Skill)
    "ContextManager",  # Context block state backend
    "EventsApi",  # Runtime events query API
    "Channel",  # Named queue/event channel for agent input
    "QueueManager",  # Channel registry + race + status
    "QueueOutput",  # Event emitted by event-mode channels
    "LLMResponse",  # Re-exported from unifiedllm
    # Strategies
    "GenerationStrategy",
    "CodeActStrategy",
    "CodeActV2",
    "TextOnlyResponseAction",
    "TextOnlyResponseContext",
    "TextOnlyResponseHandler",
    "retry_text_only_response",
    "return_text_as_result",
    "ReflexionStrategy",
    "PredictStrategy",
    "DecideStrategy",
    "get_default_strategy",
    "set_default_strategy",
    # Method validators
    "InvariantError",
    "MethodPrecondition",
    "MethodPostcondition",
    # Prefill plugins
    "InspectInputsPrefill",
    # Prompt inspection
    "print_prompt",
    "build_prompt_data",
    "PromptData",
    # Media types
    "Media",
    "Image",
    "Audio",
    "Video",
    "File",
    # Agent and decorators
    "Agent",
    "Skill",
    "TextSkill",
    "slash_command",
    "get_slash_commands",
    "skill_from_module",
    "LibraryManager",
    "strategy",
    "no_trace",
    "AgentMeta",
    # Decisions
    "BooleanDecision",
    "ChoiceDecision",
    "Criteria",
    "Decision",
    "DecisionClient",
    "DecisionModel",
    "DecisionModelRequiredError",
    "DecisionRequest",
    "DecisionResponse",
    "Instructions",
    "ScoreDecision",
    "Threshold",
    "UnifiedDecisionModel",
    # Logging
    "enable_logging",
    # Visibility
    "hidden",
    "spec",
    "visible",
    # Errors
    "AsterError",
    "GenerationError",
    "ValidationError",
    "RestrictedCodeError",
    "AsterRuntimeError",
    "DynamicMethodAdditionError",
    "SerializationError",
    "SnapshotNotFoundError",
    "StorageNotConfiguredError",
    # Storage
    "StorageManager",
    # Token counting
    "char_approximate_token_counter",
]

# Install debug handler by default (zero overhead until SIGUSR2 received)
# Usage: kill -USR2 <pid> → dumps traceback + cell code to debug_dump_<pid>.txt in cwd
from aster.runtime.debug_handler import install_debug_handler  # noqa: E402

install_debug_handler()
