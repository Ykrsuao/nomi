# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime components: actor loop, prompts, PLAN, REPL, context.

Core runtime contains NO OpenTelemetry code - only the hooks protocol.
For tracing, use: from openinference_instrumentation_aster import enable_tracing

Middleware types are imported directly from ``aster.runtime.middleware``::

    from aster.runtime.middleware import LLMCallContext, LLMCallMiddleware
"""

from aster.agentdoc import FileBackedTruncatingStringIO, TruncatingStringIO
from aster.config.truncation_config import TruncationConfig
from aster.runtime.actor import ActorRuntime
from aster.runtime.event_manager import EventManager
from aster.runtime.event_query import EventQuery
from aster.runtime.events import EventsApi
from aster.runtime.hooks import (
    CompositeInstrumentationHooks,
    InstrumentationHooks,
    compose_hooks,
    get_hooks,
    hooks_scope,
    set_hooks,
)
from aster.runtime.media_capture import show
from aster.runtime.pprint import pprint

__all__ = [
    "ActorRuntime",
    # Event system
    "EventManager",
    "EventQuery",
    "EventsApi",
    # Hook-based instrumentation protocol
    "InstrumentationHooks",
    "CompositeInstrumentationHooks",
    "compose_hooks",
    "hooks_scope",
    "set_hooks",
    "get_hooks",
    # Truncation system
    "TruncationConfig",
    "TruncatingStringIO",
    "FileBackedTruncatingStringIO",
    "pprint",
    "show",
]
