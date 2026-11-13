# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in long-term memory subsystem for Aster.

A brain-inspired, fully *additive* memory add-on: spontaneous association
(similarity recall injected each turn), deliberate recall (conscious tools),
encoding (write on important events), reflection (offline consolidation), and
forgetting (online decay + offline pruning).

Quick start::

    from aster import Agent
    from aster_memory import MemoryConfig, MemoryManager, MemoryToolsMixin

    class MyAgent(MemoryToolsMixin, Agent, llm=my_llm):
        async def work(self, task: str) -> str:
            # Do {task}. Use self.recall(...) to consult memory.
            ...

    agent = MyAgent()
    MemoryManager.install(agent, config=MemoryConfig(enabled=True))
    agent.remember("Deploy with `make ship`.", type="skill", importance="HIGH")
"""

from aster_memory.config import (
    EmbeddingConfig,
    ForgetPolicy,
    MemoryConfig,
    ObservabilityConfig,
    ReflectionPolicy,
    RetrievalConfig,
    ScoringWeights,
    SpontaneousConfig,
    VectorConfig,
    WritePolicy,
)
from aster_memory.descriptors import (
    TODO_STATUSES,
    ladder,
    to_label,
    to_numeric,
    to_status,
)
from aster_memory.embeddings import (
    Embedder,
    HashingEmbedder,
    LiteLLMEmbedder,
    get_embedder,
)
from aster_memory.generative import llm_reasoner, llm_reconciler
from aster_memory.manager import MemoryManager, MemoryToolsMixin
from aster_memory.monitoring import (
    MemoryInjected,
    MemoryRecalled,
    MemoryStats,
    MemoryWritten,
    ReflectionCompleted,
    ReflectionStarted,
)
from aster_memory.observability import per_memory_usage, prune_eta, store_kpis
from aster_memory.references import ResolvedRef, capture, parse_ref, render, resolve
from aster_memory.schema import (
    ACCESS_CHANNELS,
    REF_KINDS,
    AccessRecord,
    Edge,
    EdgeType,
    Memory,
    MemoryRef,
    MemoryType,
)
from aster_memory.store import SCHEMA_VERSION, MemorySchemaError, MemoryStore
from aster_memory.vector_backends import (
    ChromaVectorIndex,
    NumpyVectorIndex,
    SqliteVecVectorIndex,
    VectorIndex,
    make_vector_index,
)

__all__ = [
    # schema
    "Memory",
    "MemoryType",
    "Edge",
    "EdgeType",
    "MemoryRef",
    "REF_KINDS",
    "AccessRecord",
    "ACCESS_CHANNELS",
    # references (pass-by-reference resolution)
    "ResolvedRef",
    "parse_ref",
    "capture",
    "resolve",
    "render",
    # config
    "MemoryConfig",
    "RetrievalConfig",
    "SpontaneousConfig",
    "ScoringWeights",
    "EmbeddingConfig",
    "VectorConfig",
    "WritePolicy",
    "ReflectionPolicy",
    "ForgetPolicy",
    "ObservabilityConfig",
    # embeddings
    "Embedder",
    "HashingEmbedder",
    "LiteLLMEmbedder",
    "get_embedder",
    # store
    "MemoryStore",
    "MemorySchemaError",
    "SCHEMA_VERSION",
    # vector backends
    "VectorIndex",
    "NumpyVectorIndex",
    "SqliteVecVectorIndex",
    "ChromaVectorIndex",
    "make_vector_index",
    # verbal descriptors (agent-facing vocabulary)
    "ladder",
    "to_numeric",
    "to_label",
    "to_status",
    "TODO_STATUSES",
    # manager / integration
    "MemoryManager",
    "MemoryToolsMixin",
    # monitoring / observability
    "MemoryStats",
    "llm_reasoner",
    "llm_reconciler",
    "per_memory_usage",
    "store_kpis",
    "prune_eta",
    "MemoryWritten",
    "MemoryRecalled",
    "MemoryInjected",
    "ReflectionCompleted",
    "ReflectionStarted",
]
