# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test agent without any asyncio imports."""

from aster import Agent
from aster.unifiedllm import FakeLLMClient


class AgentNoAsyncioImport(Agent, llm=FakeLLMClient()):
    """Agent without any asyncio imports."""

    def run(self):
        return "done"
