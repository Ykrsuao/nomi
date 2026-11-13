# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark agent and Harbor runner for the ASTER framework.

Benchmark-agnostic coding agents for SWE-bench and Terminal-Bench, the
``aster-harbor`` container runner, and trajectory/token analysis utilities.
"""

from __future__ import annotations

# Maps --agent-type CLI values to dotted import paths: "module:ClassName"
AGENT_CLASSES: dict[str, str] = {
    # Unified SWE-bench + Terminal-Bench baseline.
    "bench": "aster_bench.bench_agent:BenchAgent",
    "rlm": "aster_bench.rlm_bench_agent:RLMBenchAgent",
}

__all__ = ["AGENT_CLASSES"]
