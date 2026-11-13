# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in startup timings, kept off the ACP protocol's stdout stream."""

import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from time import perf_counter, process_time

_STARTED_AT = perf_counter()
_CPU_STARTED_AT = process_time()


def mark(
    stage: str,
    *,
    event: str = "ready",
    duration_s: float | None = None,
    cpu_s: float | None = None,
) -> None:
    """Emit only stage names and timings; never include configuration or payloads."""
    if os.getenv("ASTER_STARTUP_TIMING") != "1":
        return
    record = {
        "pid": os.getpid(),
        "stage": stage,
        "event": event,
        "elapsed_s": round(perf_counter() - _STARTED_AT, 3),
        "cpu_elapsed_s": round(process_time() - _CPU_STARTED_AT, 3),
    }
    if duration_s is not None:
        record["duration_s"] = round(duration_s, 3)
    if cpu_s is not None:
        record["cpu_s"] = round(cpu_s, 3)
    with suppress(OSError, ValueError):
        print("[aster-startup] " + json.dumps(record), file=sys.stderr, flush=True)


@contextmanager
def phase(stage: str) -> Iterator[None]:
    """Record both boundaries so an interrupted or stuck phase remains visible."""
    if os.getenv("ASTER_STARTUP_TIMING") != "1":
        yield
        return
    started = perf_counter()
    cpu_started = process_time()
    mark(stage, event="start")
    try:
        yield
    except BaseException:
        mark(
            stage,
            event="failed",
            duration_s=perf_counter() - started,
            cpu_s=process_time() - cpu_started,
        )
        raise
    else:
        mark(
            stage,
            event="complete",
            duration_s=perf_counter() - started,
            cpu_s=process_time() - cpu_started,
        )
