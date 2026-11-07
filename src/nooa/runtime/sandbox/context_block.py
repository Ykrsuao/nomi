# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render the active sandbox constraints into an agent-facing context block.

Only guardrails that are actually in force are listed, so the agent adapts to
exactly what it faces (short cells under a tight timeout, no writes outside the
workspace, no network, picklable returns).
"""

from __future__ import annotations

from nooa.runtime.sandbox.config import SandboxConfig


def render_sandbox_block(config: SandboxConfig, *, cell_timeout: float | None, degraded=()) -> str:
    """Render the active sandbox constraints (block body only).

    The context formatter wraps this in a ``<sandbox>...</sandbox>`` envelope, so
    this returns just the inner text.
    """
    lines: list[str] = [
        "Your code runs in a persistent Python interpreter inside an isolated microVM.",
    ]

    if cell_timeout:
        lines.append(
            f"- Wall-clock: each cell is hard-killed {cell_timeout:.0f}s after it starts "
            "(a killed cell loses its output). Keep cells short and return partial results."
        )
    if config.max_cpu_seconds:
        lines.append(
            f"- CPU: {config.max_cpu_seconds}s of CPU time per worker; a runaway "
            "compute loop is terminated."
        )
    if config.max_memory_mb:
        lines.append(
            f"- Memory: {config.max_memory_mb} MiB of allocation headroom above initialization; allocating past it raises MemoryError."
        )
    if config.filesystem and "filesystem" not in degraded:
        from nooa._platform import is_native_windows

        windows = is_native_windows()
        writable = ("/workspace" if windows else config.workspace) if config.workspace else "(none)"
        rw = [
            (f"/allowed/{i}" if windows else r.path)
            for i, r in enumerate(config.allow)
            if r.access == "read_write"
        ]
        ro = [
            (f"/allowed/{i}" if windows else r.path)
            for i, r in enumerate(config.allow)
            if r.access == "read"
        ]
        lines.append(
            f"- Filesystem: writable path(s): {writable}"
            + (f", {', '.join(rw)}" if rw else "")
            + ". Other guest files are read-only or inaccessible"
            + (f" (extra readable: {', '.join(ro)})" if ro else "")
            + "."
        )
    if not config.network:
        lines.append(
            "- Network: disabled. Opening a socket to the internet raises PermissionError; "
            "do not attempt downloads or API calls from a cell."
        )
    if degraded:
        lines.append(
            "- Unavailable optional limits: "
            + ", ".join(degraded)
            + ". The microVM isolation remains active."
        )
    lines.append(
        "- Values returned from a cell must be picklable (numbers, str, list, dict, ndarray). "
        "Keep live objects in the namespace and return a summary instead."
    )
    lines.append(
        "- self.<attr> reads a fresh copy from the agent, so in-place mutation like "
        "self.items.append(x) is NOT persisted — call a method (self.record(x)) or "
        "reassign (self.items = self.items + [x]) instead. return_result(value) must "
        "pass the value itself, not a variable name. The Out[n] history and any "
        "caller-seeded variables are not available here; recompute what you need."
    )

    return "\n".join(lines)
