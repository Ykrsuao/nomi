# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process-backed, OS-enforced sandbox for CodeAct cell execution.

Public surface:

* :class:`~aster.runtime.sandbox.config.SandboxConfig` — declarative guardrails.
* :class:`~aster.runtime.sandbox.executor.SandboxedExecutor` — the parent-side
  process backend that runs cells in a locked-down worker.
* guard errors (:class:`CellTimeoutError`, :class:`CellMemoryError`, ...).
"""

from __future__ import annotations

from aster.runtime.sandbox.config import FileRule, SandboxConfig
from aster.runtime.sandbox.errors import (
    CellMemoryError,
    CellSerializationError,
    CellTimeoutError,
    SandboxError,
    SandboxExecutionError,
    SandboxUnavailable,
    WorkerDiedError,
)

__all__ = [
    "SandboxConfig",
    "FileRule",
    "SandboxError",
    "SandboxExecutionError",
    "SandboxUnavailable",
    "CellTimeoutError",
    "CellMemoryError",
    "CellSerializationError",
    "WorkerDiedError",
]
