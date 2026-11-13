# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Turn-scoped steering seam shared by coding strategies and interactive hosts."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import asyncio

    from aster.strategies.base import RuntimeServices


class SteeringControl(Protocol):
    wakeup: asyncio.Event

    @property
    def pending(self) -> bool: ...

    def matches(self, runtime: RuntimeServices) -> bool: ...

    def completed(self, result: Any) -> None: ...


_current: ContextVar[SteeringControl | None] = ContextVar("aster_coding_steering", default=None)


@contextmanager
def steering_scope(control: SteeringControl | None) -> Iterator[None]:
    """Bind to one dispatcher task, excluding other sessions and later prompts."""
    token = _current.set(control)
    try:
        yield
    finally:
        _current.reset(token)


def current_steering(runtime: RuntimeServices) -> SteeringControl | None:
    control = _current.get()
    return control if control is not None and control.matches(runtime) else None
