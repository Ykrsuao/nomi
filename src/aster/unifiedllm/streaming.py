# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request-scoped, observational access to provider stream events.

Collectors still return the complete canonical response. Hosts decide which
fields are user-facing; raw model prose, reasoning and tool arguments are not
automatically displayed. A closed scope also disables callbacks in provider
tasks that outlive a cancelled caller.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any


@dataclass
class _Observer:
    callback: Callable[[str, Any], None]
    active: bool = True


_observer: ContextVar[_Observer | None] = ContextVar("aster_provider_stream", default=None)


@contextmanager
def observe_stream(callback: Callable[[str, Any], None]) -> Iterator[None]:
    observer = _Observer(callback)
    token = _observer.set(observer)
    try:
        yield
    finally:
        observer.active = False
        _observer.reset(token)


def stream_observed() -> bool:
    observer = _observer.get()
    return observer is not None and observer.active


def emit_stream(kind: str, value: Any = None) -> None:
    observer = _observer.get()
    if observer is not None and observer.active:
        observer.callback(kind, value)
