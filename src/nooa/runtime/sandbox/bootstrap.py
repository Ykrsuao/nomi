# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted host-to-guest namespace serialization; never used for guest replies."""

from __future__ import annotations

import contextvars
import io
import threading
from types import SimpleNamespace
from typing import Any


def _new_lock() -> Any:
    return threading.Lock()


def _new_rlock() -> Any:
    return threading.RLock()


def _new_thread_local(state: dict[str, Any]) -> Any:
    local = threading.local()
    local.__dict__.update(state)
    return local


def _new_contextvar(name: str, value: Any) -> Any:
    return contextvars.ContextVar(name, default=value)


def _reduce_contextvar(variable: Any) -> Any:
    value = variable.get(None)
    if not isinstance(value, (str, bytes, int, float, bool, type(None))):
        value = None
    return _new_contextvar, (variable.name, value)


def _restore_agent_class(cls: Any, state: Any) -> Any:
    """Hydrate trusted class definitions without treating them as generated edits."""
    attributes, _ = state
    registry = attributes.get("_abc_impl")
    for name, value in attributes.items():
        if name != "_abc_impl":
            type.__setattr__(cls, name, value)
    if registry is not None:
        for subclass in registry:
            cls.register(subclass)
    return cls


def dumps_namespace(value: Any, *, modules: Any = (), path_map: Any = None) -> bytes:
    import cloudpickle

    class NamespacePickler(cloudpickle.CloudPickler):
        dispatch_table = cloudpickle.CloudPickler.dispatch_table.copy()
        dispatch_table[type(threading.Lock())] = lambda _: (_new_lock, ())
        dispatch_table[type(threading.RLock())] = lambda _: (_new_rlock, ())
        dispatch_table[threading.local] = lambda local: (_new_thread_local, (dict(local.__dict__),))
        dispatch_table[contextvars.ContextVar] = _reduce_contextvar

        def reducer_override(self, value: Any) -> Any:
            from nooa.metaclass import AgentMeta

            reduced = super().reducer_override(value)
            if isinstance(value, AgentMeta) and isinstance(reduced, tuple) and len(reduced) == 6:
                return (*reduced[:5], _restore_agent_class)
            return reduced

    if path_map is not None:
        from pathlib import Path, PosixPath, PurePosixPath, PureWindowsPath, WindowsPath

        from nooa.runtime.sandbox.paths import guest_path

        def reduce_path(path: Any) -> Any:
            return guest_path, (path_map.to_guest(str(path)),)

        for kind in (Path, PosixPath, WindowsPath, PurePosixPath, PureWindowsPath):
            NamespacePickler.dispatch_table[kind] = reduce_path

    stream = io.BytesIO()
    registered = cloudpickle.list_registry_pickle_by_value()
    added = []
    try:
        for module in modules:
            if module.__name__ not in registered:
                cloudpickle.register_pickle_by_value(module)
                added.append(module)
        NamespacePickler(stream).dump(value)
        return stream.getvalue()
    finally:
        for module in added:
            cloudpickle.unregister_pickle_by_value(module)


def describe_call(call: Any) -> Any:
    if call is None:
        return SimpleNamespace(return_type=object)
    return SimpleNamespace(
        return_type=call.return_type,
        kwargs=call.kwargs,
        method_name=call.method_name,
    )
